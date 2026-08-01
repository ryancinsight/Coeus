//! Damped Gauss-Newton (Levenberg-Marquardt) for small dense problems.

use coeus_core::Scalar;
use leto::{Array1, Array2};
use leto_ops::{cholesky_solve, RealScalar};

use super::config::{LeastSquaresReport, LevenbergMarquardtConfig, Termination};
use super::problem::{LeastSquaresProblem, ProblemError};

/// Failure modes of the solver itself, as distinct from the problem's.
#[derive(Debug, thiserror::Error)]
pub enum SolverError {
    /// The initial parameter vector length disagrees with the problem.
    #[error("expected {expected} parameters, got {actual}")]
    ParameterCount {
        /// What the problem declares.
        expected: usize,
        /// What the caller supplied.
        actual: usize,
    },

    /// A least-squares problem needs at least as many residuals as parameters.
    ///
    /// With fewer, `JᵀJ` is singular by construction and the fit is
    /// underdetermined: damping would mask that rather than solve it.
    #[error("underdetermined: {residuals} residuals for {parameters} parameters")]
    Underdetermined {
        /// Residual count.
        residuals: usize,
        /// Parameter count.
        parameters: usize,
    },

    /// The problem reported an unrecoverable evaluation failure.
    #[error(transparent)]
    Problem(#[from] ProblemError),

    /// The damped normal equations stayed unsolvable up to the damping cap.
    ///
    /// Damping drives `JᵀJ + λ·diag(JᵀJ)` toward diagonal dominance, so this
    /// means the Jacobian is degenerate rather than merely ill-conditioned.
    #[error("normal equations remained singular at maximum damping")]
    Singular,

    /// The model produced a non-finite residual or Jacobian entry.
    #[error("non-finite value in {evaluation} at iteration {iteration}")]
    NonFinite {
        /// Which evaluation produced it.
        evaluation: &'static str,
        /// Iteration it appeared at.
        iteration: usize,
    },
}

/// Scalars this solver operates on.
///
/// Composes coeus's element vocabulary with leto's dense-linear-algebra
/// vocabulary, the same bridge `AttentionScalar` uses in `coeus-leto`: the
/// solver's arithmetic is coeus's, and the normal-equations solve is leto's.
pub trait LeastSquaresScalar: Scalar + RealScalar {}

impl<T: Scalar + RealScalar> LeastSquaresScalar for T {}

/// Damping is abandoned past this factor; beyond it the step is numerically a
/// zero-length gradient step and further increases cannot recover a solve.
const MAX_DAMPING_EXPONENT: u32 = 30;

/// Solve `min 0.5‖r(p)‖²` by damped Gauss-Newton from `initial_parameters`.
///
/// Each iteration solves the damped normal equations
///
/// ```text
/// (JᵀJ + λ · diag(JᵀJ)) δ = -Jᵀr
/// ```
///
/// and accepts `p + δ` when it reduces the cost. Damping is scaled by
/// `diag(JᵀJ)` rather than the identity (Marquardt's modification), which makes
/// the step invariant to rescaling of individual parameters — a diffusion model
/// mixing diffusivities near `1e-3` with signal amplitudes near `1e3` is
/// exactly the badly-scaled case that motivates it.
///
/// Convergence is decided by the derived criteria in
/// [`LevenbergMarquardtConfig`]; the iteration cap is a runaway guard and is
/// reported as a non-converged [`Termination::IterationLimit`].
///
/// # Errors
///
/// [`SolverError`] for a malformed problem, an unrecoverable evaluation
/// failure, a non-finite model value, or a Jacobian degenerate enough that
/// damping cannot produce a solvable system.
pub fn levenberg_marquardt<T, P>(
    problem: &P,
    initial_parameters: &[T],
    config: &LevenbergMarquardtConfig<T>,
) -> Result<LeastSquaresReport<T>, SolverError>
where
    T: LeastSquaresScalar,
    P: LeastSquaresProblem<T>,
{
    let parameter_count = problem.parameter_count();
    let residual_count = problem.residual_count();

    if initial_parameters.len() != parameter_count {
        return Err(SolverError::ParameterCount {
            expected: parameter_count,
            actual: initial_parameters.len(),
        });
    }
    if residual_count < parameter_count {
        return Err(SolverError::Underdetermined {
            residuals: residual_count,
            parameters: parameter_count,
        });
    }

    let mut parameters = initial_parameters.to_vec();
    let mut residuals = vec![T::zero(); residual_count];
    let mut jacobian = vec![T::zero(); residual_count * parameter_count];
    let mut trial_parameters = vec![T::zero(); parameter_count];
    let mut trial_residuals = vec![T::zero(); residual_count];

    problem.residuals(&parameters, &mut residuals)?;
    check_finite(&residuals, "residuals", 0)?;
    let mut cost = half_sum_of_squares(&residuals);
    let mut damping = config.initial_damping;

    for iteration in 0..config.max_iterations {
        problem.jacobian(&parameters, &mut jacobian)?;
        check_finite(&jacobian, "jacobian", iteration)?;

        let gradient = jacobian_transpose_times(&jacobian, &residuals, parameter_count);
        let gradient_norm = infinity_norm(&gradient);
        if gradient_norm <= config.gradient_tolerance {
            return Ok(report(
                parameters,
                cost,
                gradient_norm,
                iteration,
                Termination::GradientTolerance,
            ));
        }

        let normal_matrix = jacobian_transpose_jacobian(&jacobian, parameter_count);
        let Some(step) = solve_damped(
            &normal_matrix,
            &gradient,
            parameter_count,
            &mut damping,
            config,
        ) else {
            return Err(SolverError::Singular);
        };

        for (slot, (current, delta)) in trial_parameters
            .iter_mut()
            .zip(parameters.iter().zip(step.iter()))
        {
            *slot = *current + *delta;
        }

        // A domain error is a statement about the trial point, not the solve:
        // treat it exactly like a cost increase and let damping pull the next
        // trial back toward the last accepted point.
        let trial_cost = match problem.residuals(&trial_parameters, &mut trial_residuals) {
            Ok(()) if trial_residuals.iter().all(|value| is_finite(*value)) => {
                Some(half_sum_of_squares(&trial_residuals))
            }
            Ok(()) => None,
            Err(ProblemError::Domain { .. }) => None,
            Err(error) => return Err(error.into()),
        };

        let accepted = trial_cost.is_some_and(|trial| trial < cost);
        if !accepted {
            damping *= config.damping_increase;
            continue;
        }

        let step_norm = euclidean_norm(&step);
        let parameter_norm = euclidean_norm(&parameters);
        let previous_cost = cost;

        parameters.copy_from_slice(&trial_parameters);
        residuals.copy_from_slice(&trial_residuals);
        cost = trial_cost.unwrap_or(cost);
        damping = damping / config.damping_decrease;

        if step_norm <= config.step_tolerance * (parameter_norm + config.step_tolerance) {
            return Ok(report(
                parameters,
                cost,
                gradient_norm,
                iteration + 1,
                Termination::StepTolerance,
            ));
        }
        // Relative against the previous cost, so the test means the same thing
        // whether residuals are in millimetres or signal counts.
        if previous_cost > T::zero()
            && (previous_cost - cost) <= config.cost_tolerance * previous_cost
        {
            return Ok(report(
                parameters,
                cost,
                gradient_norm,
                iteration + 1,
                Termination::CostTolerance,
            ));
        }
    }

    problem.jacobian(&parameters, &mut jacobian)?;
    let gradient = jacobian_transpose_times(&jacobian, &residuals, parameter_count);
    let gradient_norm = infinity_norm(&gradient);
    Ok(report(
        parameters,
        cost,
        gradient_norm,
        config.max_iterations,
        Termination::IterationLimit,
    ))
}

/// Solve the damped system, raising damping until it is solvable.
///
/// Returns the step `δ`, or `None` once damping passes the cap without
/// producing a solvable system. `damping` is left at the value that worked, so
/// the caller's accept/reject update continues from it.
fn solve_damped<T: LeastSquaresScalar>(
    normal_matrix: &[T],
    gradient: &[T],
    parameter_count: usize,
    damping: &mut T,
    config: &LevenbergMarquardtConfig<T>,
) -> Option<Vec<T>> {
    let mut damped = vec![T::zero(); parameter_count * parameter_count];
    let negative_gradient: Vec<T> = gradient.iter().map(|value| T::zero() - *value).collect();

    let rhs = Array1::from_shape_vec([parameter_count], negative_gradient).ok()?;

    for _ in 0..MAX_DAMPING_EXPONENT {
        damped.copy_from_slice(normal_matrix);
        for index in 0..parameter_count {
            let diagonal = normal_matrix[index * parameter_count + index];
            // A structurally zero diagonal means the parameter has no local
            // influence; fall back to absolute damping so the row stays
            // solvable rather than scaling by nothing.
            let scale = if diagonal > T::zero() {
                diagonal
            } else {
                T::one()
            };
            damped[index * parameter_count + index] = diagonal + *damping * scale;
        }

        // The damped matrix is symmetric positive definite whenever damping is
        // positive and the Jacobian has full column rank, so Cholesky is the
        // correct factorization and its failure is the signal to raise damping.
        let matrix =
            Array2::from_shape_vec([parameter_count, parameter_count], damped.clone()).ok()?;
        if let Ok(step) = cholesky_solve(&matrix.view(), &rhs.view()) {
            if let Some(values) = step.as_slice() {
                return Some(values.to_vec());
            }
        }

        *damping *= config.damping_increase;
    }

    None
}

/// `JᵀJ`, row-major and symmetric.
fn jacobian_transpose_jacobian<T: Scalar>(jacobian: &[T], parameter_count: usize) -> Vec<T> {
    let residual_count = jacobian.len() / parameter_count;
    let mut product = vec![T::zero(); parameter_count * parameter_count];
    for row in 0..parameter_count {
        for column in row..parameter_count {
            let mut sum = T::zero();
            for residual in 0..residual_count {
                let base = residual * parameter_count;
                sum += jacobian[base + row] * jacobian[base + column];
            }
            product[row * parameter_count + column] = sum;
            product[column * parameter_count + row] = sum;
        }
    }
    product
}

/// `Jᵀr`, the gradient of `0.5‖r‖²`.
fn jacobian_transpose_times<T: Scalar>(
    jacobian: &[T],
    residuals: &[T],
    parameter_count: usize,
) -> Vec<T> {
    let mut gradient = vec![T::zero(); parameter_count];
    for (residual_index, residual) in residuals.iter().enumerate() {
        let base = residual_index * parameter_count;
        for (parameter, slot) in gradient.iter_mut().enumerate() {
            *slot += jacobian[base + parameter] * *residual;
        }
    }
    gradient
}

fn half_sum_of_squares<T: Scalar>(values: &[T]) -> T {
    let sum = values.iter().fold(T::zero(), |accumulator, value| {
        accumulator + *value * *value
    });
    sum / T::from_f64(2.0)
}

fn euclidean_norm<T: Scalar>(values: &[T]) -> T {
    values
        .iter()
        .fold(T::zero(), |accumulator, value| {
            accumulator + *value * *value
        })
        .sqrt_val()
}

fn infinity_norm<T: Scalar>(values: &[T]) -> T {
    values.iter().fold(T::zero(), |accumulator, value| {
        let magnitude = value.abs_val();
        if magnitude > accumulator {
            magnitude
        } else {
            accumulator
        }
    })
}

/// Whether `value` is finite, using only the ordering `Scalar` guarantees.
///
/// NaN fails every comparison, and an infinity fails the bound against the
/// largest finite magnitude the type round-trips.
fn is_finite<T: Scalar>(value: T) -> bool {
    let magnitude = value.abs_val();
    magnitude >= T::zero() && magnitude <= T::from_f64(f64::MAX)
}

fn check_finite<T: Scalar>(
    values: &[T],
    evaluation: &'static str,
    iteration: usize,
) -> Result<(), SolverError> {
    if values.iter().all(|value| is_finite(*value)) {
        Ok(())
    } else {
        Err(SolverError::NonFinite {
            evaluation,
            iteration,
        })
    }
}

fn report<T>(
    parameters: Vec<T>,
    cost: T,
    gradient_norm: T,
    iterations: usize,
    termination: Termination,
) -> LeastSquaresReport<T> {
    LeastSquaresReport {
        parameters,
        cost,
        gradient_norm,
        iterations,
        termination,
    }
}
