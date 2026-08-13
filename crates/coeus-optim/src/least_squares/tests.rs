//! Least-squares solver verification against analytical oracles.
//!
//! Every case has a known closed-form answer or a published minimum, so the
//! assertions check the recovered parameters rather than that the solver merely
//! returned. Tolerances derive from the working scalar's epsilon and the
//! problem's conditioning, never from what the implementation happens to hit.

use super::*;
use coeus_core::Scalar;

/// `sqrt(ε)` for `T`, the accuracy floor of a first-order criterion.
fn sqrt_epsilon<T: Scalar>() -> T {
    let mut epsilon = T::one();
    let two = <T as Scalar>::from_f64(2.0);
    while T::one() + epsilon / two > T::one() {
        epsilon = epsilon / two;
    }
    epsilon.sqrt_val()
}

/// A linear model `r = A·p - b`, whose Jacobian is constant.
///
/// Gauss-Newton is exact on a linear residual, so this is the oracle that
/// separates "the step direction is right" from "the damping loop eventually
/// wanders to the answer".
struct LinearProblem<T> {
    matrix: Vec<T>,
    target: Vec<T>,
    parameters: usize,
}

impl<T: Scalar> LeastSquaresProblem<T> for LinearProblem<T> {
    fn residual_count(&self) -> usize {
        self.target.len()
    }

    fn parameter_count(&self) -> usize {
        self.parameters
    }

    fn residuals(&self, parameters: &[T], residuals: &mut [T]) -> Result<(), ProblemError> {
        for (row, slot) in residuals.iter_mut().enumerate() {
            let base = row * self.parameters;
            let mut sum = T::zero();
            for (column, parameter) in parameters.iter().enumerate() {
                sum += self.matrix[base + column] * *parameter;
            }
            *slot = sum - self.target[row];
        }
        Ok(())
    }

    fn jacobian(&self, _parameters: &[T], jacobian: &mut [T]) -> Result<(), ProblemError> {
        jacobian.copy_from_slice(&self.matrix);
        Ok(())
    }
}

/// Two independent 2×2 linear systems sharing one residual/parameter shape.
struct BatchedLinearProblem<T> {
    matrix: Vec<T>,
    target: Vec<T>,
    problems: usize,
    residuals: usize,
    parameters: usize,
}

impl<T: Scalar> BatchedLeastSquaresProblem<T> for BatchedLinearProblem<T> {
    fn problem_count(&self) -> usize {
        self.problems
    }

    fn residual_count(&self) -> usize {
        self.residuals
    }

    fn parameter_count(&self) -> usize {
        self.parameters
    }

    fn residuals(
        &self,
        problem_index: usize,
        parameters: &[T],
        residuals: &mut [T],
    ) -> Result<(), ProblemError> {
        let matrix_start = problem_index * self.residuals * self.parameters;
        let target_start = problem_index * self.residuals;
        for (row, slot) in residuals.iter_mut().enumerate() {
            let row_start = matrix_start + row * self.parameters;
            let mut sum = T::zero();
            for (column, parameter) in parameters.iter().enumerate() {
                sum += self.matrix[row_start + column] * *parameter;
            }
            *slot = sum - self.target[target_start + row];
        }
        Ok(())
    }

    fn jacobian(
        &self,
        problem_index: usize,
        _parameters: &[T],
        jacobian: &mut [T],
    ) -> Result<(), ProblemError> {
        let start = problem_index * self.residuals * self.parameters;
        let end = start + self.residuals * self.parameters;
        jacobian.copy_from_slice(&self.matrix[start..end]);
        Ok(())
    }
}

/// Monoexponential decay `S(b) = s0 · exp(-b · d)`, the shape every diffusion
/// signal model is built from.
///
/// Parameters are `[s0, d]`; residuals are `model - measured`.
struct DecayProblem<T> {
    b_values: Vec<T>,
    measured: Vec<T>,
}

impl<T: Scalar> DecayProblem<T> {
    fn model(&self, b: T, s0: T, d: T) -> T {
        <T as Scalar>::from_f64((-(Scalar::to_f64(b)) * Scalar::to_f64(d)).exp()) * s0
    }
}

impl<T: Scalar> LeastSquaresProblem<T> for DecayProblem<T> {
    fn residual_count(&self) -> usize {
        self.b_values.len()
    }

    fn parameter_count(&self) -> usize {
        2
    }

    fn residuals(&self, parameters: &[T], residuals: &mut [T]) -> Result<(), ProblemError> {
        let (s0, d) = (parameters[0], parameters[1]);
        for (index, slot) in residuals.iter_mut().enumerate() {
            *slot = self.model(self.b_values[index], s0, d) - self.measured[index];
        }
        Ok(())
    }

    fn jacobian(&self, parameters: &[T], jacobian: &mut [T]) -> Result<(), ProblemError> {
        // d/ds0 = exp(-b·d); d/dd = -b · s0 · exp(-b·d)
        let (s0, d) = (parameters[0], parameters[1]);
        for (index, b) in self.b_values.iter().enumerate() {
            let decay = <T as Scalar>::from_f64((-(Scalar::to_f64(*b)) * Scalar::to_f64(d)).exp());
            jacobian[index * 2] = decay;
            jacobian[index * 2 + 1] = T::zero() - *b * s0 * decay;
        }
        Ok(())
    }
}

/// Rosenbrock in least-squares form: `r = [10(y - x²), 1 - x]`.
///
/// The published minimum is `(1, 1)` with zero residual. Its curved valley is
/// the standard check that damping adapts rather than taking a fixed step.
struct RosenbrockProblem;

impl<T: Scalar> LeastSquaresProblem<T> for RosenbrockProblem {
    fn residual_count(&self) -> usize {
        2
    }

    fn parameter_count(&self) -> usize {
        2
    }

    fn residuals(&self, parameters: &[T], residuals: &mut [T]) -> Result<(), ProblemError> {
        let (x, y) = (parameters[0], parameters[1]);
        residuals[0] = <T as Scalar>::from_f64(10.0) * (y - x * x);
        residuals[1] = T::one() - x;
        Ok(())
    }

    fn jacobian(&self, parameters: &[T], jacobian: &mut [T]) -> Result<(), ProblemError> {
        let x = parameters[0];
        jacobian[0] = <T as Scalar>::from_f64(-20.0) * x;
        jacobian[1] = <T as Scalar>::from_f64(10.0);
        jacobian[2] = <T as Scalar>::from_f64(-1.0);
        jacobian[3] = T::zero();
        Ok(())
    }
}

/// A model undefined for a negative second parameter, to exercise the
/// domain-rejection path.
struct DomainLimitedProblem;

impl<T: Scalar> LeastSquaresProblem<T> for DomainLimitedProblem {
    fn residual_count(&self) -> usize {
        2
    }

    fn parameter_count(&self) -> usize {
        1
    }

    fn residuals(&self, parameters: &[T], residuals: &mut [T]) -> Result<(), ProblemError> {
        if parameters[0] < T::zero() {
            return Err(ProblemError::Domain {
                reason: "parameter must be non-negative".to_owned(),
            });
        }
        residuals[0] = parameters[0].sqrt_val() - <T as Scalar>::from_f64(2.0);
        residuals[1] = T::zero();
        Ok(())
    }

    fn jacobian(&self, parameters: &[T], jacobian: &mut [T]) -> Result<(), ProblemError> {
        let value = parameters[0];
        let guarded = if value > T::zero() {
            value
        } else {
            <T as Scalar>::from_f64(1e-12)
        };
        jacobian[0] = T::one() / (<T as Scalar>::from_f64(2.0) * guarded.sqrt_val());
        jacobian[1] = T::zero();
        Ok(())
    }
}

// ── Generic bodies, instantiated per shipped scalar type below ───────────────

fn linear_problem_is_solved_exactly<T: LeastSquaresScalar>() {
    // 3x2 system with the exact solution [2, -1]: rows (1,0), (0,1), (1,1)
    // against targets 2, -1, 1 are consistent, so the minimum has zero residual.
    let problem = LinearProblem::<T> {
        matrix: [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
            .map(<T as Scalar>::from_f64)
            .to_vec(),
        target: [2.0, -1.0, 1.0].map(<T as Scalar>::from_f64).to_vec(),
        parameters: 2,
    };

    let report = levenberg_marquardt(
        &problem,
        &[T::zero(), T::zero()],
        &LevenbergMarquardtConfig::default(),
    )
    .expect("a consistent linear system is solvable");

    let tolerance = sqrt_epsilon::<T>() * <T as Scalar>::from_f64(100.0);
    assert!(
        (report.parameters[0] - <T as Scalar>::from_f64(2.0)).abs_val() < tolerance,
        "first parameter must recover 2, got {:?}",
        Scalar::to_f64(report.parameters[0])
    );
    assert!(
        (report.parameters[1] - <T as Scalar>::from_f64(-1.0)).abs_val() < tolerance,
        "second parameter must recover -1, got {:?}",
        Scalar::to_f64(report.parameters[1])
    );
    assert!(
        report.termination.is_converged(),
        "a linear problem must converge by a derived criterion, got {:?}",
        report.termination
    );
}

fn decay_model_recovers_known_parameters<T: LeastSquaresScalar>() {
    // Noise-free measurements from s0 = 1000, d = 0.0007 over a realistic
    // diffusion b-value ladder. An exact fit exists, so the recovered
    // parameters must reproduce the generators.
    let s0 = <T as Scalar>::from_f64(1000.0);
    let d = <T as Scalar>::from_f64(7e-4);
    let b_values: Vec<T> = [0.0, 200.0, 400.0, 700.0, 1000.0, 1500.0, 2000.0, 3000.0]
        .map(<T as Scalar>::from_f64)
        .to_vec();
    let measured: Vec<T> = b_values
        .iter()
        .map(|b| <T as Scalar>::from_f64((-(Scalar::to_f64(*b)) * Scalar::to_f64(d)).exp()) * s0)
        .collect();

    let problem = DecayProblem { b_values, measured };

    let report = levenberg_marquardt(
        &problem,
        &[
            <T as Scalar>::from_f64(500.0),
            <T as Scalar>::from_f64(2e-3),
        ],
        &LevenbergMarquardtConfig::default(),
    )
    .expect("a noise-free decay fit is solvable");

    // Relative tolerance: the parameters differ by six orders of magnitude, so
    // an absolute bound would be meaningless for one of them.
    let relative = sqrt_epsilon::<T>() * <T as Scalar>::from_f64(1000.0);
    assert!(
        (report.parameters[0] - s0).abs_val() / s0 < relative,
        "s0 must recover 1000, got {}",
        Scalar::to_f64(report.parameters[0])
    );
    assert!(
        (report.parameters[1] - d).abs_val() / d < relative,
        "d must recover 7e-4, got {}",
        Scalar::to_f64(report.parameters[1])
    );
}

fn rosenbrock_reaches_its_published_minimum<T: LeastSquaresScalar>() {
    let report = levenberg_marquardt(
        &RosenbrockProblem,
        &[<T as Scalar>::from_f64(-1.2), T::one()],
        &LevenbergMarquardtConfig::default(),
    )
    .expect("Rosenbrock is solvable from the standard start");

    let tolerance = sqrt_epsilon::<T>() * <T as Scalar>::from_f64(100.0);
    assert!(
        (report.parameters[0] - T::one()).abs_val() < tolerance
            && (report.parameters[1] - T::one()).abs_val() < tolerance,
        "must reach the published minimum (1, 1), got ({}, {})",
        Scalar::to_f64(report.parameters[0]),
        Scalar::to_f64(report.parameters[1])
    );
}

fn domain_rejection_does_not_abort_the_solve<T: LeastSquaresScalar>() {
    // Starting near zero with a steep Jacobian drives the first trial step
    // negative, where the model is undefined. The solver must damp and recover
    // rather than propagate the domain error.
    let report = levenberg_marquardt(
        &DomainLimitedProblem,
        &[<T as Scalar>::from_f64(0.01)],
        &LevenbergMarquardtConfig::default(),
    )
    .expect("a domain rejection is a rejected step, not a solver failure");

    let tolerance = sqrt_epsilon::<T>() * <T as Scalar>::from_f64(1000.0);
    assert!(
        (report.parameters[0] - <T as Scalar>::from_f64(4.0)).abs_val() < tolerance,
        "sqrt(p) = 2 has the solution p = 4, got {}",
        Scalar::to_f64(report.parameters[0])
    );
}

fn gradient_criterion_reports_stationarity<T: LeastSquaresScalar>() {
    // Started at the solution, the very first gradient test must fire, before
    // any step is taken.
    let problem = LinearProblem::<T> {
        matrix: [1.0, 0.0, 0.0, 1.0].map(<T as Scalar>::from_f64).to_vec(),
        target: [3.0, 5.0].map(<T as Scalar>::from_f64).to_vec(),
        parameters: 2,
    };

    let report = levenberg_marquardt(
        &problem,
        &[<T as Scalar>::from_f64(3.0), <T as Scalar>::from_f64(5.0)],
        &LevenbergMarquardtConfig::default(),
    )
    .expect("solvable");

    assert_eq!(
        report.termination,
        Termination::GradientTolerance,
        "starting at the minimum must terminate on the gradient criterion"
    );
    assert_eq!(
        report.iterations, 0,
        "no step is needed when the start is already stationary"
    );
}

fn iteration_limit_is_reported_as_unconverged<T: LeastSquaresScalar>() {
    let config = LevenbergMarquardtConfig::<T> {
        max_iterations: 1,
        ..LevenbergMarquardtConfig::default()
    };
    let report = levenberg_marquardt(
        &RosenbrockProblem,
        &[<T as Scalar>::from_f64(-1.2), T::one()],
        &config,
    )
    .expect("solvable");

    assert_eq!(report.termination, Termination::IterationLimit);
    assert!(
        !report.termination.is_converged(),
        "an exhausted budget must not be reported as convergence"
    );
}

fn underdetermined_problem_is_rejected<T: LeastSquaresScalar>() {
    let problem = LinearProblem::<T> {
        matrix: [1.0, 1.0].map(<T as Scalar>::from_f64).to_vec(),
        target: [1.0].map(<T as Scalar>::from_f64).to_vec(),
        parameters: 2,
    };

    let error = levenberg_marquardt(
        &problem,
        &[T::zero(), T::zero()],
        &LevenbergMarquardtConfig::default(),
    )
    .expect_err("one residual cannot determine two parameters");

    assert!(
        matches!(error, SolverError::Underdetermined { .. }),
        "must name the underdetermined contract, got {error}"
    );
}

fn parameter_count_mismatch_is_rejected<T: LeastSquaresScalar>() {
    let problem = LinearProblem::<T> {
        matrix: [1.0, 0.0, 0.0, 1.0].map(<T as Scalar>::from_f64).to_vec(),
        target: [1.0, 1.0].map(<T as Scalar>::from_f64).to_vec(),
        parameters: 2,
    };

    let error = levenberg_marquardt(&problem, &[T::zero()], &LevenbergMarquardtConfig::default())
        .expect_err("a one-element start cannot initialize two parameters");

    assert!(matches!(
        error,
        SolverError::ParameterCount {
            expected: 2,
            actual: 1
        }
    ));
}

fn batched_linear_problems_recover_independent_minima<T: LeastSquaresScalar>() {
    let problem = BatchedLinearProblem::<T> {
        // A₀ = I, target₀ = [2, -1]; A₁ = [[2, 1], [1, 3]], target₁ = [4, 7].
        matrix: [1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 3.0]
            .map(<T as Scalar>::from_f64)
            .to_vec(),
        target: [2.0, -1.0, 4.0, 7.0].map(<T as Scalar>::from_f64).to_vec(),
        problems: 2,
        residuals: 2,
        parameters: 2,
    };

    let reports = batched_levenberg_marquardt(
        &problem,
        &[T::zero(), T::zero(), T::zero(), T::zero()],
        &LevenbergMarquardtConfig::default(),
    )
    .expect("both independent linear systems are solvable");

    assert_eq!(reports.len(), 2, "the leading problem axis is preserved");
    let tolerance = sqrt_epsilon::<T>() * <T as Scalar>::from_f64(100.0);
    let expected = [[2.0, -1.0], [1.0, 2.0]];
    for (report, expected_parameters) in reports.iter().zip(expected) {
        assert!(report.termination.is_converged());
        for (actual, expected) in report.parameters.iter().zip(expected_parameters) {
            assert!(((*actual - <T as Scalar>::from_f64(expected)).abs_val()) < tolerance);
        }
    }
}

fn batched_parameter_count_mismatch_is_rejected<T: LeastSquaresScalar>() {
    let problem = BatchedLinearProblem::<T> {
        matrix: [1.0, 0.0, 0.0, 1.0].map(<T as Scalar>::from_f64).to_vec(),
        target: [1.0, 1.0].map(<T as Scalar>::from_f64).to_vec(),
        problems: 1,
        residuals: 2,
        parameters: 2,
    };

    let error =
        batched_levenberg_marquardt(&problem, &[T::zero()], &LevenbergMarquardtConfig::default())
            .expect_err("the flattened leading-axis buffer has the wrong length");

    assert!(matches!(
        error,
        BatchedSolverError::ParameterCount {
            expected: 2,
            problems: 1,
            parameters_per_problem: 2,
            actual: 1
        }
    ));
}

/// Instantiate every generic body across the shipped scalar types.
///
/// A solver verified at one concrete type is unverified for the rest; `f32` in
/// particular exercises the epsilon-derived tolerances at a precision where a
/// hardcoded bound would silently pass or fail.
macro_rules! scalar_suite {
    ($module:ident, $scalar:ty) => {
        mod $module {
            #[test]
            fn linear_problem_is_solved_exactly() {
                super::linear_problem_is_solved_exactly::<$scalar>();
            }

            #[test]
            fn decay_model_recovers_known_parameters() {
                super::decay_model_recovers_known_parameters::<$scalar>();
            }

            #[test]
            fn rosenbrock_reaches_its_published_minimum() {
                super::rosenbrock_reaches_its_published_minimum::<$scalar>();
            }

            #[test]
            fn domain_rejection_does_not_abort_the_solve() {
                super::domain_rejection_does_not_abort_the_solve::<$scalar>();
            }

            #[test]
            fn gradient_criterion_reports_stationarity() {
                super::gradient_criterion_reports_stationarity::<$scalar>();
            }

            #[test]
            fn iteration_limit_is_reported_as_unconverged() {
                super::iteration_limit_is_reported_as_unconverged::<$scalar>();
            }

            #[test]
            fn underdetermined_problem_is_rejected() {
                super::underdetermined_problem_is_rejected::<$scalar>();
            }

            #[test]
            fn parameter_count_mismatch_is_rejected() {
                super::parameter_count_mismatch_is_rejected::<$scalar>();
            }

            #[test]
            fn batched_linear_problems_recover_independent_minima() {
                super::batched_linear_problems_recover_independent_minima::<$scalar>();
            }

            #[test]
            fn batched_parameter_count_mismatch_is_rejected() {
                super::batched_parameter_count_mismatch_is_rejected::<$scalar>();
            }
        }
    };
}

scalar_suite!(single_precision, f32);
scalar_suite!(double_precision, f64);
