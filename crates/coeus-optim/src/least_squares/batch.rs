//! Leading-axis batching for independent least-squares problems.

use coeus_core::Scalar;

use super::{
    levenberg_marquardt, LeastSquaresProblem, LeastSquaresReport, LeastSquaresScalar,
    LevenbergMarquardtConfig, ProblemError, SolverError,
};

/// A collection of independent least-squares problems with one shared shape.
///
/// The leading problem axis is selected by `problem_index`; every problem has
/// the same residual and parameter counts. Implementors write into the slices
/// supplied by the solver, so batching does not require an owned tensor or a
/// conversion through another array provider.
pub trait BatchedLeastSquaresProblem<T: Scalar> {
    /// Number of independent problems on the leading axis.
    fn problem_count(&self) -> usize;

    /// Number of residual components in every problem.
    fn residual_count(&self) -> usize;

    /// Number of free parameters in every problem.
    fn parameter_count(&self) -> usize;

    /// Evaluate one problem's residual vector.
    ///
    /// # Errors
    ///
    /// Returns [`ProblemError::Domain`] for a trial point where the model is
    /// undefined and [`ProblemError::Evaluation`] for an unrecoverable
    /// evaluation failure. Domain errors are rejected steps, matching
    /// [`super::levenberg_marquardt`].
    fn residuals(
        &self,
        problem_index: usize,
        parameters: &[T],
        residuals: &mut [T],
    ) -> Result<(), ProblemError>;

    /// Evaluate one problem's row-major Jacobian.
    ///
    /// # Errors
    ///
    /// Returns the same typed failures as [`Self::residuals`].
    fn jacobian(
        &self,
        problem_index: usize,
        parameters: &[T],
        jacobian: &mut [T],
    ) -> Result<(), ProblemError>;
}

/// Failure modes that identify which batched problem rejected the solve.
#[derive(Debug, thiserror::Error)]
pub enum BatchedSolverError {
    /// The flattened leading-axis parameter buffer has the wrong length.
    #[error("expected {expected} parameters for {problems} problems with {parameters_per_problem} parameters each, got {actual}")]
    ParameterCount {
        /// Required flattened parameter length.
        expected: usize,
        /// Number of independent problems.
        problems: usize,
        /// Parameters per problem.
        parameters_per_problem: usize,
        /// Supplied flattened parameter length.
        actual: usize,
    },

    /// The parameter-count multiplication overflowed `usize`.
    #[error("parameter buffer size overflows for {problems} problems with {parameters_per_problem} parameters each")]
    ParameterCountOverflow {
        /// Number of independent problems.
        problems: usize,
        /// Parameters per problem.
        parameters_per_problem: usize,
    },

    /// One independent problem failed during its solve.
    #[error("batched problem {index} failed: {source}")]
    Problem {
        /// Leading-axis index of the failed problem.
        index: usize,
        /// Failure from the single-problem solver.
        #[source]
        source: SolverError,
    },
}

/// Solve independent problems over a flattened leading parameter axis.
///
/// `initial_parameters` is laid out as
/// `problem_count × parameter_count`, with each problem's parameters
/// contiguous. The returned reports preserve that leading-axis order. The
/// single-problem Levenberg–Marquardt implementation remains the numerical
/// source of truth; this function supplies only the indexed view and result
/// assembly, so batched and scalar fits share convergence and damping rules.
///
/// # Errors
///
/// Returns [`BatchedSolverError::ParameterCount`] for a malformed flattened
/// input, [`BatchedSolverError::ParameterCountOverflow`] when its required
/// length cannot be represented, or [`BatchedSolverError::Problem`] with the
/// failing leading-axis index when one fit cannot complete.
pub fn batched_levenberg_marquardt<T, P>(
    problem: &P,
    initial_parameters: &[T],
    config: &LevenbergMarquardtConfig<T>,
) -> Result<Vec<LeastSquaresReport<T>>, BatchedSolverError>
where
    T: LeastSquaresScalar,
    P: BatchedLeastSquaresProblem<T>,
{
    let problems = problem.problem_count();
    let parameters_per_problem = problem.parameter_count();
    let expected = problems.checked_mul(parameters_per_problem).ok_or(
        BatchedSolverError::ParameterCountOverflow {
            problems,
            parameters_per_problem,
        },
    )?;
    if initial_parameters.len() != expected {
        return Err(BatchedSolverError::ParameterCount {
            expected,
            problems,
            parameters_per_problem,
            actual: initial_parameters.len(),
        });
    }

    let residual_count = problem.residual_count();
    let mut reports = Vec::with_capacity(problems);
    for index in 0..problems {
        let start = index * parameters_per_problem;
        let end = start + parameters_per_problem;
        let view = IndexedProblem {
            problem,
            index,
            residual_count,
            parameters_per_problem,
        };
        let report = levenberg_marquardt(&view, &initial_parameters[start..end], config)
            .map_err(|source| BatchedSolverError::Problem { index, source })?;
        reports.push(report);
    }
    Ok(reports)
}

struct IndexedProblem<'a, P> {
    problem: &'a P,
    index: usize,
    residual_count: usize,
    parameters_per_problem: usize,
}

impl<T, P> LeastSquaresProblem<T> for IndexedProblem<'_, P>
where
    T: Scalar,
    P: BatchedLeastSquaresProblem<T>,
{
    fn residual_count(&self) -> usize {
        self.residual_count
    }

    fn parameter_count(&self) -> usize {
        self.parameters_per_problem
    }

    fn residuals(&self, parameters: &[T], residuals: &mut [T]) -> Result<(), ProblemError> {
        self.problem.residuals(self.index, parameters, residuals)
    }

    fn jacobian(&self, parameters: &[T], jacobian: &mut [T]) -> Result<(), ProblemError> {
        self.problem.jacobian(self.index, parameters, jacobian)
    }
}
