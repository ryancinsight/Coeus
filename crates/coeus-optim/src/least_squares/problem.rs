//! The nonlinear least-squares problem contract.

use coeus_core::Scalar;

/// Failure modes a problem definition can report to the solver.
#[derive(Debug, thiserror::Error)]
pub enum ProblemError {
    /// The model is undefined at the trial parameters — a negative diffusivity
    /// under a square root, a log of a non-positive signal.
    ///
    /// This is not a solver failure: Levenberg-Marquardt responds by rejecting
    /// the trial step and increasing damping, which pulls the next trial back
    /// toward the last accepted point.
    #[error("model undefined at the trial parameters: {reason}")]
    Domain {
        /// What made the evaluation invalid.
        reason: String,
    },

    /// Evaluation failed for a reason the solver cannot recover from.
    #[error("residual evaluation failed: {reason}")]
    Evaluation {
        /// What failed.
        reason: String,
    },
}

/// A nonlinear least-squares problem: minimize `0.5 * ||r(p)||²`.
///
/// Implementors supply the residual vector `r(p)` and its Jacobian
/// `J(p) = ∂r/∂p`. The solver owns the iteration, damping, and convergence
/// tests; it never differentiates or optimizes on its own.
///
/// # Why a residual contract rather than an [`Optimizer`](crate::Optimizer)
///
/// The gradient-descent optimizers in this crate step on gradients already
/// accumulated into parameters, which suits network training. A least-squares
/// solver instead re-evaluates the model at trial points to decide whether a
/// step is acceptable, and exploits the Gauss-Newton structure `JᵀJ ≈ H` that
/// a bare gradient does not expose. The two are different contracts, not two
/// spellings of one.
///
/// # Jacobian layout
///
/// Row-major, `residual_count()` rows by `parameter_count()` columns, so entry
/// `(i, j)` at index `i * parameter_count() + j` is `∂rᵢ/∂pⱼ`.
pub trait LeastSquaresProblem<T: Scalar> {
    /// Number of residual components.
    fn residual_count(&self) -> usize;

    /// Number of free parameters.
    fn parameter_count(&self) -> usize;

    /// Evaluate `r(parameters)` into `residuals`.
    ///
    /// # Errors
    ///
    /// [`ProblemError::Domain`] when the model is undefined at these
    /// parameters, which the solver treats as a rejected step rather than a
    /// failure. [`ProblemError::Evaluation`] for unrecoverable failures.
    fn residuals(&self, parameters: &[T], residuals: &mut [T]) -> Result<(), ProblemError>;

    /// Evaluate the Jacobian at `parameters` into `jacobian`, row-major.
    ///
    /// # Errors
    ///
    /// As [`Self::residuals`].
    fn jacobian(&self, parameters: &[T], jacobian: &mut [T]) -> Result<(), ProblemError>;
}
