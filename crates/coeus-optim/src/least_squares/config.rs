//! Solver configuration and termination vocabulary.

use coeus_core::Scalar;

/// Why the solver stopped.
///
/// Every variant except [`Self::IterationLimit`] is a derived criterion — a
/// statement about the problem's own quantities, not about effort spent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Termination {
    /// `‖Jᵀr‖_∞ ≤ gradient_tolerance`: the point is stationary to tolerance.
    ///
    /// This is the criterion that actually certifies a minimum. The others say
    /// only that progress stopped.
    GradientTolerance,

    /// `‖δ‖ ≤ step_tolerance · (‖p‖ + step_tolerance)`: the step is negligible
    /// relative to the parameter scale.
    StepTolerance,

    /// The relative reduction in `0.5‖r‖²` fell below `cost_tolerance`.
    CostTolerance,

    /// The iteration cap was reached with no criterion met.
    ///
    /// The returned parameters are the best accepted point, but they are not
    /// certified: treat this as a diagnostic, not a converged fit.
    IterationLimit,
}

impl Termination {
    /// Whether a derived convergence criterion was met.
    ///
    /// `false` only for [`Self::IterationLimit`], where the solver ran out of
    /// budget rather than converging.
    #[must_use]
    pub const fn is_converged(self) -> bool {
        !matches!(self, Self::IterationLimit)
    }
}

/// Levenberg-Marquardt tuning.
///
/// The tolerance defaults are `sqrt(ε)` for the working scalar type, the
/// standard choice for a first-order criterion evaluated in floating point: a
/// residual computed to relative accuracy `ε` cannot certify stationarity below
/// roughly `sqrt(ε)`, so a tighter request only buys iterations spent on noise.
#[derive(Clone, Copy, Debug)]
pub struct LevenbergMarquardtConfig<T> {
    /// Stop when `‖Jᵀr‖_∞` falls to or below this.
    pub gradient_tolerance: T,
    /// Stop when the relative parameter step falls to or below this.
    pub step_tolerance: T,
    /// Stop when the relative cost reduction falls to or below this.
    pub cost_tolerance: T,
    /// Maximum iterations before reporting [`Termination::IterationLimit`].
    pub max_iterations: usize,
    /// Initial damping factor `λ`.
    pub initial_damping: T,
    /// Factor damping is multiplied by on a rejected step.
    pub damping_increase: T,
    /// Factor damping is divided by on an accepted step.
    pub damping_decrease: T,
}

impl<T: Scalar> LevenbergMarquardtConfig<T> {
    /// Machine epsilon of `T`, obtained through the scalar's own round-trip.
    ///
    /// `Scalar` exposes no epsilon associated constant, so this derives it by
    /// bisection on the round-tripped value — exact for IEEE binary formats and
    /// evaluated once per construction, not per iteration.
    fn epsilon() -> T {
        let mut epsilon = T::one();
        let two = T::from_f64(2.0);
        while T::one() + epsilon / two > T::one() {
            epsilon = epsilon / two;
        }
        epsilon
    }
}

impl<T: Scalar> Default for LevenbergMarquardtConfig<T> {
    fn default() -> Self {
        let sqrt_epsilon = Self::epsilon().sqrt_val();
        Self {
            gradient_tolerance: sqrt_epsilon,
            step_tolerance: sqrt_epsilon,
            cost_tolerance: sqrt_epsilon,
            // Small dense problems of the kind this solver targets converge in
            // single-digit iterations; the cap is a runaway guard, not a budget
            // callers are expected to tune.
            max_iterations: 100,
            initial_damping: T::from_f64(1e-3),
            damping_increase: T::from_f64(10.0),
            damping_decrease: T::from_f64(10.0),
        }
    }
}

/// Outcome of a Levenberg-Marquardt solve.
#[derive(Clone, Debug)]
pub struct LeastSquaresReport<T> {
    /// Best accepted parameters.
    pub parameters: Vec<T>,
    /// `0.5‖r‖²` at [`Self::parameters`].
    pub cost: T,
    /// `‖Jᵀr‖_∞` at [`Self::parameters`].
    pub gradient_norm: T,
    /// Iterations executed.
    pub iterations: usize,
    /// Why the solver stopped.
    pub termination: Termination,
}
