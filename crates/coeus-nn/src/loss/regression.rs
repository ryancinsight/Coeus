//! Elementwise prediction-error losses.
//!
//! Pointwise losses between a prediction and a target of shared shape:
//! squared error, absolute error, the Huber/Smooth-L1 family, and the
//! distributional Poisson and Gaussian negative-log-likelihoods.

use coeus_autograd::Var;
use coeus_core::Float;

/// Mean Squared Error loss.
///
/// Computes mean squared error between pred and target.
/// Returns a scalar Var (shape `[1]`).
#[inline]
pub fn mse_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    let diff = coeus_autograd::sub(pred, target);
    let sq = coeus_autograd::mul(&diff, &diff);
    coeus_autograd::mean(&sq)
}

/// Huber (Smooth L1) Loss.
/// pred: `[N]`, target: `[N]`, delta: huber threshold.
///
/// # Errors
///
/// Returns the backend error type when the input shapes differ, the reduction
/// is empty, or `delta` is non-finite or non-positive.
#[inline]
pub fn huber_loss<
    T: coeus_core::FloatElement + Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
    delta: T,
) -> Result<Var<T, B>, B::Error> {
    coeus_autograd::huber_loss(pred, target, delta)
}

/// L1 (mean absolute error) loss.
/// pred: `[N]`, target: `[N]`. Computes `mean(|pred - target|)`.
#[inline]
pub fn l1_loss<
    T: coeus_core::FloatElement + Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::l1_loss(pred, target)
}

/// Poisson negative-log-likelihood loss (log-input form).
/// input holds `log(λ)`, target the observed counts; both share shape.
/// Computes `mean(exp(input) - target * input)` (PyTorch
/// `PoissonNLLLoss(log_input=True, full=False)`).
#[inline]
pub fn poisson_nll<
    T: coeus_core::FloatElement + Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    input: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::poisson_nll(input, target)
}

/// Smooth L1 (Huber-β) loss (PyTorch
/// `SmoothL1Loss(reduction="mean", beta=float)`). Computes
/// `mean_i loss_smooth(pred[i] - target[i], beta)` with
/// `loss_smooth(z, β) = 0.5 z²/β` if `|z| < β`, else `|z| - 0.5 β`.
/// `pred` and `target` must share shape.
#[inline]
pub fn smooth_l1_loss<
    T: coeus_core::FloatElement + Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
    beta: T,
) -> Var<T, B> {
    coeus_autograd::smooth_l1_loss(pred, target, beta)
}

/// Gaussian negative-log-likelihood loss (PyTorch `GaussianNLLLoss` with
/// `reduction="mean"` and `full=False`).
///
/// `input`, `target`, and `var` share shape. Computes:
/// `loss = 0.5 * mean((input - target)^2 / var + log(var))`
///
/// Composed from existing autograd ops. When `full=true`, adds `0.5 * log(2π)`.
#[inline]
pub fn gaussian_nll_loss<
    T: Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    input: &Var<T, B>,
    target: &Var<T, B>,
    var: &Var<T, B>,
    full: bool,
) -> Var<T, B> {
    let diff = coeus_autograd::sub(input, target);
    let diff_sq = coeus_autograd::mul(&diff, &diff);
    let var_term = coeus_autograd::div(&diff_sq, var);
    let log_var = coeus_autograd::log(var);
    let loss = coeus_autograd::scalar_mul(
        &coeus_autograd::add(&var_term, &log_var),
        <T as coeus_core::Scalar>::from_f64(0.5),
    );
    if full {
        let two_pi = <T as coeus_core::Scalar>::from_f64(2.0 * std::f64::consts::PI);
        coeus_autograd::scalar_add(
            &coeus_autograd::mean(&loss),
            <T as coeus_core::Scalar>::from_f64(0.5) * two_pi.log_op(),
        )
    } else {
        coeus_autograd::mean(&loss)
    }
}
