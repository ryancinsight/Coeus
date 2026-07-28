// ── Autograd reduction nodes ──
//
// Organized into sub-modules: max_axis/min_axis in `max_min`, norm variants in `norm`.

mod max_min;
mod norm;
mod prod;
pub mod sort;
pub mod topk;
mod variance;

pub use max_min::{max_axis, min_axis};
pub use norm::{norm, norm_p, norm_p_axis};
pub use prod::prod;
pub use sort::sort;
pub use topk::topk;
pub use variance::{
    std_dev, std_dev_axis, std_mean, std_mean_axis, var, var_axis, var_mean, var_mean_axis,
};

// ── log_sum_exp ────────────────────────────────────────────────────────────
//
// Numerically stable log-sum-exp along `axis`.
//
// Composed from existing tracked ops so that grad flows through the DAG
// automatically without a bespoke node:
//
//   x_max  = max_axis(x, axis)             → tracked, broadcasts back as size-1 dim
//   x_sh   = x - x_max                     → stable shifted input, exp in (-inf, 0]
//   sum_e  = sum_axis(exp(x_sh), axis)      → sum of stabilised exponentials
//   lse    = log(sum_e) + x_max             → add back the max offset
//
// The backward through log and exp reproduces the softmax probabilities, so
// d(lse)/dx_i = softmax(x)_i — correct by composition.

/// Tracked numerically stable log-sum-exp along `axis`.
///
/// Output shape equals input shape with `axis` dimension reduced to 1.
/// No new backward node — gradients flow through the composed tracked ops.
///
/// Precision: all computation in native `T` precision; max-subtraction
/// constrains exp input to (−∞, 0], eliminating overflow at any precision.
#[inline]
pub fn log_sum_exp<T: coeus_core::Float, B: coeus_ops::BackendOps<T> + Default>(
    x: &crate::Var<T, B>,
    axis: usize,
) -> Result<crate::Var<T, B>, B::Error> {
    let x_max = max_axis(x, axis)?;
    let x_shifted = crate::ops::arithmetic::sub(x, &x_max)?;
    let exp_sh = crate::ops::activation::exp(&x_shifted)?;
    let sum_exp = crate::ops::arithmetic::sum_axis(&exp_sh, axis)?;
    let log_sum = crate::ops::activation::log(&sum_exp)?;
    crate::ops::arithmetic::add(&log_sum, &x_max)
}
