// ── Reduction module ──

pub(crate) mod cumprod;
pub(crate) mod cumsum;
pub(crate) mod linalg;
pub(crate) mod mean;
pub(crate) mod stats;
pub(crate) mod sum;
pub(crate) mod topk;

pub use cumprod::cumprod;
pub use cumsum::{cumsum, suffix_sum};
pub use linalg::{cross, dot};
pub use mean::{mean, mean_axis};
pub use stats::{
    frobenius_norm, frobenius_norm_batched, norm, norm_p, norm_p_axis, std_dev, std_dev_axis, var,
    var_axis,
};
pub use sum::{amax, amin, max_axis, min_axis, prod, sum, sum_axis};
pub use topk::{argmax, argmin, topk};
