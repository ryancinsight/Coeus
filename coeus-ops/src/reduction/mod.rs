// ── Reduction module ──

pub(crate) mod cumsum;
pub(crate) mod mean;
pub(crate) mod stats;
pub(crate) mod sum;
pub(crate) mod topk;

pub use cumsum::{cumsum, suffix_sum};
pub use mean::{mean, mean_axis};
pub use stats::{norm, norm_p, norm_p_axis, std_dev, std_dev_axis, var, var_axis};
pub use sum::{max_axis, min_axis, sum, sum_axis};
pub use topk::{argmax, argmin, topk};
