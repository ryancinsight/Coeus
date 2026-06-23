// ── Reduction module ──

pub(crate) mod cumsum;
pub(crate) mod mean;
pub(crate) mod sum;
pub(crate) mod topk;

pub use cumsum::{cumsum, suffix_sum};
pub use mean::{mean, mean_axis};
pub use sum::{max_axis, min_axis, sum, sum_axis};
pub use topk::{argmax, argmin, topk};
