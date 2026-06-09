// ── Reduction module ──

mod cumsum;
mod mean;
mod sum;
mod topk;

pub use cumsum::{cumsum, suffix_sum};
pub use mean::{mean, mean_axis};
pub use sum::{max_axis, min_axis, sum, sum_axis};
pub use topk::{argmax, argmin, topk};
