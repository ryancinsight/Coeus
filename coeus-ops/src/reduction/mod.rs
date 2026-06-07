// ── Reduction module ──

mod sum;
mod mean;
mod cumsum;
mod topk;

pub use sum::{sum, sum_axis, max_axis, min_axis};
pub use mean::{mean, mean_axis};
pub use cumsum::{cumsum, suffix_sum};
pub use topk::{topk, argmax, argmin};
