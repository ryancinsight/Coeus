mod backward;
mod forward;
mod shader;

pub use backward::{dispatch_avg_pool1d_backward, dispatch_max_pool1d_backward};
pub use forward::{dispatch_avg_pool1d, dispatch_max_pool1d};
