//! Pooling nodes and entry points, grouped by the pooling rule.

mod adaptive;
mod avg;
mod max;

pub use adaptive::{adaptive_max_pool1d, adaptive_region, AdaptiveMaxPoolNode};
pub use avg::{avg_pool1d, avg_pool2d, avg_pool3d, AvgPoolNode};
pub use max::{max_pool1d, max_pool2d, max_pool3d, MaxPoolNode};

use coeus_core::Scalar;

struct PoolBackwardInputs<'a, T: Scalar, B: coeus_ops::BackendOps<T> + Default> {
    backend: &'a B,
    grad_out_storage: &'a B::DeviceBuffer<T>,
    grad_out_layout: &'a coeus_core::Layout,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    grad_input: &'a mut B::DeviceBuffer<T>,
    grad_input_layout: &'a coeus_core::Layout,
}
