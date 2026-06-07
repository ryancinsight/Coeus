pub mod max;
pub mod avg;
pub mod max3d;
pub mod avg3d;

pub use max::{dispatch_max_pool2d, dispatch_max_pool2d_backward};
pub use avg::{dispatch_avg_pool2d, dispatch_avg_pool2d_backward};
pub use max3d::{dispatch_max_pool3d, dispatch_max_pool3d_backward};
pub use avg3d::{dispatch_avg_pool3d, dispatch_avg_pool3d_backward};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PoolParams {
    pub(crate) kernel_size: u32,
    pub(crate) stride: u32,
    pub(crate) padding: u32,
    pub(crate) dilation: u32,
}
