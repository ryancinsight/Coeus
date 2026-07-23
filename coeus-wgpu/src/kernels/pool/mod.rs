pub mod avg;
pub mod avg3d;
pub mod max;
pub mod max3d;
pub mod pool1d;

pub use avg::{dispatch_avg_pool2d, dispatch_avg_pool2d_backward};
pub use avg3d::{dispatch_avg_pool3d, dispatch_avg_pool3d_backward};
pub use max::{dispatch_max_pool2d, dispatch_max_pool2d_backward};
pub use max3d::{dispatch_max_pool3d, dispatch_max_pool3d_backward};
pub use pool1d::{
    dispatch_avg_pool1d, dispatch_avg_pool1d_backward, dispatch_max_pool1d,
    dispatch_max_pool1d_backward,
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PoolParams {
    pub(crate) kernel_size: u32,
    pub(crate) stride: u32,
    pub(crate) padding: u32,
    pub(crate) dilation: u32,
}
