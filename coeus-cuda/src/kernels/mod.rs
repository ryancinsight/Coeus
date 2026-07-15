/// Kernel module for scaled-dot-product attention operations.
pub mod attention;
/// Kernel module for transposed convolution operations.
pub mod conv_transpose;
/// Kernel module for fused element-wise expression compilation and dispatch.
pub mod fuse;
/// Kernel module for convolution kernel launch helpers.
pub mod launch_conv;
/// Kernel module for tiled matrix multiplication kernel launch.
pub mod launch_matmul;
/// Kernel module for element-wise operator kernel launches.
pub mod launch_ops;
/// Kernel module for optimizer step kernels.
pub mod optim;
/// Kernel module for pooling operations.
pub mod pool;
/// Kernel module for embedded PTX kernel source.
pub mod ptx;
/// Kernel module for reduction operations.
pub mod reduce;
/// Kernel module for sliding-window unfold and adjoint fold operations.
pub mod unfold_fold;

pub use attention::{launch_sdp_attention, launch_sdp_attention_backward};
pub use conv_transpose::{launch_conv_transpose1d, launch_conv_transpose2d};
pub use fuse::dispatch_fused;
pub use launch_conv::{
    launch_conv1d, launch_conv1d_backward, launch_conv2d, launch_conv2d_backward, launch_conv3d,
    launch_conv3d_backward,
};
pub use launch_matmul::launch_matmul_tiled;
pub use launch_ops::{
    launch_contiguous_binary, launch_contiguous_unary, launch_strided_binary, launch_strided_unary,
};
pub use optim::{
    launch_adagrad_step, launch_adam_step, launch_adamw_step, launch_rmsprop_step, launch_sgd_step,
};
pub use pool::{
    dispatch_avg_pool1d, dispatch_avg_pool1d_backward, dispatch_avg_pool2d,
    dispatch_avg_pool2d_backward, dispatch_avg_pool3d, dispatch_avg_pool3d_backward,
    dispatch_max_pool1d, dispatch_max_pool1d_backward, dispatch_max_pool2d,
    dispatch_max_pool2d_backward, dispatch_max_pool3d, dispatch_max_pool3d_backward,
};
pub use reduce::{dispatch_fused_reduce, dispatch_reduce};
pub use unfold_fold::{dispatch_fold1d, dispatch_fold2d, dispatch_unfold1d, dispatch_unfold2d};

use crate::backend::CudaBackend;
use crate::driver::{get_cuda_context, CUfunction, CUmodule, CudaDriver};
use crate::kernels::ptx::PTX_SOURCE;
use crate::storage::CudaStorage;
use coeus_core::{ComputeBackend, Layout};
use std::sync::OnceLock;

/// GPU-side layout descriptor passed to CUDA kernels as a POD struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLayoutInfo {
    /// Element offset into the underlying buffer.
    pub offset: u32,
    /// Number of dimensions in the layout.
    pub ndim: u32,
    /// Per-dimension shape, padded to 8 entries.
    pub shape: [u32; 8],
    /// Per-dimension strides, padded to 8 entries.
    pub strides: [u32; 8],
}

impl GpuLayoutInfo {
    /// Convert a CPU [`Layout`] into a GPU-compatible layout descriptor.
    pub fn from_layout(layout: &Layout) -> Self {
        let mut shape = [0u32; 8];
        let mut strides = [0u32; 8];
        let ndim = layout.ndim();
        assert!(ndim <= 8, "CUDA backend supports up to 8 dimensions");
        for i in 0..ndim {
            shape[i] = layout.shape()[i] as u32;
            strides[i] = layout.strides()[i] as u32;
        }
        Self {
            offset: layout.offset() as u32,
            ndim: ndim as u32,
            shape,
            strides,
        }
    }
}

/// Create a device buffer containing the serialized GPU layout descriptor.
pub fn create_layout_buffer(layout: &Layout) -> CudaStorage<u32> {
    let gpu_layout = GpuLayoutInfo::from_layout(layout);
    let size_u32 = std::mem::size_of::<GpuLayoutInfo>() / 4;
    let mut storage = CudaStorage::<u32>::new(size_u32);
    let slice = unsafe {
        std::slice::from_raw_parts(&gpu_layout as *const GpuLayoutInfo as *const u32, size_u32)
    };
    CudaBackend::new().copy_to_device(slice, &mut storage);
    storage
}

struct CudaModuleWrapper {
    module: CUmodule,
}

impl Drop for CudaModuleWrapper {
    fn drop(&mut self) {
        if !self.module.is_null() {
            if let Some(drv) = CudaDriver::get() {
                unsafe {
                    (drv.cu_module_unload)(self.module);
                }
            }
        }
    }
}

unsafe impl Send for CudaModuleWrapper {}
unsafe impl Sync for CudaModuleWrapper {}

static CUDA_MODULE: OnceLock<Option<CudaModuleWrapper>> = OnceLock::new();

/// Retrieve the lazily-loaded CUDA module singleton containing the embedded PTX kernels.
pub fn get_cuda_module() -> Option<CUmodule> {
    CUDA_MODULE
        .get_or_init(|| {
            let drv = CudaDriver::get()?;
            let _ctx = get_cuda_context()?;

            let ptx_src = format!("{}\0", PTX_SOURCE);
            let mut module: CUmodule = std::ptr::null_mut();
            unsafe {
                let res = (drv.cu_module_load_data)(
                    &mut module,
                    ptx_src.as_ptr() as *const std::ffi::c_void,
                );
                if res == 0 {
                    Some(CudaModuleWrapper { module })
                } else {
                    panic!("cu_module_load_data failed with error code: {}", res);
                }
            }
        })
        .as_ref()
        .map(|wrapper| wrapper.module)
}

/// Look up a kernel function by name from the loaded CUDA module.
pub fn get_cuda_function(name: &str) -> Option<CUfunction> {
    let drv = CudaDriver::get()?;
    let module = get_cuda_module()?;
    let c_name = std::ffi::CString::new(name).ok()?;
    let mut func: CUfunction = std::ptr::null_mut();
    unsafe {
        let res = (drv.cu_module_get_function)(&mut func, module, c_name.as_ptr());
        if res == 0 {
            Some(func)
        } else {
            None
        }
    }
}

/// Launch a kernel over a flat 1-D grid of `total` threads (256/block).
///
/// Shared by the NVRTC-compiled kernels (attention, conv_transpose) that map
/// one thread to one output element. Returns `false` if the driver is absent or
/// the launch fails so the operation boundary can report the dispatch failure.
pub(crate) fn launch_1d(
    func: CUfunction,
    total: usize,
    args: &mut [*mut std::ffi::c_void],
) -> bool {
    let Some(drv) = CudaDriver::get() else {
        return false;
    };
    let block_size = 256usize;
    let grid_size = total.div_ceil(block_size);
    unsafe {
        let res = (drv.cu_launch_kernel)(
            func,
            grid_size as u32,
            1,
            1,
            block_size as u32,
            1,
            1,
            0,
            std::ptr::null_mut(),
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        );
        res == 0
    }
}
