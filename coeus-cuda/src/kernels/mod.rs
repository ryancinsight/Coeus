pub mod ptx;
pub mod launch_ops;
pub mod launch_matmul;
pub mod launch_conv;
pub mod fuse;
pub mod reduce;
pub mod pool;
pub mod optim;

pub use launch_ops::{
    launch_contiguous_binary, launch_contiguous_unary, launch_strided_binary, launch_strided_unary,
};
pub use launch_matmul::launch_matmul_tiled;
pub use launch_conv::{
    launch_conv1d, launch_conv1d_backward, launch_conv2d, launch_conv2d_backward,
};
pub use fuse::dispatch_fused;
pub use reduce::{dispatch_reduce, dispatch_fused_reduce};
pub use pool::{
    dispatch_max_pool2d, dispatch_max_pool2d_backward,
    dispatch_avg_pool2d, dispatch_avg_pool2d_backward,
};
pub use optim::{
    launch_sgd_step, launch_adam_step, launch_rmsprop_step, launch_adagrad_step,
};

use std::sync::OnceLock;
use coeus_core::{Layout, ComputeBackend};
use crate::storage::CudaStorage;
use crate::driver::{CudaDriver, CUmodule, CUfunction, get_cuda_context};
use crate::backend::CudaBackend;
use crate::kernels::ptx::PTX_SOURCE;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLayoutInfo {
    pub offset: u32,
    pub ndim: u32,
    pub shape: [u32; 8],
    pub strides: [u32; 8],
}

impl GpuLayoutInfo {
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

pub fn get_cuda_module() -> Option<CUmodule> {
    CUDA_MODULE.get_or_init(|| {
        let drv = CudaDriver::get()?;
        let _ctx = get_cuda_context()?;
        
        let ptx_src = format!("{}\0", PTX_SOURCE);
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            let res = (drv.cu_module_load_data)(&mut module, ptx_src.as_ptr() as *const std::ffi::c_void);
            if res == 0 {
                Some(CudaModuleWrapper { module })
            } else {
                None
            }
        }
    }).as_ref().map(|wrapper| wrapper.module)
}

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
