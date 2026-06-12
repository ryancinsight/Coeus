use std::sync::Arc;

pub type CUdevice = i32;
pub type CUcontext = *mut std::ffi::c_void;
pub type CUdeviceptr = u64;
pub type CUresult = i32;
pub type CUmodule = *mut std::ffi::c_void;
pub type CUfunction = *mut std::ffi::c_void;
pub type CUstream = *mut std::ffi::c_void;

/// CUDA driver facade for builds compiled without the `cuda` feature.
pub struct CudaDriver;

impl CudaDriver {
    /// CUDA is not compiled into this crate variant.
    #[inline]
    pub const fn get() -> Option<&'static Self> {
        None
    }
}

/// CUDA contexts are unavailable without the `cuda` feature.
#[inline]
pub const fn get_cuda_context() -> Option<CUcontext> {
    None
}

/// Borrowed cutile devices are unavailable without the `cuda` feature.
#[inline]
pub fn get_borrowed_device() -> Option<Arc<()>> {
    None
}

/// Borrowed cutile streams are unavailable without the `cuda` feature.
#[inline]
pub fn get_borrowed_stream() -> Option<Arc<()>> {
    None
}
