use std::sync::Arc;

/// CUDA device identifier.
pub type CUdevice = i32;
/// CUDA context handle.
pub type CUcontext = *mut std::ffi::c_void;
/// CUDA device memory pointer.
pub type CUdeviceptr = u64;
/// CUDA driver API result code.
pub type CUresult = i32;
/// CUDA module handle.
pub type CUmodule = *mut std::ffi::c_void;
/// CUDA function handle.
pub type CUfunction = *mut std::ffi::c_void;
/// CUDA stream handle.
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
