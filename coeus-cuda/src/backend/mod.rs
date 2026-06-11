use crate::driver::{get_cuda_context, CudaDriver};
use crate::storage::CudaStorage;
use coeus_core::{Backend, ComputeBackend, Scalar, Storage};

pub mod ops;

pub trait CudaScalar: Scalar + leto_ops::Scalar {
    const CUDA_TYPE: &'static str;
}

impl CudaScalar for f32 {
    const CUDA_TYPE: &'static str = "float";
}

impl CudaScalar for f64 {
    const CUDA_TYPE: &'static str = "double";
}

impl CudaScalar for half::f16 {
    const CUDA_TYPE: &'static str = "__half";
}

impl CudaScalar for half::bf16 {
    const CUDA_TYPE: &'static str = "__nv_bfloat16";
}

impl CudaScalar for i32 {
    const CUDA_TYPE: &'static str = "int";
}

/// NVIDIA CUDA acceleration backend.
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaBackend;

impl coeus_core::backend::private::Sealed for CudaBackend {}

impl CudaBackend {
    pub const fn new() -> Self {
        Self
    }
}

impl ComputeBackend for CudaBackend {
    type DeviceBuffer<T: Scalar> = CudaStorage<T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = std::future::Ready<T>;

    #[inline]
    fn name(&self) -> &'static str {
        "cuda"
    }

    #[inline]
    fn num_threads(&self) -> usize {
        1
    }

    #[inline]
    fn allocate<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        CudaStorage::allocate(len)
    }

    #[inline]
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) {
        let size = dst.len();
        let data = vec![val; size];
        self.copy_to_device(&data, dst);
    }

    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>) {
        if dst.cu_deviceptr() == 0 {
            panic!("copy_to_device: dst.cu_deviceptr() is 0!");
        }
        if let Some(stream) = crate::driver::get_borrowed_stream() {
            unsafe {
                let _ = stream.synchronize();
            }
        }
        if get_cuda_context().is_some() {
            let bytesize = std::mem::size_of_val(src);
            unsafe {
                let res = cuda_core::sys::cuMemcpyHtoD_v2(
                    dst.cu_deviceptr() as cuda_core::sys::CUdeviceptr,
                    src.as_ptr() as *const std::ffi::c_void,
                    bytesize,
                );
                if res != 0 {
                    panic!("cuMemcpyHtoD_v2 failed with error code: {}", res);
                }
            }
        }
    }

    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        if src.cu_deviceptr() == 0 {
            panic!("copy_to_host: src.cu_deviceptr() is 0!");
        }
        if let Some(stream) = crate::driver::get_borrowed_stream() {
            unsafe {
                let _ = stream.synchronize();
            }
        }
        if get_cuda_context().is_some() {
            let bytesize = std::mem::size_of_val(dst);
            unsafe {
                let res = cuda_core::sys::cuMemcpyDtoH_v2(
                    dst.as_mut_ptr() as *mut std::ffi::c_void,
                    src.cu_deviceptr() as cuda_core::sys::CUdeviceptr,
                    bytesize,
                );
                if res != 0 {
                    panic!("cuMemcpyDtoH_v2 failed with error code: {}", res);
                }
            }
        }
    }
}

impl Backend for CudaBackend {
    #[inline]
    fn parallel_for<F>(&self, start: usize, end: usize, f: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        coeus_core::SequentialBackend::new().parallel_for(start, end, f);
    }
}
