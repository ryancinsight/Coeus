use crate::driver::CUdeviceptr;
use coeus_core::{Scalar, Storage, StorageMut};
use hephaestus_cuda::{ComputeDevice, CudaBuffer, DeviceBuffer};
use std::sync::Arc;

/// Device storage allocated on an NVIDIA GPU using the hephaestus-cuda backend.
pub struct CudaStorage<T> {
    pub buffer: Arc<CudaBuffer<T>>,
}

impl<T> coeus_core::storage::private::Sealed for CudaStorage<T> {}

impl<T> Clone for CudaStorage<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
        }
    }
}

unsafe impl<T: Send> Send for CudaStorage<T> {}
unsafe impl<T: Sync> Sync for CudaStorage<T> {}

impl<T: Scalar> CudaStorage<T> {
    /// Allocate a new GPU device buffer.
    pub fn new(len: usize) -> Self {
        let device = crate::backend::get_cuda_device();
        let buffer = device
            .alloc_zeroed::<T>(len)
            .expect("CudaStorage::new failed to allocate GPU buffer");
        Self {
            buffer: Arc::new(buffer),
        }
    }

    /// Retrieve the raw CUDA device pointer.
    #[inline]
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.buffer.raw()
    }
}

impl<T: Scalar> Storage<T> for CudaStorage<T> {
    #[inline]
    fn len(&self) -> usize {
        self.buffer.len()
    }

    #[inline]
    fn allocate(len: usize) -> Self {
        Self::new(len)
    }

    #[inline]
    fn try_as_slice(&self) -> Option<&[T]> {
        None
    }
}

impl<T: Scalar> StorageMut<T> for CudaStorage<T> {
    #[inline]
    fn try_as_mut_slice(&mut self) -> Option<&mut [T]> {
        None
    }

    fn make_unique(&mut self) {
        if Arc::strong_count(&self.buffer) > 1 {
            let device = crate::backend::get_cuda_device();
            let len = self.buffer.len();
            let new_buffer = device
                .alloc_zeroed::<T>(len)
                .expect("Failed to allocate CoW buffer");

            if len > 0 {
                let bytes = len * std::mem::size_of::<T>();
                device.bind().expect("Failed to bind CUDA device");
                unsafe {
                    let res =
                        cuda_core::sys::cuMemcpyDtoD_v2(new_buffer.raw(), self.buffer.raw(), bytes);
                    if res != 0 {
                        panic!("cuMemcpyDtoD_v2 failed with code: {}", res);
                    }
                }
            }

            self.buffer = Arc::new(new_buffer);
        }
    }
}
