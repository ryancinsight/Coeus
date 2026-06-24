use crate::driver::CUdeviceptr;
use coeus_core::{Scalar, Storage, StorageMut};
use hephaestus_cuda::{ComputeDevice, CudaBuffer, DeviceBuffer};
use std::sync::Arc;
use themis::{MemoryTier, PlacementHint};

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
    #[inline]
    fn alloc_device_zeroed(len: usize) -> CudaBuffer<T> {
        let device = crate::backend::get_cuda_device();
        device
            .alloc_zeroed_with_hint(len, PlacementHint::Tier(MemoryTier::Device))
            .expect("CudaStorage::new failed to allocate GPU buffer in device tier")
    }

    /// Allocate a new GPU device buffer.
    pub fn new(len: usize) -> Self {
        let buffer = Self::alloc_device_zeroed(len);
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
            let new_buffer = Self::alloc_device_zeroed(len);

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_allocates_device_tier() {
        let storage = CudaStorage::<f32>::new(8);
        assert_eq!(storage.buffer.tier(), MemoryTier::Device);
    }

    #[test]
    fn host_pinned_staging_roundtrip_preserves_values() {
        let device = crate::backend::get_cuda_device();
        let input = vec![1.0f32, -2.5, 3.25, 8.0];
        let staging = device
            .upload_with_hint(&input, PlacementHint::Tier(MemoryTier::HostPinned))
            .expect("failed to upload into host-pinned tier");
        assert_eq!(staging.tier(), MemoryTier::HostPinned);
        let mut roundtrip = vec![0.0f32; input.len()];
        device
            .download(&staging, &mut roundtrip)
            .expect("failed to download from host-pinned tier");
        assert_eq!(roundtrip, input);
    }
}
