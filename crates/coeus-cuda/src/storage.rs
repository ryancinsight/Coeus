use crate::driver::CUdeviceptr;
use coeus_core::{Scalar, Storage, StorageMut};
use hephaestus_cuda::{ComputeDevice, CudaBuffer, DeviceBuffer};
use std::sync::Arc;
use themis::{MemoryTier, PlacementHint};

/// Device storage allocated on an NVIDIA GPU using the hephaestus-cuda backend.
pub struct CudaStorage<T> {
    /// Reference-counted GPU device buffer backing this storage.
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
            let new_buffer = Self::alloc_device_zeroed(self.buffer.len());
            device
                .copy_buffer(self.buffer.as_ref(), &new_buffer)
                .expect("CudaStorage::make_unique failed to copy the device buffer");

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
    fn host_pinned_hint_uses_truthful_device_tier() {
        let device = crate::backend::get_cuda_device();
        let input = vec![1.0f32, -2.5, 3.25, 8.0];
        let staging = device
            .upload_with_hint(&input, PlacementHint::Tier(MemoryTier::HostPinned))
            .expect("failed to upload into host-pinned tier");
        // CUDA's ComputeDevice buffer contract represents device allocations;
        // host-pinned transfer memory is transient and is not exposed as this
        // persistent buffer type. The provider therefore reports Device.
        assert_eq!(staging.tier(), MemoryTier::Device);
        let mut roundtrip = vec![0.0f32; input.len()];
        device
            .download(&staging, &mut roundtrip)
            .expect("failed to download from host-pinned tier");
        assert_eq!(roundtrip, input);
    }

    #[test]
    fn copy_on_write_preserves_values_in_both_device_buffers() {
        let device = crate::backend::get_cuda_device();
        let input = vec![1.0f32, -2.5, 3.25, 8.0];
        let source = device
            .upload_with_hint(&input, PlacementHint::Tier(MemoryTier::Device))
            .expect("failed to upload COW source");
        let mut writable = CudaStorage {
            buffer: Arc::new(source),
        };
        let retained = writable.clone();

        writable.make_unique();

        assert!(!Arc::ptr_eq(&writable.buffer, &retained.buffer));
        let mut writable_values = vec![0.0f32; input.len()];
        let mut retained_values = vec![0.0f32; input.len()];
        device
            .download(&writable.buffer, &mut writable_values)
            .expect("failed to download detached COW buffer");
        device
            .download(&retained.buffer, &mut retained_values)
            .expect("failed to download retained COW buffer");

        assert_eq!(writable_values, input);
        assert_eq!(retained_values, input);
    }
}
