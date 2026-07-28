use crate::{driver::CUdeviceptr, CudaBackendError};
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
    /// Allocate a CUDA buffer without masking provider failure as a panic.
    pub fn try_new(len: usize) -> Result<Self, CudaBackendError> {
        let device = crate::backend::get_cuda_device()?;
        let buffer = device
            .alloc_zeroed_with_hint(len, PlacementHint::Tier(MemoryTier::Device))
            .map_err(|source| CudaBackendError::provider("allocate", source))?;
        Ok(Self {
            buffer: Arc::new(buffer),
        })
    }

    /// Retrieve the raw CUDA device pointer.
    #[inline]
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.buffer.raw()
    }
}

impl<T: Scalar> Storage<T> for CudaStorage<T> {
    type Error = CudaBackendError;

    #[inline]
    fn len(&self) -> usize {
        self.buffer.len()
    }

    #[inline]
    fn try_allocate(len: usize) -> Result<Self, Self::Error> {
        Self::try_new(len)
    }

    #[inline]
    fn try_as_slice(&self) -> Option<&[T]> {
        None
    }
}

impl<T: Scalar> StorageMut<T> for CudaStorage<T> {
    #[inline]
    fn try_as_mut_slice(&mut self) -> Result<Option<&mut [T]>, Self::Error> {
        Ok(None)
    }

    fn make_unique(&mut self) -> Result<(), Self::Error> {
        if Arc::strong_count(&self.buffer) > 1 {
            let device = crate::backend::get_cuda_device()?;
            let len = self.buffer.len();
            let new_buffer = Self::try_new(len)?;

            if len > 0 {
                let bytes = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                    CudaBackendError::from(coeus_core::BackendError::Overflow {
                        operation: "cow copy",
                        reason: "element-count to byte-size arithmetic overflow",
                    })
                })?;
                device
                    .bind()
                    .map_err(|source| CudaBackendError::provider("cow copy", source))?;
                unsafe {
                    let res = cuda_core::sys::cuMemcpyDtoD_v2(
                        new_buffer.buffer.raw(),
                        self.buffer.raw(),
                        bytes,
                    );
                    if res != 0 {
                        return Err(CudaBackendError::provider(
                            "cow copy",
                            hephaestus_cuda::HephaestusError::DispatchFailed {
                                message: format!("cuMemcpyDtoD_v2 failed with code {res}"),
                            },
                        ));
                    }
                }
            }

            self.buffer = new_buffer.buffer;
        }
        Ok(())
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
        let device =
            crate::backend::get_cuda_device().expect("test requires an available CUDA device");
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
}
