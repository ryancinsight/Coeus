use crate::backend::{get_wgpu_context, try_get_wgpu_context, WgpuBackendError};
use coeus_core::{Scalar, Storage, StorageMut};
use hephaestus_wgpu::{ComputeDevice, DeviceBuffer};
use std::sync::Arc;
use themis::{MemoryTier, PlacementHint};

/// GPU-allocated buffer managed by hephaestus-wgpu.
pub struct WgpuStorage<T> {
    /// Underlying GPU buffer handle.
    pub buffer: Arc<hephaestus_wgpu::WgpuBuffer<T>>,
}

impl<T> coeus_core::storage::private::Sealed for WgpuStorage<T> {}

impl<T> Clone for WgpuStorage<T> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
        }
    }
}

unsafe impl<T: Send> Send for WgpuStorage<T> {}
unsafe impl<T: Sync> Sync for WgpuStorage<T> {}

impl<T: Scalar> WgpuStorage<T> {
    /// Allocate a device buffer while preserving provider allocation errors.
    pub fn try_new(len: usize) -> Result<Self, WgpuBackendError> {
        let ctx = try_get_wgpu_context()?;
        let buffer = ctx
            .hephaestus_device
            .alloc_zeroed_with_hint(len, PlacementHint::Tier(MemoryTier::Device))
            .map_err(|source| WgpuBackendError::dispatch("allocate", source))?;
        Ok(Self {
            buffer: Arc::new(buffer),
        })
    }
}

impl<T: Scalar> Storage<T> for WgpuStorage<T> {
    type Error = WgpuBackendError;

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

impl<T: Scalar> StorageMut<T> for WgpuStorage<T> {
    #[inline]
    fn try_as_mut_slice(&mut self) -> Result<Option<&mut [T]>, Self::Error> {
        Ok(None)
    }

    fn make_unique(&mut self) -> Result<(), Self::Error> {
        if Arc::strong_count(&self.buffer) > 1 {
            let len = self.buffer.len();
            let ctx = get_wgpu_context();
            let new_buffer = Self::try_new(len)?.buffer;

            let size_in_bytes = len
                .checked_mul(std::mem::size_of::<T>())
                .ok_or_else(|| {
                    WgpuBackendError::Validation(coeus_core::BackendError::Overflow {
                        operation: "cow copy",
                        reason: "element-count to byte-size arithmetic overflow",
                    })
                })?
                .max(4) as u64;

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("coeus-wgpu-cow-copy"),
                });
            encoder.copy_buffer_to_buffer(self.buffer.raw(), 0, new_buffer.raw(), 0, size_in_bytes);
            ctx.queue.submit(std::iter::once(encoder.finish()));

            self.buffer = new_buffer;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_allocates_device_tier() {
        let storage = WgpuStorage::<f32>::try_new(16).expect("allocation succeeds");
        assert_eq!(storage.buffer.tier(), MemoryTier::Device);
    }

    #[test]
    fn device_upload_roundtrip_preserves_values() {
        let ctx = get_wgpu_context();
        let input = vec![1.0f32, -2.5, 3.25, 8.0];
        let device_buf = ctx
            .hephaestus_device
            .upload_with_hint(&input, PlacementHint::Tier(MemoryTier::Device))
            .expect("failed to upload into device tier");
        assert_eq!(device_buf.tier(), MemoryTier::Device);
        let mut out = vec![0.0f32; input.len()];
        ctx.hephaestus_device
            .download(&device_buf, &mut out)
            .expect("failed to download from device tier");
        assert_eq!(out, input);
    }

    #[test]
    fn host_pinned_upload_is_rejected_without_false_tier() {
        let ctx = get_wgpu_context();
        let input = vec![1.0f32, -2.5, 3.25, 8.0];
        let error = ctx
            .hephaestus_device
            .upload_with_hint(&input, PlacementHint::Tier(MemoryTier::HostPinned))
            .expect_err("WGPU cannot guarantee persistent host-pinned placement");
        match error {
            hephaestus_core::HephaestusError::AllocationFailed { message } => assert_eq!(
                message,
                "WGPU cannot guarantee requested memory tier HostPinned; use Device placement"
            ),
            other => panic!("expected allocation failure, got {other:?}"),
        }
    }
}
