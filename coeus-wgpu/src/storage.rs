use crate::backend::get_wgpu_context;
use coeus_core::{Scalar, Storage, StorageMut};
use hephaestus_wgpu::{ComputeDevice, DeviceBuffer};
use std::sync::Arc;
use themis::{MemoryTier, PlacementHint};

/// GPU-allocated buffer managed by hephaestus-wgpu.
pub struct WgpuStorage<T> {
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
    #[inline]
    fn alloc_device_zeroed(len: usize) -> hephaestus_wgpu::WgpuBuffer<T> {
        let ctx = get_wgpu_context();
        ctx.hephaestus_device
            .alloc_zeroed_with_hint(len, PlacementHint::Tier(MemoryTier::Device))
            .expect("Failed to allocate GPU buffer in device tier")
    }

    /// Allocate a new GPU buffer for `len` elements.
    pub fn new(len: usize) -> Self {
        let buffer = Self::alloc_device_zeroed(len);
        Self {
            buffer: Arc::new(buffer),
        }
    }
}

impl<T: Scalar> Storage<T> for WgpuStorage<T> {
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

impl<T: Scalar> StorageMut<T> for WgpuStorage<T> {
    #[inline]
    fn try_as_mut_slice(&mut self) -> Option<&mut [T]> {
        None
    }

    fn make_unique(&mut self) {
        if Arc::strong_count(&self.buffer) > 1 {
            let len = self.buffer.len();
            let ctx = get_wgpu_context();
            let new_buffer = Self::alloc_device_zeroed(len);

            let size_in_bytes = (len * std::mem::size_of::<T>()).max(4) as u64;

            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("coeus-wgpu-cow-copy"),
                });
            encoder.copy_buffer_to_buffer(self.buffer.raw(), 0, new_buffer.raw(), 0, size_in_bytes);
            ctx.queue.submit(std::iter::once(encoder.finish()));

            self.buffer = Arc::new(new_buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn storage_allocates_device_tier() {
        let storage = WgpuStorage::<f32>::new(16);
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
    fn host_pinned_upload_uses_staging_tier() {
        let ctx = get_wgpu_context();
        let input = vec![1.0f32, -2.5, 3.25, 8.0];
        let staging = ctx
            .hephaestus_device
            .upload_with_hint(&input, PlacementHint::Tier(MemoryTier::HostPinned))
            .expect("failed to upload into host-pinned staging tier");
        assert_eq!(staging.tier(), MemoryTier::HostPinned);
    }
}
