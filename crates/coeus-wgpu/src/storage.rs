use crate::backend::get_wgpu_context;
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
    #[inline]
    fn alloc_device_zeroed(len: usize) -> hephaestus_wgpu::WgpuBuffer<T> {
        let ctx = get_wgpu_context();
        ctx.hephaestus_device
            .alloc_zeroed_with_hint(len, PlacementHint::Tier(MemoryTier::Device))
            .expect("Failed to allocate GPU buffer in device tier")
    }

    #[inline]
    fn alloc_device_uninitialized(len: usize) -> hephaestus_wgpu::WgpuBuffer<T> {
        let ctx = get_wgpu_context();
        ctx.hephaestus_device
            .alloc_uninitialized_with_hint(len, PlacementHint::Tier(MemoryTier::Device))
            .expect("Failed to allocate GPU buffer in device tier")
    }

    /// Allocate a new GPU buffer for `len` elements.
    pub fn new(len: usize) -> Self {
        let buffer = Self::alloc_device_zeroed(len);
        Self {
            buffer: Arc::new(buffer),
        }
    }

    #[inline]
    pub(crate) fn uninitialized(len: usize) -> Self {
        let buffer = Self::alloc_device_uninitialized(len);
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
            let ctx = get_wgpu_context();
            let new_buffer = Self::alloc_device_uninitialized(self.buffer.len());
            ctx.hephaestus_device
                .copy_buffer(self.buffer.as_ref(), &new_buffer)
                .expect("WgpuStorage::make_unique failed to copy the device buffer");

            self.buffer = Arc::new(new_buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::ComputeBackend;

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
    fn backend_zero_memory_operations_preserve_exact_values() {
        let backend = crate::backend::WgpuBackend::new();
        let mut storage = backend.allocate_zeroed::<u32>(4);
        let mut values = [u32::MAX; 4];
        backend.copy_to_host(&storage, &mut values);
        assert_eq!(values, [0; 4]);

        backend.fill(&mut storage, 0xdead_beef);
        backend.fill(&mut storage, 0);
        backend.copy_to_host(&storage, &mut values);
        assert_eq!(values, [0; 4]);
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

    #[test]
    fn copy_on_write_preserves_values_in_both_device_buffers() {
        let ctx = get_wgpu_context();
        let input = vec![1.0f32, -2.5, 3.25, 8.0];
        let source = ctx
            .hephaestus_device
            .upload_with_hint(&input, PlacementHint::Tier(MemoryTier::Device))
            .expect("failed to upload COW source");
        let mut writable = WgpuStorage {
            buffer: Arc::new(source),
        };
        let retained = writable.clone();

        writable.make_unique();

        assert!(!Arc::ptr_eq(&writable.buffer, &retained.buffer));
        let mut writable_values = vec![0.0f32; input.len()];
        let mut retained_values = vec![0.0f32; input.len()];
        ctx.hephaestus_device
            .download(&writable.buffer, &mut writable_values)
            .expect("failed to download detached COW buffer");
        ctx.hephaestus_device
            .download(&retained.buffer, &mut retained_values)
            .expect("failed to download retained COW buffer");

        assert_eq!(writable_values, input);
        assert_eq!(retained_values, input);
    }
}
