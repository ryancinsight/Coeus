use crate::backend::get_wgpu_context;
use coeus_core::{Scalar, Storage, StorageMut};
use hephaestus_wgpu::{ComputeDevice, DeviceBuffer};
use std::sync::Arc;

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
    /// Allocate a new GPU buffer for `len` elements.
    pub fn new(len: usize) -> Self {
        let ctx = get_wgpu_context();
        let buffer = ctx
            .hephaestus_device
            .alloc_zeroed::<T>(len)
            .expect("Failed to allocate GPU buffer");
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
            let len = self.buffer.len();
            let new_buffer = ctx
                .hephaestus_device
                .alloc_zeroed::<T>(len)
                .expect("Failed to allocate CoW buffer");

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
