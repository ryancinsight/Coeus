use std::sync::Arc;
use std::marker::PhantomData;
use coeus_core::{Storage, StorageMut, Scalar};
use crate::backend::get_wgpu_context;

/// GPU-allocated buffer managed by wgpu.
pub struct WgpuStorage<T> {
    pub buffer: Arc<wgpu::Buffer>,
    pub len: usize,
    pub(crate) _marker: PhantomData<T>,
}

impl<T> coeus_core::storage::private::Sealed for WgpuStorage<T> {}

impl<T> Clone for WgpuStorage<T> {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            len: self.len,
            _marker: PhantomData,
        }
    }
}

unsafe impl<T: Send> Send for WgpuStorage<T> {}
unsafe impl<T: Sync> Sync for WgpuStorage<T> {}

impl<T: Scalar> WgpuStorage<T> {
    /// Allocate a new GPU buffer for `len` elements.
    pub fn new(len: usize) -> Self {
        let ctx = get_wgpu_context();
        let size_in_bytes = (len * std::mem::size_of::<T>()).max(4) as u64; // wgpu requires min 4 bytes
        
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coeus-wgpu-buffer"),
            size: size_in_bytes,
            usage: wgpu::BufferUsages::STORAGE 
                | wgpu::BufferUsages::COPY_SRC 
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer: Arc::new(buffer),
            len,
            _marker: PhantomData,
        }
    }
}

impl<T: Scalar> Storage<T> for WgpuStorage<T> {
    #[inline]
    fn len(&self) -> usize {
        self.len
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
            let size_in_bytes = (self.len * std::mem::size_of::<T>()).max(4) as u64;
            
            let new_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("coeus-wgpu-cow-buffer"),
                size: size_in_bytes,
                usage: wgpu::BufferUsages::STORAGE 
                    | wgpu::BufferUsages::COPY_SRC 
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("coeus-wgpu-cow-copy"),
            });
            encoder.copy_buffer_to_buffer(
                &self.buffer,
                0,
                &new_buffer,
                0,
                size_in_bytes,
            );
            ctx.queue.submit(std::iter::once(encoder.finish()));
            
            self.buffer = Arc::new(new_buffer);
        }
    }
}
