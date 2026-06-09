use crate::storage::WgpuStorage;
use coeus_core::{ComputeBackend, Scalar, Storage};
use std::sync::OnceLock;

pub mod ops;

/// Trait mapping CPU types to their WGSL representation types on the GPU.
pub trait WgpuScalar: Scalar + bytemuck::Pod {
    const WGSL_TYPE: &'static str;
    const WGSL_ZERO: &'static str;
    const WGSL_ONE: &'static str;
}

impl WgpuScalar for f32 {
    const WGSL_TYPE: &'static str = "f32";
    const WGSL_ZERO: &'static str = "0.0";
    const WGSL_ONE: &'static str = "1.0";
}

impl WgpuScalar for i32 {
    const WGSL_TYPE: &'static str = "i32";
    const WGSL_ZERO: &'static str = "0";
    const WGSL_ONE: &'static str = "1";
}

impl WgpuScalar for u32 {
    const WGSL_TYPE: &'static str = "u32";
    const WGSL_ZERO: &'static str = "0u";
    const WGSL_ONE: &'static str = "1u";
}

/// Context holding the active wgpu connection.
pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub staging_pool: std::sync::Mutex<Vec<wgpu::Buffer>>,
    pub metadata_pool: std::sync::Mutex<Vec<wgpu::Buffer>>,
}

impl WgpuContext {
    /// Retrieve a metadata buffer from the pool, or create a new one.
    pub fn get_metadata_buffer(&self) -> wgpu::Buffer {
        let mut pool = self.metadata_pool.lock().unwrap();
        if let Some(buf) = pool.pop() {
            buf
        } else {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("coeus-wgpu-metadata-buffer"),
                size: 1024,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        }
    }

    /// Recycle a metadata buffer back to the pool.
    pub fn recycle_metadata_buffer(&self, buf: wgpu::Buffer) {
        let mut pool = self.metadata_pool.lock().unwrap();
        pool.push(buf);
    }
}

/// RAII guard that manages a pooled WebGPU metadata buffer.
pub struct PooledMetadataBuffer {
    buffer: Option<wgpu::Buffer>,
}

impl PooledMetadataBuffer {
    /// Get a buffer from the global metadata pool.
    pub fn new() -> Self {
        let ctx = get_wgpu_context();
        let buffer = ctx.get_metadata_buffer();
        Self {
            buffer: Some(buffer),
        }
    }
}

impl std::ops::Deref for PooledMetadataBuffer {
    type Target = wgpu::Buffer;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().unwrap()
    }
}

impl Drop for PooledMetadataBuffer {
    #[inline]
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            get_wgpu_context().recycle_metadata_buffer(buf);
        }
    }
}

static WGPU_CONTEXT: OnceLock<WgpuContext> = OnceLock::new();

/// Retrieve a reference to the global lazily-initialized wgpu context.
pub fn get_wgpu_context() -> &'static WgpuContext {
    WGPU_CONTEXT.get_or_init(|| {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .expect("Failed to find a wgpu adapter");

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("coeus-wgpu-device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::default(),
                    },
                    None,
                )
                .await
                .expect("Failed to create wgpu device");

            WgpuContext {
                device,
                queue,
                staging_pool: std::sync::Mutex::new(Vec::new()),
                metadata_pool: std::sync::Mutex::new(Vec::new()),
            }
        })
    })
}

/// WebGPU acceleration backend.
///
/// # ZST
/// Encoded as a Zero-Sized Type to guarantee static routing and zero runtime context overhead.
#[derive(Debug, Clone, Copy, Default)]
pub struct WgpuBackend;

impl coeus_core::backend::private::Sealed for WgpuBackend {}

impl WgpuBackend {
    /// Create a new instance of the WebGPU backend ZST.
    pub const fn new() -> Self {
        Self
    }
}

impl ComputeBackend for WgpuBackend {
    type DeviceBuffer<T: Scalar> = WgpuStorage<T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = std::future::Ready<T>;

    #[inline]
    fn name(&self) -> &'static str {
        "wgpu"
    }

    #[inline]
    fn num_threads(&self) -> usize {
        1
    }

    #[inline]
    fn allocate<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        WgpuStorage::allocate(len)
    }

    #[inline]
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) {
        let size = dst.len();
        let data = vec![val; size];
        self.copy_to_device(&data, dst);
    }

    #[inline]
    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>) {
        let ctx = get_wgpu_context();
        let bytes = bytemuck::cast_slice(src);
        ctx.queue.write_buffer(&dst.buffer, 0, bytes);
    }

    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        let ctx = get_wgpu_context();
        let size_in_bytes = (src.len * std::mem::size_of::<T>()).max(4) as u64;

        let staging_buffer = {
            let mut pool = ctx.staging_pool.lock().unwrap();
            let mut found_idx = None;
            for (idx, buf) in pool.iter().enumerate() {
                if buf.size() >= size_in_bytes {
                    found_idx = Some(idx);
                    break;
                }
            }
            if let Some(idx) = found_idx {
                pool.remove(idx)
            } else {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("coeus-wgpu-staging-read"),
                    size: size_in_bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            }
        };

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("coeus-wgpu-read-encoder"),
            });
        encoder.copy_buffer_to_buffer(&src.buffer, 0, &staging_buffer, 0, size_in_bytes);
        ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..size_in_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .unwrap()
            .expect("Failed to map staging buffer for read");

        let data = buffer_slice.get_mapped_range();
        let dst_bytes = bytemuck::cast_slice_mut(dst);
        dst_bytes.copy_from_slice(&data);

        drop(data);
        staging_buffer.unmap();

        // Recycle the buffer back to the pool
        let mut pool = ctx.staging_pool.lock().unwrap();
        pool.push(staging_buffer);
    }
}
