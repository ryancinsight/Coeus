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
    pub hephaestus_device: hephaestus_wgpu::WgpuDevice,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
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
        let hephaestus_device = hephaestus_wgpu::WgpuDevice::try_default_with_limits(
            "coeus-wgpu-device",
            wgpu::Limits::default(),
        )
        .expect("Failed to initialize hephaestus-wgpu device");
        let device = (**hephaestus_device.device()).clone();
        let queue = (**hephaestus_device.queue()).clone();
        WgpuContext {
            hephaestus_device,
            device,
            queue,
            metadata_pool: std::sync::Mutex::new(Vec::new()),
        }
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
        ctx.queue.write_buffer(dst.buffer.raw(), 0, bytes);
    }

    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        let ctx = get_wgpu_context();
        let size_in_bytes = (src.len() * std::mem::size_of::<T>()).max(4) as u64;

        let staging_buffer = ctx.hephaestus_device.get_staging_buffer(size_in_bytes);

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("coeus-wgpu-read-encoder"),
        });
        encoder.copy_buffer_to_buffer(src.buffer.raw(), 0, &staging_buffer, 0, size_in_bytes);
        ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..size_in_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        let _ = ctx.device.poll(wgpu::PollType::Wait);
        rx.recv()
            .unwrap()
            .expect("Failed to map staging buffer for read");

        let data = buffer_slice.get_mapped_range();
        let dst_bytes = bytemuck::cast_slice_mut(dst);
        dst_bytes.copy_from_slice(&data);

        drop(data);
        staging_buffer.unmap();

        ctx.hephaestus_device.recycle_staging_buffer(staging_buffer);
    }
}
