use crate::storage::WgpuStorage;
use coeus_core::{ComputeBackend, Scalar, Storage};
use hephaestus_core::{CommandStream, KernelDevice};
use hephaestus_wgpu::ComputeDevice;
use std::sync::OnceLock;

mod error;
pub mod ops;

pub(crate) use error::{checked_numel, checked_workgroup_count};
pub use error::{LayoutError, WgpuBackendError};

pub(crate) const METADATA_BUFFER_SIZE: u64 = 1024;
const METADATA_POOL_CAPACITY: usize = 64;

/// Trait mapping CPU types to their WGSL representation types on the GPU.
///
/// # Example
///
/// ```
/// use coeus_wgpu::WgpuScalar;
///
/// assert_eq!(f32::WGSL_TYPE, "f32");
/// assert_eq!(f32::WGSL_ZERO, "0.0");
/// assert_eq!(f32::WGSL_ONE, "1.0");
///
/// assert_eq!(i32::WGSL_TYPE, "i32");
/// assert_eq!(i32::WGSL_ZERO, "0");
///
/// assert_eq!(u32::WGSL_TYPE, "u32");
/// assert_eq!(u32::WGSL_ZERO, "0u");
/// ```
pub trait WgpuScalar: Scalar + hephaestus_wgpu::WgpuFusionScalar {
    /// WGSL type name string (e.g. `"f32"`).
    const WGSL_TYPE: &'static str;
    /// WGSL zero literal string (e.g. `"0.0"`).
    const WGSL_ZERO: &'static str;
    /// WGSL one literal string (e.g. `"1.0"`).
    const WGSL_ONE: &'static str;
    /// Lowest finite WGSL value used to initialize maximum reductions.
    const WGSL_LOWEST: &'static str;
    /// Highest finite WGSL value used to initialize minimum reductions.
    const WGSL_HIGHEST: &'static str;
}

impl WgpuScalar for f32 {
    const WGSL_TYPE: &'static str = "f32";
    const WGSL_ZERO: &'static str = "0.0";
    const WGSL_ONE: &'static str = "1.0";
    const WGSL_LOWEST: &'static str = "-3.40282347e+38";
    const WGSL_HIGHEST: &'static str = "3.40282347e+38";
}

impl WgpuScalar for i32 {
    const WGSL_TYPE: &'static str = "i32";
    const WGSL_ZERO: &'static str = "0";
    const WGSL_ONE: &'static str = "1";
    const WGSL_LOWEST: &'static str = "(-2147483647 - 1)";
    const WGSL_HIGHEST: &'static str = "2147483647";
}

impl WgpuScalar for u32 {
    const WGSL_TYPE: &'static str = "u32";
    const WGSL_ZERO: &'static str = "0u";
    const WGSL_ONE: &'static str = "1u";
    const WGSL_LOWEST: &'static str = "0u";
    const WGSL_HIGHEST: &'static str = "4294967295u";
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
        if let Ok(mut pool) = self.metadata_pool.try_lock() {
            if let Some(buf) = pool.pop() {
                return buf;
            }
        }
        self.create_metadata_buffer()
    }

    /// Recycle a metadata buffer back to the pool.
    pub fn recycle_metadata_buffer(&self, buf: wgpu::Buffer) {
        if let Ok(mut pool) = self.metadata_pool.try_lock() {
            if pool.len() < METADATA_POOL_CAPACITY {
                pool.push(buf);
            }
        }
    }

    fn create_metadata_buffer(&self) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coeus-wgpu-metadata-buffer"),
            size: METADATA_BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
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
    try_get_wgpu_context().expect("Failed to initialize hephaestus-wgpu device")
}

/// Try to retrieve the process-global WGPU context.
///
/// # Errors
///
/// Returns the typed Hephaestus acquisition failure when WGPU is unavailable.
pub fn try_get_wgpu_context() -> hephaestus_core::Result<&'static WgpuContext> {
    if let Some(context) = WGPU_CONTEXT.get() {
        return Ok(context);
    }
    // No backend is forced here. Hephaestus selects the compiled backend set
    // unless the process explicitly sets `WGPU_BACKEND`; a request for an
    // unavailable backend therefore fails through the provider's typed
    // acquisition error. Backend selection belongs to the provider and to
    // whoever runs the process, not to this library.
    let hephaestus_device = hephaestus_wgpu::WgpuDevice::try_default_with_limits(
        "coeus-wgpu-device",
        wgpu::Limits::default(),
    )?;
    let device = (**hephaestus_device.device()).clone();
    let queue = (**hephaestus_device.queue()).clone();
    let candidate = WgpuContext {
        hephaestus_device,
        device,
        queue,
        metadata_pool: std::sync::Mutex::new(Vec::new()),
    };
    let _ = WGPU_CONTEXT.set(candidate);
    WGPU_CONTEXT
        .get()
        .ok_or_else(|| hephaestus_core::HephaestusError::DeviceUnavailable {
            message: "WGPU context initialization did not publish the acquired device".to_owned(),
        })
}

/// WebGPU acceleration backend.
///
/// # ZST
/// Encoded as a Zero-Sized Type to guarantee static routing and zero runtime context overhead.
///
/// # Example
///
/// ```
/// use coeus_wgpu::WgpuBackend;
/// use coeus_core::ComputeBackend;
///
/// let backend = WgpuBackend::new();
/// assert_eq!(backend.name(), "wgpu");
/// assert_eq!(backend.num_threads(), 1);
///
/// // ZST: occupies no memory
/// assert_eq!(std::mem::size_of::<WgpuBackend>(), 0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct WgpuBackend;

impl WgpuBackend {
    /// Create a new instance of the WebGPU backend ZST.
    ///
    /// # Example
    ///
    /// ```
    /// use coeus_wgpu::WgpuBackend;
    /// use coeus_core::ComputeBackend;
    ///
    /// let backend = WgpuBackend::new();
    /// let default = WgpuBackend::default();
    /// assert_eq!(backend.name(), default.name());
    /// ```
    pub const fn new() -> Self {
        Self
    }
}

impl ComputeBackend for WgpuBackend {
    type Error = WgpuBackendError;
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
        WgpuStorage::uninitialized(len)
    }

    #[inline]
    fn allocate_zeroed<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        WgpuStorage::new(len)
    }

    #[inline]
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, val: T) {
        if val.has_zero_bit_pattern() {
            self.fill_zero(dst);
            return;
        }
        let size = dst.len();
        let data = vec![val; size];
        self.copy_to_device(&data, dst);
    }

    #[inline]
    fn fill_zero<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>) {
        let device = &get_wgpu_context().hephaestus_device;
        let mut stream = device
            .stream()
            .expect("WGPU zero fill stream creation failed");
        stream
            .fill_zero(dst.buffer.as_ref())
            .expect("WGPU zero fill encoding failed");
        stream.submit().expect("WGPU zero fill submission failed");
    }

    #[inline]
    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>) {
        let ctx = get_wgpu_context();
        ctx.hephaestus_device
            .write_buffer(dst.buffer.as_ref(), src)
            .expect("Failed to copy host tensor into WgpuBuffer");
    }

    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        let ctx = get_wgpu_context();
        ctx.hephaestus_device
            .download(src.buffer.as_ref(), dst)
            .expect("Failed to copy WgpuBuffer into host tensor");
    }
}
