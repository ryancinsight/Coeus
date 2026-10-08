use coeus_core::{ComputeBackend, Scalar, Storage, StorageMut};
use coeus_hephaestus::get_or_try_init;
use hephaestus_core::{CommandStream, ComputeDevice, DeviceFeature, KernelDevice};
use std::sync::OnceLock;

mod error;
pub mod ops;

pub(crate) use error::checked_numel;
pub use error::WgpuBackendError;

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

impl WgpuScalar for eunomia::F16 {
    const WGSL_TYPE: &'static str = "f16";
    const WGSL_ZERO: &'static str = "0.0";
    const WGSL_ONE: &'static str = "1.0";
    const WGSL_LOWEST: &'static str = "-65504.0";
    const WGSL_HIGHEST: &'static str = "65504.0";
}

impl WgpuScalar for f64 {
    const WGSL_TYPE: &'static str = "f64";
    const WGSL_ZERO: &'static str = "0.0";
    const WGSL_ONE: &'static str = "1.0";
    const WGSL_LOWEST: &'static str = "-1.7976931348623157e+308";
    const WGSL_HIGHEST: &'static str = "1.7976931348623157e+308";
}

/// Context holding the active wgpu connection.
pub struct WgpuContext {
    pub hephaestus_device: hephaestus_wgpu::WgpuDevice,
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
    // No backend is forced here. Hephaestus selects the compiled backend set
    // unless the process explicitly sets `WGPU_BACKEND`; a request for an
    // unavailable backend therefore fails through the provider's typed
    // acquisition error. Backend selection belongs to the provider and to
    // whoever runs the process, not to this library.
    get_or_try_init(
        &WGPU_CONTEXT,
        "WGPU context initialization did not publish the acquired device",
        || {
            Ok(WgpuContext {
            // Fused expressions bind the tensor inputs, output, and layout table.
            // The provider's downlevel baseline exposes only four storage slots,
            // while Coeus' public fusion contract permits three tensor inputs.
            hephaestus_device:
                hephaestus_wgpu::WgpuDevice::try_with_device_preference_and_optional_device_features_and_limits(
                "coeus-wgpu-device",
                hephaestus_core::DevicePreference::HighPerformance,
                // Optional: f64/f16 shaders need SHADER_F64/SHADER_F16 at
                // creation time, but adapters without them must still serve
                // the f32 paths.
                &[DeviceFeature::ShaderF64, DeviceFeature::ShaderF16],
                hephaestus_wgpu::WgpuDevice::default_device_limits(),
            )?,
        })
        },
    )
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

    /// True when the process-global WGPU device serves f64 shaders.
    ///
    /// f64 dispatch needs `SHADER_F64`, which is requested optionally at
    /// device creation: adapters without it still serve every f32 path.
    /// Consumers gate f64 work on this probe instead of failing dispatch.
    #[must_use]
    pub fn supports_f64() -> bool {
        try_get_wgpu_context()
            .map(|context| {
                context
                    .hephaestus_device
                    .supports_device_feature(DeviceFeature::ShaderF64)
            })
            .unwrap_or(false)
    }

    /// True when the process-global WGPU device serves f16 shaders.
    ///
    /// f16 dispatch needs `SHADER_F16`, requested optionally alongside
    /// `SHADER_F64` at device creation. Consumers gate f16 work on this
    /// probe instead of failing dispatch.
    #[must_use]
    pub fn supports_f16() -> bool {
        try_get_wgpu_context()
            .map(|context| {
                context
                    .hephaestus_device
                    .supports_device_feature(DeviceFeature::ShaderF16)
            })
            .unwrap_or(false)
    }
}

impl ComputeBackend for WgpuBackend {
    type Error = WgpuBackendError;
    type DeviceBuffer<T: Scalar> = coeus_hephaestus::HephaestusStorage<crate::WgpuBackend, T>;
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
        coeus_hephaestus::HephaestusBackend::<WgpuBackend>::new().allocate(len)
    }

    #[inline]
    fn allocate_zeroed<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        coeus_hephaestus::HephaestusStorage::<WgpuBackend, _>::new(len)
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
        dst.make_unique();
        let device = &get_wgpu_context().hephaestus_device;
        let mut stream = device
            .stream()
            .expect("WGPU zero fill stream creation failed");
        stream
            .fill_zero(dst.buffer())
            .expect("WGPU zero fill encoding failed");
        stream.submit().expect("WGPU zero fill submission failed");
    }

    #[inline]
    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>) {
        dst.make_unique();
        let ctx = get_wgpu_context();
        ctx.hephaestus_device
            .write_buffer(dst.buffer(), src)
            .expect("Failed to copy host tensor into WgpuBuffer");
    }

    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        let ctx = get_wgpu_context();
        ctx.hephaestus_device
            .download(src.buffer(), dst)
            .expect("Failed to copy WgpuBuffer into host tensor");
    }
}
