use coeus_core::{Backend, BackendError, ComputeBackend, CpuStorage, Scalar, SequentialBackend};

/// Scalar types supported by the CUDA backend and Hephaestus fusion.
pub trait CudaScalar: Scalar + leto_ops::Scalar + hephaestus_cuda::CudaFusionScalar {}

impl CudaScalar for f32 {}
impl CudaScalar for f64 {}
impl CudaScalar for eunomia::F16 {}
impl CudaScalar for eunomia::Bf16 {}
impl CudaScalar for i32 {}

/// CUDA metadata backend compiled without CUDA provider support.
///
/// This selector uses CPU storage for metadata-only builds and implements no
/// Coeus mathematical backend traits.
/// Selecting CUDA execution requires the crate's `cuda` feature.
///
/// # Examples
///
/// ```
/// use coeus_cuda::CudaBackend;
/// use coeus_core::ComputeBackend;
///
/// let backend = CudaBackend::new();
/// assert_eq!(backend.name(), "cuda-unavailable");
/// assert_eq!(backend.num_threads(), 1);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CudaBackend;

impl CudaBackend {
    /// Create a new backend instance.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

// SAFETY: the stub delegates safe storage operations to SequentialBackend,
// whose CPU storage tracks initialization and full writes.
unsafe impl ComputeBackend for CudaBackend {
    type Error = BackendError;
    type DeviceBuffer<T: Scalar> = CpuStorage<T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = std::future::Ready<T>;

    #[inline]
    fn name(&self) -> &'static str {
        "cuda-unavailable"
    }

    #[inline]
    fn num_threads(&self) -> usize {
        1
    }

    #[inline]
    unsafe fn allocate<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        // SAFETY: the CPU fallback returns the same uninitialized storage, so
        // the caller's initialization obligation remains unchanged.
        unsafe { SequentialBackend::new().allocate(len) }
    }

    #[inline]
    fn allocate_zeroed<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        SequentialBackend::new().allocate_zeroed(len)
    }

    #[inline]
    fn fill<T: Scalar>(
        &self,
        dst: &mut Self::DeviceBuffer<T>,
        value: T,
    ) -> Result<(), Self::Error> {
        SequentialBackend::new().fill(dst, value)
    }

    #[inline]
    fn copy_to_device<T: Scalar>(
        &self,
        src: &[T],
        dst: &mut Self::DeviceBuffer<T>,
    ) -> Result<(), Self::Error> {
        SequentialBackend::new().copy_to_device(src, dst)
    }

    #[inline]
    fn copy_to_host<T: Scalar>(
        &self,
        src: &Self::DeviceBuffer<T>,
        dst: &mut [T],
    ) -> Result<(), Self::Error> {
        SequentialBackend::new().copy_to_host(src, dst)
    }
}

// SAFETY: the feature-disabled backend delegates to synchronous CPU dispatch.
unsafe impl Backend for CudaBackend {
    #[inline]
    fn parallel_for<F>(&self, start: usize, end: usize, operation: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        coeus_core::SequentialBackend::new().parallel_for(start, end, operation);
    }
}
