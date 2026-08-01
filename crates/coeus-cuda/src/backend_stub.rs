use crate::storage::CudaStorage;
use coeus_core::{Backend, BackendError, ComputeBackend, Scalar, Storage, StorageMut};

/// Trait mapping CPU scalar types to their CUDA type representation.
///
/// # Examples
///
/// ```
/// use coeus_cuda::CudaScalar;
///
/// assert_eq!(f32::CUDA_TYPE, "float");
/// assert_eq!(f64::CUDA_TYPE, "double");
/// assert_eq!(i32::CUDA_TYPE, "int");
/// ```
pub trait CudaScalar: Scalar + leto_ops::Scalar {
    /// CUDA type name string used in NVRTC kernel compilation.
    const CUDA_TYPE: &'static str;
}

impl CudaScalar for f32 {
    const CUDA_TYPE: &'static str = "float";
}

impl CudaScalar for f64 {
    const CUDA_TYPE: &'static str = "double";
}

impl CudaScalar for eunomia::F16 {
    const CUDA_TYPE: &'static str = "__half";
}

impl CudaScalar for eunomia::Bf16 {
    const CUDA_TYPE: &'static str = "__nv_bfloat16";
}

impl CudaScalar for i32 {
    const CUDA_TYPE: &'static str = "int";
}

/// CUDA metadata backend compiled without CUDA provider support.
///
/// This type keeps non-provider builds source-compatible with code that names
/// CUDA storage, but it does not implement Coeus mathematical backend traits.
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

impl coeus_core::backend::private::Sealed for CudaBackend {}

impl CudaBackend {
    /// Create a new backend instance.
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}

impl ComputeBackend for CudaBackend {
    type Error = BackendError;
    type DeviceBuffer<T: Scalar> = CudaStorage<T>;
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
    fn allocate<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        CudaStorage::new(len)
    }

    #[inline]
    fn allocate_zeroed<T: Scalar>(&self, len: usize) -> Self::DeviceBuffer<T> {
        let mut storage = CudaStorage::new(len);
        self.fill_zero(&mut storage);
        storage
    }

    #[inline]
    fn fill<T: Scalar>(&self, dst: &mut Self::DeviceBuffer<T>, value: T) {
        dst.try_as_mut_slice()
            .expect("invariant: no-CUDA storage is CPU-addressable")
            .fill(value);
    }

    #[inline]
    fn copy_to_device<T: Scalar>(&self, src: &[T], dst: &mut Self::DeviceBuffer<T>) {
        dst.try_as_mut_slice()
            .expect("invariant: no-CUDA storage is CPU-addressable")
            .copy_from_slice(src);
    }

    #[inline]
    fn copy_to_host<T: Scalar>(&self, src: &Self::DeviceBuffer<T>, dst: &mut [T]) {
        dst.copy_from_slice(
            src.try_as_slice()
                .expect("invariant: no-CUDA storage is CPU-addressable"),
        );
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
