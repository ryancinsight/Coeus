use crate::storage::CudaStorage;
use coeus_core::{Backend, ComputeBackend, Scalar, Storage, StorageMut};

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

impl CudaScalar for half::f16 {
    const CUDA_TYPE: &'static str = "__half";
}

impl CudaScalar for half::bf16 {
    const CUDA_TYPE: &'static str = "__nv_bfloat16";
}

impl CudaScalar for i32 {
    const CUDA_TYPE: &'static str = "int";
}

/// CUDA API-compatible backend compiled without CUDA provider support.
///
/// Its storage is CPU-addressable and all mathematical operations use Coeus'
/// canonical generic CPU kernels. This preserves value semantics on hosts that
/// cannot compile CUDA without maintaining a second operation implementation.
///
/// # Examples
///
/// ```
/// use coeus_cuda::CudaBackend;
/// use coeus_core::ComputeBackend;
///
/// let backend = CudaBackend::new();
/// assert_eq!(backend.name(), "cuda-cpu");
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
    type DeviceBuffer<T: Scalar> = CudaStorage<T>;
    type KernelDescriptor = ();
    type DispatchFuture<T: Scalar> = std::future::Ready<T>;

    #[inline]
    fn name(&self) -> &'static str {
        "cuda-cpu"
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

impl Backend for CudaBackend {
    #[inline]
    fn parallel_for<F>(&self, start: usize, end: usize, operation: F)
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        coeus_core::SequentialBackend::new().parallel_for(start, end, operation);
    }
}

impl coeus_ops::CpuBackend for CudaBackend {
    #[inline]
    fn as_mut_slice_i64<'a>(&self, buffer: &'a mut Self::DeviceBuffer<i64>) -> &'a mut [i64] {
        buffer
            .try_as_mut_slice()
            .expect("invariant: no-CUDA storage is CPU-addressable")
    }
}
