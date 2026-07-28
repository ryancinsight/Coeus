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
    type Error = BackendError;
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
    fn allocate<T: Scalar>(&self, len: usize) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        CudaStorage::try_new(len)
    }

    #[inline]
    fn fill<T: Scalar>(
        &self,
        dst: &mut Self::DeviceBuffer<T>,
        value: T,
    ) -> Result<(), Self::Error> {
        dst.try_as_mut_slice()?
            .ok_or_else(|| BackendError::Storage {
                operation: "fill",
                reason: "no-CUDA storage is not CPU-addressable".to_owned(),
            })?
            .fill(value);
        Ok(())
    }

    #[inline]
    fn copy_to_device<T: Scalar>(
        &self,
        src: &[T],
        dst: &mut Self::DeviceBuffer<T>,
    ) -> Result<(), Self::Error> {
        if src.len() != dst.len() {
            return Err(BackendError::Storage {
                operation: "copy_to_device",
                reason: format!(
                    "source length {} differs from destination length {}",
                    src.len(),
                    dst.len()
                ),
            });
        }
        dst.try_as_mut_slice()?
            .ok_or_else(|| BackendError::Storage {
                operation: "copy_to_device",
                reason: "no-CUDA storage is not CPU-addressable".to_owned(),
            })?
            .copy_from_slice(src);
        Ok(())
    }

    #[inline]
    fn copy_to_host<T: Scalar>(
        &self,
        src: &Self::DeviceBuffer<T>,
        dst: &mut [T],
    ) -> Result<(), Self::Error> {
        if src.len() != dst.len() {
            return Err(BackendError::Storage {
                operation: "copy_to_host",
                reason: format!(
                    "source length {} differs from destination length {}",
                    src.len(),
                    dst.len()
                ),
            });
        }
        dst.copy_from_slice(src.try_as_slice().ok_or_else(|| BackendError::Storage {
            operation: "copy_to_host",
            reason: "no-CUDA storage is not CPU-addressable".to_owned(),
        })?);
        Ok(())
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
    fn as_mut_slice_i64<'a>(
        &self,
        buffer: &'a mut Self::DeviceBuffer<i64>,
    ) -> Result<&'a mut [i64], BackendError> {
        Ok(buffer
            .try_as_mut_slice()?
            .ok_or_else(|| BackendError::Storage {
                operation: "integer output",
                reason: "no-CUDA storage is not CPU-addressable".to_owned(),
            })?)
    }
}
