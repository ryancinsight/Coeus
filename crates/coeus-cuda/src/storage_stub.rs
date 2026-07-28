use coeus_core::{
    BackendError, CpuAddressableStorage, CpuAddressableStorageMut, CpuStorage, Scalar, Storage,
    StorageMut,
};

/// CPU-backed storage used when `coeus-cuda` is compiled without CUDA support.
///
/// # Examples
///
/// ```
/// use coeus_cuda::CudaStorage;
/// use coeus_core::Storage;
///
/// let storage: CudaStorage<f32> = CudaStorage::try_new(4).expect("allocation succeeds");
/// assert_eq!(storage.len(), 4);
/// assert_eq!(storage.cu_deviceptr(), 0);
/// ```
#[derive(Clone)]
pub struct CudaStorage<T: Scalar> {
    inner: CpuStorage<T>,
}

impl<T: Scalar> coeus_core::storage::private::Sealed for CudaStorage<T> {}

impl<T: Scalar> CudaStorage<T> {
    /// Allocate fallback host storage without masking an allocation failure.
    #[inline]
    pub fn try_new(len: usize) -> Result<Self, BackendError> {
        Ok(Self {
            inner: CpuStorage::try_new(len)?,
        })
    }

    /// No CUDA device pointer exists in the no-CUDA build.
    #[inline]
    pub const fn cu_deviceptr(&self) -> u64 {
        0
    }
}

impl<T: Scalar> Storage<T> for CudaStorage<T> {
    type Error = BackendError;

    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    fn try_allocate(len: usize) -> Result<Self, Self::Error> {
        Self::try_new(len)
    }

    #[inline]
    fn try_as_slice(&self) -> Option<&[T]> {
        Some(self.inner.as_slice())
    }
}

impl<T: Scalar> StorageMut<T> for CudaStorage<T> {
    #[inline]
    fn try_as_mut_slice(&mut self) -> Result<Option<&mut [T]>, Self::Error> {
        Ok(Some(self.inner.as_mut_slice()?))
    }

    #[inline]
    fn make_unique(&mut self) -> Result<(), Self::Error> {
        self.inner.make_unique()
    }
}

impl<T: Scalar> CpuAddressableStorage<T> for CudaStorage<T> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }
}

impl<T: Scalar> CpuAddressableStorageMut<T> for CudaStorage<T> {
    #[inline]
    fn as_mut_slice(&mut self) -> Result<&mut [T], Self::Error> {
        self.inner.as_mut_slice()
    }
}
