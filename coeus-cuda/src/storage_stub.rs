use coeus_core::{
    CpuAddressableStorage, CpuAddressableStorageMut, CpuStorage, Scalar, Storage, StorageMut,
};

/// CPU-backed storage used when `coeus-cuda` is compiled without CUDA support.
#[derive(Clone)]
pub struct CudaStorage<T: Scalar> {
    inner: CpuStorage<T>,
}

impl<T: Scalar> coeus_core::storage::private::Sealed for CudaStorage<T> {}

impl<T: Scalar> CudaStorage<T> {
    /// Allocate fallback host storage.
    #[inline]
    pub fn new(len: usize) -> Self {
        Self {
            inner: CpuStorage::new(len),
        }
    }

    /// No CUDA device pointer exists in the no-CUDA build.
    #[inline]
    pub const fn cu_deviceptr(&self) -> u64 {
        0
    }
}

impl<T: Scalar> Storage<T> for CudaStorage<T> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    fn allocate(len: usize) -> Self {
        Self::new(len)
    }

    #[inline]
    fn try_as_slice(&self) -> Option<&[T]> {
        Some(self.inner.as_slice())
    }
}

impl<T: Scalar> StorageMut<T> for CudaStorage<T> {
    #[inline]
    fn try_as_mut_slice(&mut self) -> Option<&mut [T]> {
        Some(self.inner.as_mut_slice())
    }

    #[inline]
    fn make_unique(&mut self) {
        self.inner.make_unique();
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
    fn as_mut_slice(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }
}
