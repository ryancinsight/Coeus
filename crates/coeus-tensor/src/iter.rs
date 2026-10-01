// ── Tensor iterators ──
// Simple element-wise iteration via adaptor methods on contiguous tensors.

use crate::tensor::Tensor;
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};

impl<T: Scalar, B: ComputeBackend> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    /// Iterate over contiguous elements by reference.
    ///
    /// # Panics
    /// If the tensor is not contiguous.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        assert!(self.is_contiguous(), "iter requires contiguous tensor");
        let start = self.layout.offset();
        let len = self.numel();
        self.storage.as_slice()[start..start + len].iter()
    }
}

impl<T: Scalar, B: ComputeBackend> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    /// Iterate over contiguous elements mutably.
    ///
    /// # Errors
    /// Returns the backend storage error if copy-on-write allocation or copying fails.
    ///
    /// # Panics
    /// If the tensor is not contiguous.
    #[inline]
    pub fn try_iter_mut(&mut self) -> Result<impl Iterator<Item = &mut T> + '_, B::Error> {
        assert!(
            self.is_contiguous(),
            "try_iter_mut requires contiguous tensor"
        );
        let start = self.layout.offset();
        let len = self.numel();
        let storage = self.storage.as_mut_slice()?;
        Ok(storage[start..start + len].iter_mut())
    }
}
