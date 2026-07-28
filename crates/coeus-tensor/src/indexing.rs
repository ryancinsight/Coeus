// ── Tensor indexing ──
// Multi-dimensional get/set helpers.

use crate::tensor::Tensor;
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Scalar};

impl<T: Scalar, B: ComputeBackend> Tensor<T, B> {
    /// Compute 1-D physical offset from logical coordinates.
    #[inline]
    pub fn physical_index(&self, index: &[usize]) -> usize {
        self.layout.physical_index(index)
    }
}

impl<T: Scalar, B: ComputeBackend> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorage<T>,
{
    /// Get element at a 1-D flat offset.
    #[inline]
    pub fn get_flat(&self, offset: usize) -> T {
        self.storage.as_slice()[self.layout.offset() + offset]
    }
}

impl<T: Scalar, B: ComputeBackend> Tensor<T, B>
where
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    /// Set element at a 1-D flat offset.
    ///
    /// # Errors
    /// Returns the backend storage error if copy-on-write cannot allocate a
    /// unique mutable buffer.
    #[inline]
    pub fn set_flat(&mut self, offset: usize, val: T) -> Result<(), B::Error> {
        let base = self.layout.offset();
        let slice = self.storage.as_mut_slice()?;
        slice[base + offset] = val;
        Ok(())
    }
}
