use crate::Tensor;
use backend::{Backend, DataType};
use storage::{Storage, StorageFromVec, StorageToDense};

/// Returns a new tensor with the same data but a different shape.
pub fn view<B, S, T>(
    tensor: &Tensor<B, S, T>,
    dims: &[isize],
) -> crate::Result<Tensor<B, crate::DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Default + Clone + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
    T: DataType + Clone + 'static,
{
    crate::ops::shape::reshape::reshape(tensor, dims)
}

impl<B, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T> + Default + Clone + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
    T: DataType + Clone + 'static,
{
    /// Returns a new tensor with the same data but a different shape.
    ///
    /// The returned tensor shares the same data and must have the same number
    /// of elements, but may have a different size. For a tensor to be viewed, the new
    /// view size must be compatible with its original size and stride, i.e., each new
    /// view dimension must either be a subspace of an original dimension, or only span
    /// across original dimensions d, d+1, ..., d+k that satisfy the following
    /// contiguity-like condition that for all i = 0, ..., k-1, stride[i] = stride[i+1] * size[i+1].
    ///
    /// Currently, implementation falls back to `reshape` which ensures contiguous memory layout.
    /// Future work will support true strided views.
    ///
    /// # Arguments
    /// * `dims` - New dimension sizes.
    ///
    /// # Returns
    /// Tensor with new shape.
    pub fn view(&self, dims: &[isize]) -> crate::Result<Tensor<B, crate::DenseStorage<T>, T>> {
        view(self, dims)
    }
}
