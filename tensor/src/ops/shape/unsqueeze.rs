use crate::Tensor;
use backend::{Backend, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};

/// Unsqueezes the tensor by inserting a dimension of size 1 at the specified position.
pub fn unsqueeze<B, S, T>(
    tensor: &Tensor<B, S, T>,
    dim: usize,
) -> crate::Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Default + Clone + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
    T: DataType + Clone + 'static,
{
    let current_dims = tensor.shape().dims();
    let ndim = current_dims.len();
    if dim > ndim {
        return Err(crate::TensorError::ShapeError {
            expected: ndim,
            actual: dim,
            message: format!("unsqueeze: dimension {dim} out of bounds for ndim {ndim}"),
        });
    }

    let mut new_dims = Vec::with_capacity(ndim + 1);
    new_dims.extend(current_dims.iter().take(dim).copied());
    new_dims.push(1);
    new_dims.extend(current_dims.iter().skip(dim).copied());

    // Use reshape to preserve autograd if enabled
    let new_dims_isize: Vec<isize> = new_dims.iter().map(|&d| d as isize).collect();
    crate::ops::shape::reshape::reshape(tensor, &new_dims_isize)
}

impl<B, T> Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T> + Default + Clone + 'static,
    T: DataType + Clone + std::ops::Neg<Output = T> + 'static,
{
    /// Unsqueezes the tensor by inserting a dimension of size 1 at the specified position.
    pub fn unsqueeze(&self, dim: usize) -> crate::Result<Self> {
        unsqueeze(self, dim)
    }
}
