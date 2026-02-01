use crate::Tensor;
use backend::{Backend, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};

/// Squeezes the tensor by removing a dimension of size 1 at the specified position.
pub fn squeeze<B, S, T>(
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
    if dim >= ndim {
        return Err(crate::TensorError::ShapeError {
            expected: ndim,
            actual: dim,
            message: format!("squeeze: dimension {dim} out of bounds for ndim {ndim}"),
        });
    }

    if current_dims[dim] != 1 {
        return Err(crate::TensorError::ShapeError {
            expected: 1,
            actual: current_dims[dim],
            message: format!(
                "squeeze: dimension {dim} must have size 1, got {}",
                current_dims[dim]
            ),
        });
    }

    let mut new_dims = Vec::with_capacity(ndim - 1);
    new_dims.extend(
        current_dims
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| (i != dim).then_some(d)),
    );

    // Use reshape to preserve autograd if enabled
    let new_dims_isize: Vec<isize> = new_dims.iter().map(|&d| d as isize).collect();
    crate::ops::shape::reshape::reshape(tensor, &new_dims_isize)
}

impl<B, T> Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T> + Default + Clone + 'static,
    T: DataType + Clone + std::ops::Neg<Output = T> + 'static,
{
    /// Squeezes the tensor by removing a dimension of size 1 at the specified position.
    pub fn squeeze(&self, dim: usize) -> crate::Result<Self> {
        squeeze(self, dim)
    }
}
