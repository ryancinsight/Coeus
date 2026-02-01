use crate::Tensor;
use backend::{Backend, DataType};
use storage::{Storage, StorageFromVec, StorageToDense};

/// Flattens the tensor by contiguous dimensions between `start_dim` and `end_dim`.
pub fn flatten<B, S, T>(
    tensor: &Tensor<B, S, T>,
    start_dim: usize,
    end_dim: isize,
) -> crate::Result<Tensor<B, crate::DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Default + Clone + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
    T: DataType + Clone + 'static,
{
    let current_dims = tensor.shape().dims();
    let ndim = current_dims.len();

    let end_dim_idx = if end_dim < 0 {
        let idx = (ndim as isize + end_dim) as usize;
        if idx >= ndim {
            return Err(crate::TensorError::ShapeError {
                expected: ndim,
                actual: idx,
                message: format!("flatten: end_dim {end_dim} out of bounds for ndim {ndim}"),
            });
        }
        idx
    } else {
        let idx = end_dim as usize;
        if idx >= ndim {
            return Err(crate::TensorError::ShapeError {
                expected: ndim,
                actual: idx,
                message: format!("flatten: end_dim {end_dim} out of bounds for ndim {ndim}"),
            });
        }
        idx
    };

    if start_dim >= ndim {
        return Err(crate::TensorError::ShapeError {
            expected: ndim,
            actual: start_dim,
            message: format!("flatten: start_dim {start_dim} out of bounds for ndim {ndim}"),
        });
    }

    if start_dim > end_dim_idx {
        return Err(crate::TensorError::ShapeError {
            expected: 0,
            actual: 0,
            message: format!("flatten: start_dim {start_dim} must be <= end_dim_idx {end_dim_idx}"),
        });
    }

    let mut new_dims = Vec::new();
    new_dims.extend(current_dims.iter().take(start_dim).map(|&d| d as isize));

    let flattened_dim: usize = current_dims[start_dim..=end_dim_idx].iter().product();
    new_dims.push(flattened_dim as isize);

    new_dims.extend(
        current_dims
            .iter()
            .take(ndim)
            .skip(end_dim_idx + 1)
            .map(|&d| d as isize),
    );

    crate::ops::shape::reshape::reshape(tensor, &new_dims)
}

impl<B, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T> + Default + Clone + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + 'static,
    T: DataType + Clone + 'static,
{
    /// Flattens the tensor by contiguous dimensions between `start_dim` and `end_dim`.
    pub fn flatten(
        &self,
        start_dim: usize,
        end_dim: isize,
    ) -> crate::Result<Tensor<B, crate::DenseStorage<T>, T>> {
        flatten(self, start_dim, end_dim)
    }
}
