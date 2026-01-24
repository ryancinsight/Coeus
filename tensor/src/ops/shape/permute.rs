use backend::{Backend, DataType};
use storage::{Storage, StorageFromVec, StorageToDense, DenseStorage};
use crate::Tensor;

/// Permutes dimensions of the tensor.
pub fn permute<B, S, T>(
    tensor: &Tensor<B, S, T>,
    dims: &[usize]
) -> crate::Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Default + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + Clone,
{
    let source_dims = tensor.shape().dims();
    let ndim = source_dims.len();

    if dims.len() != ndim {
        return Err(crate::TensorError::ShapeError {
            expected: ndim,
            actual: dims.len(),
            message: format!(
                "permute: number of dimensions must match, got {} != {}",
                dims.len(),
                ndim
            ),
        });
    }

    // Validate permutation
    let mut seen = vec![false; ndim];
    for &d in dims {
        if d >= ndim {
            return Err(crate::TensorError::ShapeError {
                expected: ndim,
                actual: d,
                message: format!("permute: dimension {d} out of bounds for ndim {ndim}"),
            });
        }
        if seen[d] {
            return Err(crate::TensorError::ShapeError {
                expected: ndim,
                actual: d,
                message: format!("permute: duplicate dimension {d}"),
            });
        }
        seen[d] = true;
    }

    let mut target_dims = vec![0; ndim];
    for i in 0..ndim {
        target_dims[i] = source_dims[dims[i]];
    }
    
    // Convert to dense to access data
    let dense_tensor = tensor.to_dense_generic().map_err(|e| crate::TensorError::BackendError(format!("{:?}", e)))?;

    let mut permuted_data = Vec::with_capacity(dense_tensor.len());
    let mut coords = vec![0; ndim]; // coordinates in the target tensor

    loop {
        // map target coordinates back to source coordinates
        let mut source_coords = vec![0; ndim];
        for i in 0..ndim {
            source_coords[dims[i]] = coords[i];
        }

        // compute linear index in source tensor
        let mut linear_idx = 0;
        let mut stride = 1;
        for i in 0..ndim {
            linear_idx += source_coords[i] * stride;
            stride *= source_dims[i];
        }

        permuted_data.push(dense_tensor.as_slice()[linear_idx].clone());

        // increment target coordinates
        let mut carry = 1;
        for i in (0..ndim).rev() {
            coords[i] += carry;
            if coords[i] < target_dims[i] {
                carry = 0;
                break;
            }
            coords[i] = 0;
        }

        if carry != 0 {
            break;
        }
    }

    let new_storage = DenseStorage::from_vec(permuted_data, &target_dims)
        .map_err(crate::TensorError::StorageError)?;
    Ok(Tensor::from_storage(new_storage, B::default()))
}

impl<B, T> Tensor<B, DenseStorage<T>, T>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + Clone,
{
    /// Permutes dimensions of the tensor.
    pub fn permute(&self, dims: &[usize]) -> crate::Result<Self> {
        permute(self, dims)
    }
}
