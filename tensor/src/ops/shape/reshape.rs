use backend::{Backend, DataType};
use storage::{Storage, StorageFromVec, StorageToDense};
use crate::{Tensor, TensorError};
use std::sync::Arc;
use crate::functions::ReshapeFunction;

/// Standalone reshape logic with Autograd integration
pub fn reshape<B, T, S>(
    tensor: &Tensor<B, S, T>,
    dims: &[isize],
) -> crate::Result<Tensor<B, crate::DenseStorage<T>, T>> 
where
    B: Backend<Data = T> + Default + Clone + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + Clone + 'static,
{
     // Convert to dense storage for arbitrary element rearrangement
    let dense_tensor = tensor.to_dense_generic()?;

    // Validate and resolve dimensions
    let resolved_dims =
        Tensor::<B, S, T>::resolve_reshape_dims_generic(dense_tensor.len(), dims)?;

    // Check total element count matches
    let new_size: usize = resolved_dims.iter().product();
    if new_size != dense_tensor.len() {
        return Err(TensorError::ShapeError {
            expected: dense_tensor.len(),
            actual: new_size,
            message: "Total element count mismatch in reshape".to_string(),
        });
    }

    // Create new dense storage with reshaped data
    let data = dense_tensor.as_slice().to_vec();
    let new_storage = crate::DenseStorage::from_vec(data, &resolved_dims)
        .map_err(TensorError::StorageError)?;

    let mut result = Tensor::from_storage(
        new_storage,
        dense_tensor.backend.clone(),
    );
    
    // Autograd connection
    if crate::tensor_core::grad_enabled() && tensor.requires_grad() {
         // We must use dense_tensor here because ReshapeFunction must match Output Storage (Dense)
         // if we want to attach it to result.
         // If dense_tensor is a clone of tensor (S=Dense), history is preserved.
         // If dense_tensor is new (S!=Dense), it's a leaf, so graph starts here.
         let grad_fn = ReshapeFunction::new(Arc::new(dense_tensor.clone()), tensor.shape().dims().to_vec());
         result = result
             .requires_grad_(true)
             .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}


impl<B, S, T> Tensor<B, S, T>
where
    B: Backend<Data = T> + Default + Clone + 'static,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + Clone + 'static,
{
    /// Reshapes the tensor to new dimensions.
    ///
    /// This method converts the tensor to dense storage if needed, then reshapes it.
    /// The total number of elements must remain the same.
    ///
    /// # Arguments
    /// * `dims` - New dimensions. Use -1 to infer one dimension.
    ///
    /// # Returns
    /// A new tensor with the reshaped dimensions (always dense storage).
    ///
    /// # Errors
    /// Returns error if total element count doesn't match or conversion fails.
    pub fn reshape(
        &self,
        dims: &[isize],
    ) -> crate::Result<Tensor<B, crate::DenseStorage<T>, T>> {
        reshape(self, dims)
    }
}
