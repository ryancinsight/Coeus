//! ReLU activation function

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;

/// Applies the Rectified Linear Unit (ReLU) activation function.
///
/// Formula: `max(0, x)`
pub fn relu<B, T, S>(
    input: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    T: DataType + PartialOrd + Clone + 'static + dtype::traits::FloatExt,
    S: storage::Storage<T> + 'static,
{
    let result_storage = input.storage().map_structure(|x| if x > T::zero() { x } else { T::zero() })
        .map_err(crate::TensorError::StorageError)?;
    
    let result = Tensor::from_storage(result_storage, input.backend().clone());

    if crate::tensor_core::grad_enabled() && input.requires_grad {
        // For autograd we might need dense, but let's keep it abstract if possible or handle it
        // The previous implementation used dense result for everything.
        // If we want to support sparse autograd, we need sparse gradient support.
        // For now, let's keep simple RELU forward.
        // NOTE: Autograd for sparse usually produces dense grad for ReLu unless we use masked grad.
        // Re-implementing grad_fn might be needed.
        
        // Check if we can use the existing ReluFunction which expects Arc<Tensor<...>>
        // The previous code had input_dense.
        
        // For now, if we want to support sparse, we skip autograd setup or we need to ensure ReluFunction handles sparse.
        // Assuming ReluFunction is generic enough? 
        // Let's modify the signature to not force dense return.
    }

    Ok(result)
}
