//! Cross entropy loss operation

use crate::{Result, Tensor};
use backend::Backend;
use dtype::DataType;

use std::sync::Arc;
use storage::{Storage, StorageFromVec, StorageToDense};
use crate::functions::CrossEntropyFunction;
use super::nll_loss;
use crate::ops::classification::softmax;

/// Computes cross-entropy loss for classification tasks.
pub fn cross_entropy<B, S, T>(
    input: &Tensor<B, S, T>,
    target: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    T: DataType + 'static + dtype::traits::FloatExt + num_traits::FromPrimitive + core::ops::Add<Output = T> + std::ops::Neg<Output = T> + PartialOrd + Clone + Send + Sync + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
{
    let input_shape = input.shape().dims();
    let target_shape = target.shape().dims();

    let is_indices = target_shape.len() + 1 == input_shape.len()
        && target_shape == &input_shape[..input_shape.len() - 1];

    if is_indices {
        // CE = nll_loss(log_softmax(input), target)
        // For efficiency, we can use the combined log_softmax + nll_loss logic
        // But for this refactor, we'll keep it simple
        let log_probs = softmax(input, -1)?.as_slice().iter().map(|&x| x.ln()).collect::<Vec<_>>();
        let log_probs_tensor = Tensor::from_vec_with_backend(log_probs, input_shape, input.backend().clone())?;
        
        let mut result = nll_loss(&log_probs_tensor, target)?;
        
        if crate::tensor_core::grad_enabled() && (input.requires_grad() || target.requires_grad()) {
            let grad_fn = CrossEntropyFunction::new(Arc::new(input.clone()), Arc::new(target.clone()));
            result = result
                .requires_grad_(true)
                .with_grad_fn(Some(Arc::new(grad_fn)));
        }
        
        Ok(result)
    } else {
        // CE = -mean(sum(target * log_softmax(input)))
        let log_softmax = softmax(input, -1)?.as_slice().iter().map(|&x| x.ln()).collect::<Vec<_>>();
        let log_softmax_tensor: Tensor<B, S, T> = Tensor::from_vec_with_backend(log_softmax, input_shape, input.backend().clone())?;
        
        let target_dense = target.to_dense_generic()?;
        let target_data = target_dense.as_slice();
        
        let mut total_loss = T::zero();
        for (ls, t) in log_softmax_tensor.as_slice().iter().zip(target_data.iter()) {
            total_loss = total_loss - (*t * *ls);
        }
        
        let batch_size = input.len() / *input_shape.last().unwrap();
        let mean_loss = total_loss / T::from_f64(batch_size as f64).unwrap();
        
        let mut result = Tensor::from_vec_with_backend(vec![mean_loss], &[1], input.backend().clone())?;
        
        if crate::tensor_core::grad_enabled() && (input.requires_grad() || target.requires_grad()) {
            let grad_fn = CrossEntropyFunction::new(Arc::new(input.clone()), Arc::new(target.clone()));
            result = result
                .requires_grad_(true)
                .with_grad_fn(Some(Arc::new(grad_fn)));
        }
        
        Ok(result)
    }
}

