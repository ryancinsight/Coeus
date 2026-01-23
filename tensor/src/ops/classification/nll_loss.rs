//! Negative log likelihood loss operation

use crate::{Result, Tensor};
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use num_traits::{FromPrimitive, ToPrimitive};
use std::sync::Arc;
use storage::{Storage, StorageFromVec, StorageToDense};
use crate::functions::NLLLossFunction;

/// Computes negative log likelihood (NLL) loss.
pub fn nll_loss<B, S, T>(
    input: &Tensor<B, S, T>,
    target: &Tensor<B, S, T>,
) -> Result<Tensor<B, S, T>>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T> + Clone + FromPrimitive + ToPrimitive + Send + Sync + 'static,
    S: Storage<T> + StorageToDense<T> + StorageFromVec<T> + Clone + Send + Sync + 'static,
{
    let input_shape = input.shape().dims();
    let target_shape = target.shape().dims();
    let num_classes = *input_shape.last().unwrap();
    
    let is_indices = target_shape.len() + 1 == input_shape.len()
        && target_shape == &input_shape[..input_shape.len() - 1];

    let input_dense = input.to_dense_generic()?;
    let target_dense = target.to_dense_generic()?;
    let input_data = input_dense.as_slice();
    let target_data = target_dense.as_slice();
    
    let batch_elems = input_data.len() / num_classes;
    let mut total_loss = T::zero();

    if is_indices {
        for (b, &target_val) in target_data.iter().enumerate() {
            let target_idx = target_val.to_usize().unwrap();
            let log_prob = input_data[b * num_classes + target_idx];
            total_loss = total_loss - log_prob;
        }
    } else {
        for (&lp, &t) in input_data.iter().zip(target_data.iter()) {
            total_loss = total_loss - (lp * t);
        }
    }

    let mean_loss = total_loss / T::from_f64(batch_elems as f64).unwrap();
    let mut result = Tensor::from_vec_with_backend(vec![mean_loss], &[1], input.backend().clone())?;

    if crate::tensor_core::grad_enabled() && (input.requires_grad() || target.requires_grad()) {
        let grad_fn = NLLLossFunction::new(Arc::new(input.clone()), Arc::new(target.clone()));
        result = result
            .requires_grad_(true)
            .with_grad_fn(Some(Arc::new(grad_fn)));
    }

    Ok(result)
}

