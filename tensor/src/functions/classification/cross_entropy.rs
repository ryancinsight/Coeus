//! CrossEntropy backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

/// CrossEntropy function for automatic differentiation
#[derive(Debug)]
pub struct CrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub target: Arc<Tensor<B, S, T>>,
}

impl<B, S, T> CrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>, target: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![input],
            target,
        }
    }
}

impl<B, S, T> Function<B, S, T> for CrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let input = &*self.inputs[0];
        let input_dense = input.to_dense_generic()?;
        let target_dense = self.target.to_dense_generic()?;
        
        // Simplified CrossEntropy gradient assuming logarithmic inputs (log_softmax)
        // d/dx CE(log_softmax(x), target) = softmax(x) - target
        // For coeus, we assume the loss is computed on probabilities if targets are probabilities
        
        let mut result_data = vec![T::zero(); input_dense.as_slice().len()];
        let in_slice = input_dense.as_slice();
        let target_slice = target_dense.as_slice();
        let grad_slice = grad_output.as_slice();

        for i in 0..result_data.len() {
            // grad_output is typically a scalar for loss
            result_data[i] = grad_slice[0] * (in_slice[i] - target_slice[i]);
        }

        let grad_input = Tensor::from_vec_with_backend(
            result_data,
            input_dense.shape().dims(),
            input.backend().clone(),
        )?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for CrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "CrossEntropyBackward"
    }
}

impl<B, S, T> crate::AsAny for CrossEntropyFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
