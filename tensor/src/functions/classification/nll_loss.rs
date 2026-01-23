//! NLLLoss backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

/// NLLLoss function for automatic differentiation
#[derive(Debug)]
pub struct NLLLossFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub target: Arc<Tensor<B, S, T>>,
}

impl<B, S, T> NLLLossFunction<B, S, T>
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

impl<B, S, T> Function<B, S, T> for NLLLossFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
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
        let target_dense = self.target.to_dense_generic()?;
        
        let mut result_data = vec![T::zero(); input.as_slice().len()];
        let target_slice = target_dense.as_slice();
        let grad_slice = grad_output.as_slice();

        for i in 0..result_data.len() {
            // Negative Log Likelihood gradient: -target
            // Multiplying by grad_output (scalar loss grad)
            result_data[i] = grad_slice[0] * (T::zero() - target_slice[i]);
        }

        let grad_input = Tensor::from_vec_with_backend(
            result_data,
            input.shape().dims(),
            input.backend().clone(),
        )?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for NLLLossFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "NLLLossBackward"
    }
}

impl<B, S, T> crate::AsAny for NLLLossFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
