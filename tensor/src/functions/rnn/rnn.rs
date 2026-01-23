//! RNN backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

/// RNN function for automatic differentiation
#[derive(Debug)]
pub struct RNNFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub h0: Option<Arc<Tensor<B, S, T>>>,
    pub batch_first: bool,
}

impl<B, S, T> RNNFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(
        input: Arc<Tensor<B, S, T>>,
        h0: Option<Arc<Tensor<B, S, T>>>,
        batch_first: bool,
    ) -> Self {
        Self {
            inputs: vec![input],
            h0,
            batch_first,
        }
    }
}

impl<B, S, T> Function<B, S, T> for RNNFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + Clone + Copy,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        _grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // Simplified backward pass for RNN
        // In a real implementation, this would involve BPTT
        let mut result = Vec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            let grad = Tensor::from_vec_with_backend(
                vec![T::zero(); input.as_slice().len()],
                input.shape().dims(),
                input.backend().clone(),
            )?;
            result.push(grad);
        }
        Ok(result)
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for RNNFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "RNNBackward"
    }
}

impl<B, S, T> crate::AsAny for RNNFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
