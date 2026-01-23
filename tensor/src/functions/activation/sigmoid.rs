//! Sigmoid backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

/// Sigmoid activation function for automatic differentiation
#[derive(Debug)]
pub struct SigmoidFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub output: Arc<Tensor<B, S, T>>,
}

impl<B, S, T> SigmoidFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>, output: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![input],
            output,
        }
    }
}

impl<B, S, T> Function<B, S, T> for SigmoidFunction<B, S, T>
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
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        let output_dense = self.output.to_dense_generic()?;
        
        let mut result_data = vec![T::zero(); output_dense.as_slice().len()];
        let out_slice = output_dense.as_slice();
        let grad_slice = grad_output.as_slice();

        for i in 0..result_data.len() {
            let s = out_slice[i];
            let derivative = s * (T::one() - s);
            result_data[i] = grad_slice[i] * derivative;
        }

        let input = &*self.inputs[0];
        let grad_input = Tensor::from_vec_with_backend(result_data, output_dense.shape().dims(), input.backend().clone())?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for SigmoidFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "SigmoidBackward"
    }
}

impl<B, S, T> crate::AsAny for SigmoidFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
