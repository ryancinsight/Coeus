//! Leaky ReLU backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

/// Leaky ReLU activation function for automatic differentiation
#[derive(Debug)]
pub struct LeakyReluFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub negative_slope: T,
}

impl<B, S, T> LeakyReluFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>, negative_slope: T) -> Self {
        Self {
            inputs: vec![input],
            negative_slope,
        }
    }
}

impl<B, S, T> Function<B, S, T> for LeakyReluFunction<B, S, T>
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
        
        let mut result_data = vec![T::zero(); input_dense.as_slice().len()];
        let in_slice = input_dense.as_slice();
        let grad_slice = grad_output.as_slice();

        for i in 0..result_data.len() {
            if in_slice[i] > T::zero() {
                result_data[i] = grad_slice[i];
            } else {
                result_data[i] = grad_slice[i] * self.negative_slope;
            }
        }

        let grad_input = Tensor::from_vec_with_backend(result_data, input_dense.shape().dims(), input.backend().clone())?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for LeakyReluFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "LeakyReluBackward"
    }
}

impl<B, S, T> crate::AsAny for LeakyReluFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
