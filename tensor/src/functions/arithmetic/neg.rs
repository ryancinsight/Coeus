//! Neg backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};
use crate::functions::utils::{unbroadcast_dense, to_storage_preserving_graph};

/// Neg function for element-wise negation
#[derive(Debug)]
pub struct NegFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> NegFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![input],
        }
    }
}

impl<B, S, T> Function<B, S, T> for NegFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + Clone + Copy + num_traits::Zero + std::ops::Neg<Output = T>,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let input = &*self.inputs[0];

        // d/dx (-x) = -1
        let neg_grad_data: Vec<T> = grad_output.as_slice().iter().map(|&g| T::zero() - g).collect();
        let neg_grad = Tensor::from_vec_with_backend(neg_grad_data, grad_output.shape().dims(), grad_output.backend.clone())?;
        let grad_dense = unbroadcast_dense(&neg_grad, input.shape().dims())?;

        let grad = to_storage_preserving_graph(grad_dense)?;

        Ok(vec![grad])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for NegFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

impl<B, S, T> crate::AsAny for NegFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
