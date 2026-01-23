//! Add backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};
use crate::functions::utils::{unbroadcast_dense, to_storage_preserving_graph};

/// Add function for element-wise addition
#[derive(Debug)]
pub struct AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [lhs, rhs]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new Add function
    #[must_use]
    pub fn new(lhs: Arc<Tensor<B, S, T>>, rhs: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![lhs, rhs],
        }
    }
}

impl<B, S, T> Function<B, S, T> for AddFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + Clone + Copy + num_traits::Zero + std::ops::Add<Output = T>,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let lhs = &*self.inputs[0];
        let rhs = &*self.inputs[1];

        let left_grad_dense = unbroadcast_dense(grad_output, lhs.shape().dims())?;
        let right_grad_dense = unbroadcast_dense(grad_output, rhs.shape().dims())?;

        let grad_lhs = to_storage_preserving_graph(left_grad_dense)?;
        let grad_rhs = to_storage_preserving_graph(right_grad_dense)?;

        Ok(vec![grad_lhs, grad_rhs])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

impl<B, S, T> crate::AsAny for AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
