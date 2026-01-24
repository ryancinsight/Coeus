//! Mul backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};
use crate::functions::utils::{unbroadcast_dense, to_storage_preserving_graph};

/// Mul function for element-wise multiplication
#[derive(Debug)]
pub struct MulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> MulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(lhs: Arc<Tensor<B, S, T>>, rhs: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![lhs, rhs],
        }
    }
}


impl<B, S, T> Function<B, S, T> for MulFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + Send + Sync + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + Send + Sync + crate::ops::dispatch::TensorStorageOps<T> + 'static,
    T: DataType + Clone + Copy + num_traits::Zero + std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Div<Output = T> + std::ops::Neg<Output = T>,
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

        // Convert inputs to dense if they are not, to match grad_output shape
        let lhs_dense = lhs.to_dense_generic()?;
        let rhs_dense = rhs.to_dense_generic()?;

        // d/dlhs (lhs * rhs) = grad_output * rhs
        let grad_lhs_out = crate::ops::arithmetic::mul(grad_output, &rhs_dense)?;
        let grad_lhs_dense = unbroadcast_dense(&grad_lhs_out, lhs.shape().dims())?;
        
        // d/drhs (lhs * rhs) = grad_output * lhs
        let grad_rhs_out = crate::ops::arithmetic::mul(grad_output, &lhs_dense)?;
        let grad_rhs_dense = unbroadcast_dense(&grad_rhs_out, rhs.shape().dims())?;

        let grad_lhs = to_storage_preserving_graph(grad_lhs_dense)?;
        let grad_rhs = to_storage_preserving_graph(grad_rhs_dense)?;

        Ok(vec![grad_lhs, grad_rhs])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for MulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

impl<B, S, T> crate::AsAny for MulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
