//! MatMul backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

/// MatMul function for linalg operations
#[derive(Debug)]
pub struct MatMulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> MatMulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(lhs: Arc<Tensor<B, S, T>>, rhs: Arc<Tensor<B, S, T>>) -> Self {
        Self { inputs: vec![lhs, rhs] }
    }
}

impl<B, S, T> Function<B, S, T> for MatMulFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + Clone + Copy + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
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

        // d/dlhs (lhs * rhs) = grad_output * rhs.T
        // d/drhs (lhs * rhs) = lhs.T * grad_output
        let rhs_t = crate::ops::transpose(rhs, rhs.shape().dims().len() - 2, rhs.shape().dims().len() - 1)?;
        let lhs_t = crate::ops::transpose(lhs, lhs.shape().dims().len() - 2, lhs.shape().dims().len() - 1)?;

        let val_rhs_t_dense = rhs_t.to_dense_generic()?;
        let val_lhs_t_dense = lhs_t.to_dense_generic()?;

        let grad_lhs_out = crate::ops::linalg::matmul(grad_output, &val_rhs_t_dense)?;
        let grad_rhs_out = crate::ops::linalg::matmul(&val_lhs_t_dense, grad_output)?;

        let grad_lhs_dense = grad_lhs_out.to_dense_generic()?;
        let grad_rhs_dense = grad_rhs_out.to_dense_generic()?;

        let grad_lhs = Tensor::from_vec_with_backend(
            grad_lhs_dense.as_slice().to_vec(),
            grad_lhs_dense.shape().dims(),
            lhs.backend().clone(),
        )?;
        let grad_rhs = Tensor::from_vec_with_backend(
            grad_rhs_dense.as_slice().to_vec(),
            grad_rhs_dense.shape().dims(),
            rhs.backend().clone(),
        )?;

        Ok(vec![grad_lhs, grad_rhs])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for MatMulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "MatMulBackward"
    }
}

impl<B, S, T> crate::AsAny for MatMulFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
