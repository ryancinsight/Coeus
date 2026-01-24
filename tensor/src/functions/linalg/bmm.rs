//! Batch Matrix Multiplication autograd function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

#[derive(Debug)]
pub struct BMMFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> BMMFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(lhs: Arc<Tensor<B, S, T>>, rhs: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![lhs, rhs],
        }
    }
}

impl<B, S, T> Function<B, S, T> for BMMFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static + crate::ops::dispatch::TensorStorageOps<T>,
    T: DataType + Clone + Copy + num_traits::Zero + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + 'static,
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
        
        // Convert to generic storage for ops
        // If S is DenseStorage, this is cheap.
        // We need to use bmm which is in ops/linalg/bmm.rs (circular dependency?)
        // Better to use crate::ops::linalg::bmm
        
        // grad_output is Dense.
        // We need grad_output compatible with S.
        // If S != Dense, we convert grad_output.
        let grad_output_s = Tensor::<B, S, T>::from_vec_with_backend(
            grad_output.as_slice().to_vec(),
            grad_output.shape().dims(),
            grad_output.backend().clone()
        )?;


        // Grad w.r.t LHS: grad * rhs^T
        // rhs shape: [B, N, P]
        // rhs^T shape: [B, P, N] (transpose last two dims)
        
        let rhs_t = crate::ops::shape::transpose::transpose(rhs, 1, 2)?;
        let grad_lhs = crate::ops::linalg::bmm(&grad_output_s, &rhs_t)?;
        
        let lhs_t = crate::ops::shape::transpose::transpose(lhs, 1, 2)?;
        let grad_rhs = crate::ops::linalg::bmm(&lhs_t, &grad_output_s)?;
        
        Ok(vec![grad_lhs, grad_rhs])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for BMMFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "BMMBackward"
    }
}

impl<B, S, T> crate::AsAny for BMMFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
