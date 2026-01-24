//! Acos backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use num_traits::FromPrimitive;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

/// Acos function for automatic differentiation
#[derive(Debug)]
pub struct AcosFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> AcosFunction<B, S, T>
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

impl<B, S, T> Function<B, S, T> for AcosFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::One + num_traits::Float + num_traits::FromPrimitive + 'static,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // d/dx acos(x) = -1 / sqrt(1 - x^2)
        let input = &*self.inputs[0];
        let input_dense = input.to_dense_generic()?;
        
        let one = Tensor::ones_like(&input_dense)?;
        let neg_one = crate::ops::arithmetic::neg(&one)?;
        let x_sq = crate::ops::math::pow_scalar(&input_dense, T::from_f64(2.0).unwrap())?;
        let one_minus_x_sq = crate::ops::arithmetic::sub(&one, &x_sq)?;
        let sqrt_term = crate::ops::sqrt(&one_minus_x_sq)?;
        let grad = crate::ops::arithmetic::div(&neg_one, &sqrt_term)?; 
        
        let grad_input_dense = grad_output * &grad;

        let grad_input = Tensor::from_vec_with_backend(
            grad_input_dense.as_slice().to_vec(),
            grad_input_dense.shape().dims(),
            input.backend().clone(),
        )?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for AcosFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "AcosBackward"
    }
}

impl<B, S, T> crate::AsAny for AcosFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
