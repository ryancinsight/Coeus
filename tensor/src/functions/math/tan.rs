//! Tan backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

/// Tan function for automatic differentiation
#[derive(Debug)]
pub struct TanFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub output: Arc<Tensor<B, S, T>>, // Save output for efficient gradient: 1 + y^2
}

impl<B, S, T> TanFunction<B, S, T>
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

impl<B, S, T> Function<B, S, T> for TanFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::One + num_traits::Float + num_traits::FromPrimitive + 'static,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // d/dx tan(x) = 1 + tan^2(x) = 1 + y^2
        // We use saved output y = tan(x)
        let y = &self.output;
        let y_dense = y.to_dense_generic()?;
        
        let y_sq = crate::ops::math::pow_scalar(&y_dense, T::from_f64(2.0).unwrap())?; // or y * y
        
        // 1 + y^2
        let one = Tensor::ones_like(&y_dense)?;
        let grad_scale = crate::ops::arithmetic::add(&one, &y_sq)?;
        
        let grad_input_dense = grad_output * &grad_scale;

        let grad_input = Tensor::from_vec_with_backend(
            grad_input_dense.as_slice().to_vec(),
            grad_input_dense.shape().dims(),
            y.backend().clone(),
        )?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for TanFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "TanBackward"
    }
}

impl<B, S, T> crate::AsAny for TanFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
