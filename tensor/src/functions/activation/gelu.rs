//! GELU backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

/// GELU activation function for automatic differentiation
#[derive(Debug)]
pub struct GeluFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> GeluFunction<B, S, T>
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

impl<B, S, T> Function<B, S, T> for GeluFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::FromPrimitive,
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

        // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
        // d/dx GELU(x) ≈ 0.5 * (1 + erf(x/sqrt(2))) + (x / sqrt(2*PI)) * exp(-x²/2)
        let _sqrt2 = T::from_f64(std::f64::consts::SQRT_2).unwrap();
        let inv_sqrt2pi = T::from_f64(1.0 / (2.0 * std::f64::consts::PI).sqrt()).unwrap();

        for i in 0..result_data.len() {
            let x = in_slice[i];
            let x_f64 = num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0);
            let erf_val = statrs::function::erf::erf(x_f64 / std::f64::consts::SQRT_2);
            let erf_t = T::from_f64(erf_val).unwrap_or(T::zero());
            
            let term1 = T::from_f64(0.5).unwrap() * (T::one() + erf_t);
            let term2 = (x * inv_sqrt2pi) * (T::zero() - (x * x / T::from_f64(2.0).unwrap())).exp();
            
            let derivative = term1 + term2;
            result_data[i] = grad_slice[i] * derivative;
        }

        let grad_input = Tensor::from_vec_with_backend(result_data, input_dense.shape().dims(), input.backend().clone())?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for GeluFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "GeluBackward"
    }
}

impl<B, S, T> crate::AsAny for GeluFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
