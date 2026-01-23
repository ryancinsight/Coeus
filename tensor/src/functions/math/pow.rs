//! Pow backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

/// Pow function for automatic differentiation
#[derive(Debug)]
pub struct PowFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub exponent: T,
}

impl<B, S, T> PowFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>, exponent: T) -> Self {
        Self {
            inputs: vec![input],
            exponent,
        }
    }
}

impl<B, S, T> Function<B, S, T> for PowFunction<B, S, T>
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
        // d/dx x^y = y * x^(y-1)
        let input = &*self.inputs[0];
        let input_dense = input.to_dense_generic()?;
        
        let mut result_data = vec![T::zero(); input_dense.as_slice().len()];
        let in_slice = input_dense.as_slice();
        let grad_slice = grad_output.as_slice();
        let exp_minus_one = self.exponent - T::one();

        for i in 0..result_data.len() {
            result_data[i] = grad_slice[i] * self.exponent * in_slice[i].powf(exp_minus_one);
        }

        let grad_input = Tensor::from_vec_with_backend(
            result_data,
            input_dense.shape().dims(),
            input.backend().clone(),
        )?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for PowFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "PowBackward"
    }
}

impl<B, S, T> crate::AsAny for PowFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// Binary Pow function (tensor ^ tensor) for automatic differentiation
#[derive(Debug)]
pub struct PowBinaryFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> PowBinaryFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(base: Arc<Tensor<B, S, T>>, exponent: Arc<Tensor<B, S, T>>) -> Self {
        Self {
            inputs: vec![base, exponent],
        }
    }
}

impl<B, S, T> Function<B, S, T> for PowBinaryFunction<B, S, T>
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
        let base = &*self.inputs[0];
        let exponent = &*self.inputs[1];
        
        let base_dense = base.to_dense_generic()?;
        let exp_dense = exponent.to_dense_generic()?;
        let grad_slice = grad_output.as_slice();
        let base_slice = base_dense.as_slice();
        let exp_slice = exp_dense.as_slice();
        
        let mut grad_base_data = vec![T::zero(); base_slice.len()];
        let mut grad_exp_data = vec![T::zero(); base_slice.len()];

        for i in 0..base_slice.len() {
            let b = base_slice[i];
            let e = exp_slice[i];
            let g = grad_slice[i];
            
            // d/db b^e = e * b^(e-1)
            grad_base_data[i] = g * e * b.powf(e - T::one());
            
            // d/de b^e = b^e * ln(b)
            if b > T::zero() {
                grad_exp_data[i] = g * b.powf(e) * b.ln();
            } else {
                grad_exp_data[i] = T::zero();
            }
        }

        let grad_base = Tensor::from_vec_with_backend(
            grad_base_data,
            base.shape().dims(),
            base.backend().clone(),
        )?;
        
        let grad_exp = Tensor::from_vec_with_backend(
            grad_exp_data,
            exponent.shape().dims(),
            exponent.backend().clone(),
        )?;

        Ok(vec![grad_base, grad_exp])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for PowBinaryFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "PowBinaryBackward"
    }
}

impl<B, S, T> crate::AsAny for PowBinaryFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
