//! Mean backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

/// Mean function for reduction operations
#[derive(Debug)]
pub struct MeanFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub input_shape: Vec<usize>,
}

impl<B, S, T> MeanFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>, input_shape: Vec<usize>) -> Self {
        Self {
            inputs: vec![input],
            input_shape,
        }
    }
}

impl<B, S, T> Function<B, S, T> for MeanFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + Clone + Copy + num_traits::Zero + num_traits::FromPrimitive,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // d/dx mean(x) = 1 / N
        let numel: usize = self.input_shape.iter().product();
        let grad_numel: usize = grad_output.shape().dims().iter().product();
        let n = numel / grad_numel;
        let factor = T::one() / T::from_f64(n as f64).unwrap();

        let mut grad_data = crate::ops::arithmetic::broadcast_tensor_data(
            grad_output.as_slice(),
            grad_output.shape().dims(),
            &self.input_shape,
        )
        .map_err(|e| anyhow::anyhow!("Broadcast error: {e:?}"))?;

        for x in &mut grad_data {
            *x = *x * factor;
        }

        let input = &*self.inputs[0];
        let grad_input = Tensor::from_vec_with_backend(
            grad_data,
            &self.input_shape,
            input.backend().clone(),
        )?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for MeanFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

impl<B, S, T> crate::AsAny for MeanFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
