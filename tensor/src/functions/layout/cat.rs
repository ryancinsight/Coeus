//! Cat backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

/// Cat function for layout operations
#[derive(Debug)]
pub struct CatFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub dim: usize,
}

impl<B, S, T> CatFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(inputs: Vec<Arc<Tensor<B, S, T>>>, dim: usize) -> Self {
        Self { inputs, dim }
    }
}

impl<B, S, T> Function<B, S, T> for CatFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        // d/dx Cat([x1, x2, ...], dim) = [Split(grad_output, dim, x1_shape), Split(grad_output, dim, x2_shape), ...]
        let mut result = Vec::with_capacity(self.inputs.len());
        let mut offset = 0;

        for input in &self.inputs {
            let dim_size = input.shape().dims()[self.dim];
            
            // Manual slice extract from grad_output
            // Since it's dense, we can just take the sub-slice
            let mut out_shape = grad_output.shape().dims().to_vec();
            out_shape[self.dim] = dim_size;
            
            let stride: usize = grad_output.shape().dims().iter().skip(self.dim + 1).product();
            let stride = if stride == 0 { 1 } else { stride };
            let outer_size: usize = grad_output.shape().dims().iter().take(self.dim).product();
            let outer_size = if outer_size == 0 { 1 } else { outer_size };
            let grad_dim_size = grad_output.shape().dims()[self.dim];

            let mut grad_input_data = Vec::with_capacity(out_shape.iter().product());
            let grad_slice = grad_output.as_slice();

            for outer in 0..outer_size {
                let base_idx = (outer * grad_dim_size + offset) * stride;
                for i in 0..(dim_size * stride) {
                    grad_input_data.push(grad_slice[base_idx + i]);
                }
            }

            let grad_input = Tensor::from_vec_with_backend(
                grad_input_data,
                &out_shape,
                input.backend().clone(),
            )?;
            result.push(grad_input);
            offset += dim_size;
        }

        Ok(result)
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for CatFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "CatBackward"
    }
}

impl<B, S, T> crate::AsAny for CatFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
