//! Softmax backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

/// Softmax function for automatic differentiation
#[derive(Debug)]
pub struct SoftmaxFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub output: Arc<Tensor<B, S, T>>,
    pub dim: usize,
}

impl<B, S, T> SoftmaxFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(input: Arc<Tensor<B, S, T>>, output: Arc<Tensor<B, S, T>>, dim: usize) -> Self {
        Self {
            inputs: vec![input],
            output,
            dim,
        }
    }
}

impl<B, S, T> Function<B, S, T> for SoftmaxFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + crate::ops::TensorStorageOps<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(
        &self,
        grad_output: &Tensor<B, DenseStorage<T>, T>,
    ) -> anyhow::Result<Vec<Tensor<B, S, T>>> {
        let output_dense = self.output.to_dense_generic()?;
        
        let mut result_data = vec![T::zero(); output_dense.as_slice().len()];
        let out_slice = output_dense.as_slice();
        let grad_slice = grad_output.as_slice();
        
        // Softmax gradient along dim:
        // grad_input[i] = y[i] * (grad_output[i] - sum(grad_output[j] * y[j]))
        
        let dims = output_dense.shape().dims();
        let stride: usize = dims.iter().skip(self.dim + 1).product();
        let stride = if stride == 0 { 1 } else { stride };
        let outer_size: usize = dims.iter().take(self.dim).product();
        let outer_size = if outer_size == 0 { 1 } else { outer_size };
        let dim_size = dims[self.dim];

        for outer in 0..outer_size {
            for inner in 0..stride {
                let base = (outer * dim_size) * stride + inner;
                
                // Calculate sum(grad_output[j] * y[j]) for this slice
                let mut weighted_sum = T::zero();
                for i in 0..dim_size {
                    let idx = base + i * stride;
                    weighted_sum = weighted_sum + (grad_slice[idx] * out_slice[idx]);
                }
                
                // Calculate grad_input[i]
                for i in 0..dim_size {
                    let idx = base + i * stride;
                    result_data[idx] = out_slice[idx] * (grad_slice[idx] - weighted_sum);
                }
            }
        }

        let input = &*self.inputs[0];
        let grad_input = Tensor::from_vec_with_backend(
            result_data,
            output_dense.shape().dims(),
            input.backend().clone(),
        )?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for SoftmaxFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "SoftmaxBackward"
    }
}

impl<B, S, T> crate::AsAny for SoftmaxFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
