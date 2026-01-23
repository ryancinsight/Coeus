//! Max backward function

use alloc::sync::Arc;
use alloc::vec::Vec;
use backend::Backend;
use dtype::traits::FloatExt;
use dtype::DataType;
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use crate::{DifferentiableFunction, Function, Tensor};

/// Max function for reduction operations
#[derive(Debug)]
pub struct MaxFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
    pub mask: Arc<Tensor<B, S, T>>,
    pub dim_val: usize,
    pub keepdim: bool,
}

impl<B, S, T> MaxFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    #[must_use]
    pub fn new(
        input: Arc<Tensor<B, S, T>>,
        mask: Arc<Tensor<B, S, T>>,
        dim: usize,
        keepdim: bool,
    ) -> Self {
        Self {
            inputs: vec![input],
            mask,
            dim_val: dim,
            keepdim,
        }
    }
}

impl<B, S, T> Function<B, S, T> for MaxFunction<B, S, T>
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
        let mut grad_output_dense = grad_output.to_dense_generic()?;

        if !self.keepdim {
            let mut new_shape = grad_output_dense.shape().dims().to_vec();
            new_shape.insert(self.dim_val, 1);

            let data = grad_output_dense.as_slice().to_vec();
            grad_output_dense = Tensor::from_vec_with_backend(
                data,
                &new_shape,
                grad_output.backend().clone(),
            )?;
        }

        let mask_dense = self.mask.to_dense_generic()?;
        
        // broadcast grad_output to mask shape if needed (for dim reduction)
        // Since we already inserted the dim, we can just multiply if they match
        let grad_input_dense = &grad_output_dense * &mask_dense;

        let input = &*self.inputs[0];
        let grad_input = Tensor::from_vec_with_backend(
            grad_input_dense.as_slice().to_vec(),
            grad_input_dense.shape().dims(),
            input.backend().clone(),
        )?;

        Ok(vec![grad_input])
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for MaxFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "MaxBackward"
    }
}

impl<B, S, T> crate::AsAny for MaxFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}
