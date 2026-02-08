//! Spatial 2D Dropout layer.

use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::{Tensor, ops::TensorStorageOps};

use crate::core::error::Result;
use crate::{Module, Parameter};
use std::ops::Mul;

/// Dropout2d layer for spatial regularization in CNNs.
#[derive(Debug, Clone)]
pub struct Dropout2d {
    pub p: f64,
    pub training: bool,
}

impl Dropout2d {
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "Dropout probability must be in [0.0, 1.0], got {}",
            p
        );
        Self { p, training: true }
    }

    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }
}

impl<B, S, T> Module<B, S, T> for Dropout2d
where
    B: Backend<Data = T> + Clone + Default,
    S: storage::Storage<T> + storage::StorageFromVec<T> + storage::StorageToDense<T> + TensorStorageOps<T> + Clone + 'static,
    T: DataType + FloatExt + Clone,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(
        &self,
        input: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let input_shape = input.shape().dims();
        if input_shape.len() != 4 {
             return Err(crate::core::error::NNError::InvalidInput {
                message: format!("Dropout2d expects 4D input [N, C, H, W], got {}D", input_shape.len()),
            });
        }

        let [batch_size, channels, _, _] = [input_shape[0], input_shape[1], input_shape[2], input_shape[3]];
        
        // Generate mask of shape [N, C, 1, 1]
        let mut rng = rand::thread_rng();
        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let p_thresh = self.p;
        
        let mut mask_data = Vec::with_capacity(batch_size * channels);
        for _ in 0..(batch_size * channels) {
            let val = if rand::Rng::gen::<f64>(&mut rng) > p_thresh {
                scale
            } else {
                T::zero()
            };
            mask_data.push(val);
        }

        let mask = Tensor::<B, storage::DenseStorage<T>, T>::from_vec_with_backend(
            mask_data,
            &[batch_size, channels, 1, 1],
            input.backend().clone(),
        )?;

        // Expand mask and multiply
        // Expand mask implicitly via broadcasting in mul
        let output = tensor::ops::mul(input, &mask)?;
        
        let dense = output.to_dense_generic()?;
        let storage = S::from_vec(dense.as_slice().to_vec(), dense.shape().dims())?;
        Ok(Tensor::from_storage(storage, input.backend().clone()))
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![]
    }
    fn zero_grad(&mut self) {}
    fn train(&mut self, mode: bool) {
        self.training = mode;
    }
    fn name(&self) -> &str {
        "Dropout2d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}

impl std::fmt::Display for Dropout2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dropout2d(p={})", self.p)
    }
}
