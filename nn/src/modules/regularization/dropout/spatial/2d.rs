//! Spatial 2D Dropout layer.

use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::Tensor;

use crate::core::error::Result;
use crate::core::module::Module;
use crate::core::parameter::Parameter;

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

impl<T: DataType + FloatExt> Module<CpuBackend<T>, DenseStorage<T>, T> for Dropout2d {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();
        assert_eq!(input_shape.len(), 4, "Input must be 4D [N, C, H, W]");

        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        if self.p == 1.0 {
            return Ok(Tensor::zeros(input_shape)?);
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];
        let spatial_size = height * width;

        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let keep_prob = 1.0 - self.p;

        let input_data = input.as_slice();
        let mut output_data = Vec::with_capacity(input_data.len());

        for n in 0..batch_size {
            for c in 0..channels {
                let keep_channel = rand::random::<f64>() < keep_prob;
                for _spatial in 0..spatial_size {
                    let idx = ((n * channels + c) * spatial_size) + _spatial;
                    output_data.push(if keep_channel {
                        input_data[idx] * scale
                    } else {
                        T::zero()
                    });
                }
            }
        }

        Ok(Tensor::from_vec(output_data, input_shape)?)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        vec![]
    }
    fn zero_grad(&mut self) {}
    fn train(&mut self, mode: bool) {
        self.training = mode;
    }
    fn name(&self) -> &str {
        "Dropout2d"
    }
    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T>> {
        Box::new(self.clone())
    }
}

impl std::fmt::Display for Dropout2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dropout2d(p={})", self.p)
    }
}
