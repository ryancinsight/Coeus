//! Standard Dropout layer.

use backend::CpuBackend;
use dtype::{traits::FloatExt, DataType};
use storage::DenseStorage;
use tensor::Tensor;

use crate::core::error::Result;
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// Dropout layer for regularization.
#[derive(Debug, Clone)]
pub struct Dropout {
    pub p: f64,
    training: bool,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        if !(0.0..=1.0).contains(&p) {
            panic!("dropout probability must be between 0 and 1");
        }
        Self { p, training: true }
    }

    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl<T: DataType + FloatExt> Module<CpuBackend<T>, DenseStorage<T>, T> for Dropout {
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }
        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let keep_prob = 1.0 - self.p;

        let output_data: Vec<T> = input
            .as_slice()
            .iter()
            .map(|&x| {
                if rand::random::<f64>() < keep_prob {
                    x * scale
                } else {
                    T::zero()
                }
            })
            .collect();

        Ok(Tensor::from_vec(output_data, input.shape().dims())?)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        Vec::new()
    }
    fn zero_grad(&mut self) {}
    fn train(&mut self, mode: bool) {
        self.training = mode;
    }
    fn name(&self) -> &str {
        "Dropout"
    }
    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T>> {
        Box::new(self.clone())
    }
}

impl std::fmt::Display for Dropout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dropout(p={})", self.p)
    }
}
