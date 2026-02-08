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
    type Input = Tensor<CpuBackend<T>, DenseStorage<T>, T>;
    type Output = Tensor<CpuBackend<T>, DenseStorage<T>, T>;

    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let mut output_data = input.as_slice().to_vec();
        let scale = 1.0 / (1.0 - self.p);

        for val in output_data.iter_mut() {
            if rand::random::<f64>() < self.p {
                *val = T::zero();
            } else {
                *val = *val * T::from(scale).unwrap();
            }
        }

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

    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}

impl std::fmt::Display for Dropout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dropout(p={})", self.p)
    }
}
