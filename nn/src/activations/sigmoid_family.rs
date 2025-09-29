//! Sigmoid-family activation functions
//!
//! This module contains activation functions based on the sigmoid function.

use crate::Module;
use coeus_tensor::{FloatDtype, Tensor, CpuBackend};
use std::fmt;

/// Sigmoid activation function
///
/// Formula: `σ(x) = 1 / (1 + exp(-x))`
///
/// The sigmoid function squashes the input to the range (0, 1).
/// It's commonly used in binary classification problems.
#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Sigmoid {
    /// Create a new Sigmoid activation
    pub fn new() -> Self {
        Sigmoid
    }
}

impl<T: FloatDtype> Module<T> for Sigmoid {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Ok(input.sigmoid()?)
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

impl fmt::Display for Sigmoid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Sigmoid()")
    }
}

/// LogSigmoid activation function
///
/// Formula: `LogSigmoid(x) = log(σ(x)) = log(1 / (1 + exp(-x)))`
///
/// LogSigmoid is numerically stable and is used in some loss functions.
#[derive(Debug, Clone, Copy)]
pub struct LogSigmoid;

impl Default for LogSigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl LogSigmoid {
    /// Create a new LogSigmoid activation
    pub fn new() -> Self {
        LogSigmoid
    }
}

impl<T: FloatDtype> Module<T> for LogSigmoid {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Ok(input.logsigmoid()?)
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

impl fmt::Display for LogSigmoid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LogSigmoid()")
    }
}


