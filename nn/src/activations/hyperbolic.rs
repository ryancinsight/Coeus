//! Hyperbolic activation functions
//!
//! This module contains activation functions based on hyperbolic functions.

use crate::Module;
use coeus_tensor::{FloatDtype, Tensor, CpuBackend};
use std::fmt;

/// Tanh (Hyperbolic Tangent) activation function
///
/// Formula: `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
///
/// Tanh squashes the input to the range (-1, 1) and is zero-centered,
/// making it preferable to sigmoid in many cases.
#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Tanh {
    /// Create a new Tanh activation
    pub fn new() -> Self {
        Tanh
    }
}

impl<T: FloatDtype> Module<T> for Tanh {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Ok(input.tanh()?)
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

impl fmt::Display for Tanh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tanh()")
    }
}


