//! Softmax-family activation functions
//!
//! This module contains activation functions based on the softmax function.

use crate::Module;
use coeus_tensor::{FloatDtype, Tensor, CpuBackend};
use std::fmt;
use std::marker::PhantomData;

/// Softmax activation function
///
/// Formula: `Softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)`
///
/// Softmax converts a vector of real numbers into a probability distribution.
/// It's commonly used in the output layer of classification networks.
#[derive(Debug, Clone)]
pub struct Softmax<T: FloatDtype> {
    dim: Option<usize>,
    _marker: PhantomData<T>,
}

impl<T: FloatDtype> Default for Softmax<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatDtype> Softmax<T> {
    /// Create a new Softmax activation with default dimension (-1)
    pub fn new() -> Self {
        Self { dim: Some(0), _marker: PhantomData } // Default to last dimension
    }

    /// Create a new Softmax activation with specified dimension
    pub fn with_dim(dim: usize) -> Self {
        Self { dim: Some(dim), _marker: PhantomData }
    }
}

impl<T: FloatDtype + Clone> Module<T> for Softmax<T> {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        match self.dim {
            Some(dim) => Ok(input.softmax(dim)?),
            None => Ok(input.softmax(input.ndim() - 1)?), // Default to last dimension
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

impl<T: FloatDtype + Clone> fmt::Display for Softmax<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.dim {
            Some(dim) => write!(f, "Softmax(dim={})", dim),
            None => write!(f, "Softmax()"),
        }
    }
}

/// Softmin activation function
///
/// Formula: `Softmin(x_i) = exp(-x_i) / sum(exp(-x_j) for all j)`
///
/// Softmin is the "soft" version of argmin, producing a probability distribution
/// where higher values get lower probabilities.
#[derive(Debug, Clone)]
pub struct Softmin<T: FloatDtype> {
    dim: Option<usize>,
    _marker: PhantomData<T>,
}

impl<T: FloatDtype> Default for Softmin<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: FloatDtype> Softmin<T> {
    /// Create a new Softmin activation with default dimension (-1)
    pub fn new() -> Self {
        Self { dim: Some(0), _marker: PhantomData } // Default to last dimension
    }

    /// Create a new Softmin activation with specified dimension
    pub fn with_dim(dim: usize) -> Self {
        Self { dim: Some(dim), _marker: PhantomData }
    }
}

impl<T: FloatDtype + Clone> Module<T> for Softmin<T> {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        match self.dim {
            Some(dim) => Ok(input.softmin(dim)?),
            None => Ok(input.softmin(input.ndim() - 1)?), // Default to last dimension
        }
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

impl<T: FloatDtype + Clone> fmt::Display for Softmin<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.dim {
            Some(dim) => write!(f, "Softmin(dim={})", dim),
            None => write!(f, "Softmin()"),
        }
    }
}

/// Softmax2d activation function
///
/// Applies softmax over the C dimension of a 4D tensor of shape (N, C, H, W).
/// This is commonly used in segmentation tasks.
#[derive(Debug, Clone, Copy, Default)]
pub struct Softmax2d;

impl Softmax2d {
    /// Create a new Softmax2d activation
    pub fn new() -> Self {
        Softmax2d
    }
}

impl<T: FloatDtype> Module<T> for Softmax2d {
    fn forward(&self, input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        // Apply softmax over the channel dimension (dim=1) for (N, C, H, W) tensors
        Ok(input.softmax(1)?)
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

impl fmt::Display for Softmax2d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Softmax2d()")
    }
}


