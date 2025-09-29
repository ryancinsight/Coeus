//! Multi-Layer Perceptron (MLP) module for transformer networks
//!
//! This module provides the feed-forward network component used in transformer architectures.
//! The MLP consists of two linear layers with a GELU activation in between.

use crate::{GELU, Linear, Module, Result};
use coeus_backend::{Backend, CpuBackend};
use coeus_dtype::Dtype;
use coeus_tensor::{FloatDtype, Tensor, CpuBackend};

/// Multi-Layer Perceptron for transformer networks
#[derive(Debug, Clone)]
pub struct MLP<T: FloatDtype, B: Backend<T> + Clone + Send + Sync = CpuBackend> {
    /// First linear layer
    pub c_fc: Linear<T, B>,
    /// GELU activation
    pub gelu: GELU,
    /// Second linear layer
    pub c_proj: Linear<T, B>,
}

impl<T: FloatDtype + rand::distributions::uniform::SampleUniform, B: Backend<T> + Clone + Default + Send + Sync> MLP<T, B> {
    /// Create a new MLP block
    ///
    /// # Arguments
    /// * `n_embd` - Embedding dimension
    pub fn new(n_embd: usize) -> Result<Self> {
        // GPT-2 uses 4x expansion for the intermediate layer
        let intermediate_size = 4 * n_embd;

        let c_fc = Linear::new(n_embd, intermediate_size)?;
        let gelu = GELU::new();
        let c_proj = Linear::new(intermediate_size, n_embd)?;

        Ok(Self { c_fc, gelu, c_proj })
    }
}

impl<T: FloatDtype, B: Backend<T> + Clone + Send + Sync + Default> Module<T, B> for MLP<T, B> {
    fn forward(&self, input: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        // MLP: input -> Linear -> GELU -> Linear -> output
        let hidden = self.c_fc.forward(input)?;
        let activated = self.gelu.forward(&hidden)?;
        let output = self.c_proj.forward(&activated)?;

        Ok(output)
    }

    fn parameters(&self) -> Vec<&Tensor<T, B>> {
        let mut params = Vec::new();
        params.extend(self.c_fc.parameters());
        params.extend(self.c_proj.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, B>> {
        let mut params = Vec::new();
        params.extend(self.c_fc.parameters_mut());
        params.extend(self.c_proj.parameters_mut());
        params
    }
}

impl<T: FloatDtype, B: Backend<T> + Clone> std::fmt::Display for MLP<T, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MLP(c_fc={}, c_proj={})",
            self.c_fc.out_features, self.c_proj.out_features
        )
    }
}


