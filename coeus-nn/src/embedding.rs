// ── Embedding layer module ──

use crate::module::Module;
use coeus_autograd::Var;
use coeus_core::{MoiraiBackend, Scalar};
use coeus_tensor::Tensor;

/// Embedding layer mapping discrete token indices to dense vectors.
#[derive(Clone)]
pub struct Embedding<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Weight matrix: [num_embeddings, embedding_dim]
    pub weight: Var<T, B>,
    /// Number of embeddings (vocabulary size)
    pub num_embeddings: usize,
    /// Dimension of each embedding vector
    pub embedding_dim: usize,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Embedding<T, B> {
    /// Create an Embedding layer with weights initialized to ones.
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let backend = B::default();
        let w_tensor = Tensor::ones_on([num_embeddings, embedding_dim], &backend);
        let weight = Var::new(w_tensor, true);
        Self {
            weight,
            num_embeddings,
            embedding_dim,
        }
    }

    /// Forward pass using explicit integer index tensor.
    pub fn forward_indices<I: Scalar + 'static>(&self, indices: &Tensor<I, B>) -> Var<T, B> {
        coeus_autograd::embedding(&self.weight, indices)
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Embedding<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone()]
    }

    fn forward(&self, input: &Var<T, B>) -> Var<T, B> {
        coeus_autograd::embedding(&self.weight, &input.tensor)
    }
}
