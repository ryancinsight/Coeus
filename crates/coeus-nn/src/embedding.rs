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
    /// Optional index whose row is forced to all-zeros and whose gradient is zeroed.
    pub padding_idx: Option<usize>,
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Embedding<T, B> {
    /// Create an Embedding layer with weights initialized to ones.
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Result<Self, B::Error> {
        let backend = B::default();
        let w_tensor = Tensor::ones_on([num_embeddings, embedding_dim], &backend);
        let weight = Var::new(w_tensor?, true)?;
        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx: None,
        })
    }

    /// Create with explicit `padding_idx`.
    ///
    /// Row `padding_idx` in the weight matrix is zeroed on construction and
    /// its gradient is zeroed by the autograd embedding backward node.
    pub fn with_padding_idx(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: usize,
    ) -> Result<Self, B::Error>
    where
        B::DeviceBuffer<T>:
            coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
    {
        assert!(
            padding_idx < num_embeddings,
            "Embedding::with_padding_idx: padding_idx {padding_idx} out of bounds [0, {num_embeddings})"
        );
        let backend = B::default();
        let mut w_data = vec![T::one(); num_embeddings * embedding_dim];
        let start = padding_idx * embedding_dim;
        for v in &mut w_data[start..start + embedding_dim] {
            *v = T::zero();
        }
        let w_tensor =
            Tensor::from_slice_on(vec![num_embeddings, embedding_dim], &w_data, &backend)?;
        let weight = Var::new(w_tensor, true)?;
        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx: Some(padding_idx),
        })
    }

    /// Forward pass using explicit integer index tensor.
    pub fn forward_indices<I: Scalar + 'static>(
        &self,
        indices: &Tensor<I, B>,
    ) -> Result<Var<T, B>, B::Error> {
        coeus_autograd::embedding_with_padding_idx(&self.weight, indices, self.padding_idx)
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Embedding<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone()]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, B::Error> {
        coeus_autograd::embedding_with_padding_idx(&self.weight, &input.tensor, self.padding_idx)
    }
}
