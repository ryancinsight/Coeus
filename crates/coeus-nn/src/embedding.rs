// ── Embedding layer module ──

use crate::module::{Module, ModuleError};
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
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let backend = B::default();
        let w_tensor = Tensor::ones_on([num_embeddings, embedding_dim], &backend);
        let weight = Var::new(w_tensor, true);
        Self {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx: None,
        }
    }

    /// Create with explicit `padding_idx`.
    ///
    /// Row `padding_idx` in the weight matrix is zeroed on construction and
    /// its gradient is zeroed by the autograd embedding backward node.
    pub fn with_padding_idx(num_embeddings: usize, embedding_dim: usize, padding_idx: usize) -> Self
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
            Tensor::from_slice_on(vec![num_embeddings, embedding_dim], &w_data, &backend);
        let weight = Var::new(w_tensor, true);
        Self {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx: Some(padding_idx),
        }
    }

    /// Forward pass using an explicit integer index tensor.
    ///
    /// # Errors
    ///
    /// Returns [`ModuleError::ShapeMismatch`] when an index is negative or
    /// outside the configured embedding vocabulary.
    pub fn forward_indices<I: Scalar + 'static>(
        &self,
        indices: &Tensor<I, B>,
    ) -> Result<Var<T, B>, ModuleError<B::Error>> {
        validate_indices(indices, self.num_embeddings, "Embedding")?;
        Ok(coeus_autograd::embedding_with_padding_idx(
            &self.weight,
            indices,
            self.padding_idx,
        ))
    }
}

impl<T: Scalar, B: coeus_ops::BackendOps<T> + Default> Module<T, B> for Embedding<T, B> {
    fn parameters(&self) -> Vec<Var<T, B>> {
        vec![self.weight.clone()]
    }

    fn forward(&self, input: &Var<T, B>) -> Result<Var<T, B>, ModuleError<B::Error>> {
        validate_indices(&input.tensor, self.num_embeddings, "Embedding")?;
        Ok(coeus_autograd::embedding_with_padding_idx(
            &self.weight,
            &input.tensor,
            self.padding_idx,
        ))
    }
}

fn validate_indices<I: Scalar, B: coeus_core::ComputeBackend + Default>(
    indices: &Tensor<I, B>,
    num_embeddings: usize,
    module: &'static str,
) -> Result<(), ModuleError<B::Error>> {
    let backend = B::default();
    for (position, &index) in indices.host_cow_on(&backend).iter().enumerate() {
        let value = <I as Scalar>::to_f64(index);
        if !value.is_finite()
            || value < 0.0
            || value.trunc() != value
            || value >= num_embeddings as f64
        {
            return Err(ModuleError::ShapeMismatch {
                module,
                parameter: "indices must be finite integers within the embedding vocabulary",
                expected: vec![num_embeddings],
                actual: vec![position],
            });
        }
    }

    Ok(())
}
