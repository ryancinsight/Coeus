//! Embedding layers for neural networks.

use std::marker::PhantomData;

use coeus_backend::{Backend, CpuBackend};
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage, StorageFromVec};
use coeus_tensor::Tensor;

use crate::error::Result;
use crate::module::Module;
use crate::parameter::Parameter;

/// Embedding layer that converts discrete tokens to continuous vectors.
///
/// This layer performs a simple lookup table operation: `output[i] = weight[input[i]]`.
/// It is commonly used for:
/// - Token embeddings (vocabulary → vectors)
/// - Position embeddings (position → vectors)
/// - Segment embeddings (segment ID → vectors)
///
/// # Examples
/// ```rust
/// use coeus_nn::{Embedding, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// // Create embedding layer: 1000 tokens, 128-dimensional embeddings
/// let embedding = Embedding::new(1000, 128, None).unwrap();
///
/// // Input: [batch_size=2, seq_len=5] token IDs
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
///     vec![Float32::new(0.0), Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0),
///          Float32::new(5.0), Float32::new(6.0), Float32::new(7.0), Float32::new(8.0), Float32::new(9.0)],
///     &[2, 5]
/// ).unwrap();
///
/// // Output: [batch_size=2, seq_len=5, embedding_dim=128]
/// let output = embedding.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 5, 128]);
/// ```
#[derive(Debug, Clone)]
pub struct Embedding<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType,
{
    /// Embedding weight matrix [num_embeddings, embedding_dim]
    pub weight: Parameter<B, S, T>,
    /// Number of embeddings (vocabulary size)
    pub num_embeddings: usize,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Optional padding token index (gradients zeroed for this index)
    pub padding_idx: Option<usize>,
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> Embedding<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + num_traits::Zero,
{
    /// Create a new Embedding layer.
    ///
    /// # Arguments
    /// * `num_embeddings` - Number of embeddings (vocabulary size)
    /// * `embedding_dim` - Dimension of each embedding vector
    /// * `padding_idx` - Optional padding token index (gradients zeroed)
    ///
    /// # Weight Initialization
    /// Weights are initialized using Xavier uniform initialization:
    /// `U(-√(1/embedding_dim), √(1/embedding_dim))`
    pub fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
    ) -> Result<Self> {
        assert!(num_embeddings > 0, "num_embeddings must be > 0");
        assert!(embedding_dim > 0, "embedding_dim must be > 0");

        if let Some(idx) = padding_idx {
            assert!(idx < num_embeddings, "padding_idx must be < num_embeddings");
        }

        // Xavier uniform initialization: U(-√(1/d), √(1/d))
        let bound = (1.0 / embedding_dim as f64).sqrt();
        let weight_data: Vec<T> = (0..num_embeddings * embedding_dim)
            .map(|_| {
                let random_val = rand::random::<f64>();
                let val = (random_val * 2.0 - 1.0) * bound;
                T::from(val).unwrap()
            })
            .collect();

        let weight_tensor =
            Tensor::<B, S, T>::from_vec(weight_data, &[num_embeddings, embedding_dim])?;

        let weight = Parameter::new(weight_tensor.requires_grad_(true), "weight".to_string());

        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx,
            _phantom: PhantomData,
        })
    }

    /// Create an Embedding layer from pre-trained weights.
    ///
    /// # Arguments
    /// * `weight` - Pre-trained embedding matrix [num_embeddings, embedding_dim]
    /// * `padding_idx` - Optional padding token index
    pub fn from_pretrained(weight: Tensor<B, S, T>, padding_idx: Option<usize>) -> Result<Self> {
        let shape = weight.shape().dims();
        assert_eq!(
            shape.len(),
            2,
            "Weight must be 2D [num_embeddings, embedding_dim]"
        );

        let num_embeddings = shape[0];
        let embedding_dim = shape[1];

        if let Some(idx) = padding_idx {
            assert!(idx < num_embeddings, "padding_idx must be < num_embeddings");
        }

        let weight_param = Parameter::new(weight.requires_grad_(true), "weight".to_string());

        Ok(Self {
            weight: weight_param,
            num_embeddings,
            embedding_dim,
            padding_idx,
            _phantom: PhantomData,
        })
    }
}

impl<B, S, T> Module<B, S, T> for Embedding<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + 'static,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Input: [batch_size, seq_len] or [seq_len] (integer token IDs)
        // Output: [batch_size, seq_len, embedding_dim] or [seq_len, embedding_dim]

        let input_shape = input.shape().dims();
        let weight_data = self.weight.data().as_slice();
        let input_data = input.as_slice();

        // Convert input to indices (assuming input contains integer values as floats)
        let indices: Vec<usize> = input_data
            .iter()
            .map(|&x| {
                let idx = x.to_f64().unwrap() as usize;
                assert!(
                    idx < self.num_embeddings,
                    "Index {} out of bounds (num_embeddings={})",
                    idx,
                    self.num_embeddings
                );
                idx
            })
            .collect();

        // Lookup embeddings
        let mut output_data = Vec::with_capacity(indices.len() * self.embedding_dim);
        for &idx in &indices {
            let start = idx * self.embedding_dim;
            let end = start + self.embedding_dim;
            output_data.extend_from_slice(&weight_data[start..end]);
        }

        // Output shape: [...input_shape, embedding_dim]
        let mut output_shape = input_shape.to_vec();
        output_shape.push(self.embedding_dim);

        Tensor::from_vec(output_data, &output_shape).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![self.weight.clone()]
    }

    fn zero_grad(&mut self) {
        self.weight.zero_grad();
    }

    fn train(&mut self, _mode: bool) {
        // No-op: Embedding behavior doesn't change between train/eval
    }

    fn name(&self) -> &str {
        "Embedding"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use coeus_dtype::float::Float32;

    #[test]
    fn test_embedding_forward() {
        let embedding =
            Embedding::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 4, None);

        // Input: [2, 3] token IDs
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(0.0),
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
                Float32::new(5.0),
            ],
            &[2, 3],
        )
        .unwrap();

        let embedding = embedding.unwrap();
        let output = embedding.forward(&input).unwrap();

        // Output shape: [2, 3, 4]
        assert_eq!(output.shape().dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_embedding_1d_input() {
        let embedding =
            Embedding::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 4, None);

        // Input: [5] token IDs
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(0.0),
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
            ],
            &[5],
        )
        .unwrap();

        let embedding = embedding.unwrap();
        let output = embedding.forward(&input).unwrap();

        // Output shape: [5, 4]
        assert_eq!(output.shape().dims(), &[5, 4]);
    }

    #[test]
    fn test_embedding_lookup_correctness() {
        // Create embedding with known weights
        let weight_data: Vec<Float32> = (0..20).map(|i| Float32::new(i as f32)).collect();
        let weight = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            weight_data,
            &[5, 4],
        )
        .unwrap();

        let embedding =
            Embedding::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_pretrained(
                weight, None,
            )
            .unwrap();

        // Input: [2] token IDs [0, 2]
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();
        let output = embedding.forward(&input).unwrap();
        let output_data: Vec<f32> = output
            .as_slice()
            .iter()
            .map(|x: &Float32| x.get())
            .collect();

        // Expected: [0, 1, 2, 3] (token 0) and [8, 9, 10, 11] (token 2)
        assert_relative_eq!(output_data[0], 0.0);
        assert_relative_eq!(output_data[1], 1.0);
        assert_relative_eq!(output_data[2], 2.0);
        assert_relative_eq!(output_data[3], 3.0);
        assert_relative_eq!(output_data[4], 8.0);
        assert_relative_eq!(output_data[5], 9.0);
        assert_relative_eq!(output_data[6], 10.0);
        assert_relative_eq!(output_data[7], 11.0);
    }

    #[test]
    fn test_embedding_parameters() {
        let embedding = EmbeddingCpu::<Float32>::new(10, 4, None).unwrap();

        let params = embedding.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].name(), "weight");
        assert!(params[0].requires_grad());
    }

    #[test]
    #[should_panic(expected = "Index 10 out of bounds (num_embeddings=10)")]
    fn test_embedding_out_of_bounds() {
        let embedding = EmbeddingCpu::<Float32>::new(10, 4, None).unwrap();

        // Input with out-of-bounds index
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(10.0)], // Index 10 is out of bounds (max is 9)
            &[1],
        )
        .unwrap();

        let _ = embedding.forward(&input);
    }
}

// ============================================================================
// TYPE ALIASES FOR BACKWARD COMPATIBILITY
// ============================================================================

/// Type alias for Embedding layer with CPU backend and dense storage.
/// This provides backward compatibility with existing code.
pub type EmbeddingCpu<T> = Embedding<CpuBackend<T>, DenseStorage<T>, T>;
