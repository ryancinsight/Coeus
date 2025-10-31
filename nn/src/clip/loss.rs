//! Contrastive loss functions for CLIP training
//!
//! This module implements InfoNCE loss and other contrastive learning objectives
//! used in CLIP training.

use std::fmt;
use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageToDense};
use tensor::Tensor;

use crate::error::{NNError, Result};

/// InfoNCE (Noise-Contrastive Estimation) loss for contrastive learning
///
/// This implements the symmetric InfoNCE loss used in CLIP:
/// L = (L_image→text + L_text→image) / 2
/// where L_x→y = -log(exp(sim(x,y)/τ) / Σ_{j} exp(sim(x,y_j)/τ))
///
/// # Arguments
/// * `image_features` - Image embeddings [batch_size, embed_dim]
/// * `text_features` - Text embeddings [batch_size, embed_dim]
/// * `temperature` - Temperature scaling parameter (typically 0.07)
///
/// # Returns
/// Scalar loss tensor
///
pub fn info_nce_loss<B, S, T>(
    image_features: &Tensor<B, S, T>,
    text_features: &Tensor<B, S, T>,
    temperature: f64,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageToDense<T> + 'static,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    let image_shape = image_features.shape().dims();
    let text_shape = text_features.shape().dims();

    if image_shape.len() != 2 || text_shape.len() != 2 {
        return Err(NNError::ShapeMismatch {
            operation: "InfoNCE loss".to_string(),
            expected: vec![0, 0],
            actual: image_shape.to_vec(),
        });
    }

    if image_shape != text_shape {
        return Err(NNError::ShapeMismatch {
            operation: "InfoNCE loss - image vs text".to_string(),
            expected: image_shape.to_vec(),
            actual: text_shape.to_vec(),
        });
    }

    let batch_size = image_shape[0];
    if batch_size < 2 {
        return Err(NNError::InvalidInput {
            message: "InfoNCE loss requires batch_size >= 2".to_string(),
        });
    }

    // Convert to dense for computation
    let image_dense = image_features.to_dense_generic()?;
    let text_dense = text_features.to_dense_generic()?;

    // Normalize features to unit vectors (cosine similarity)
    let image_norm = normalize_embeddings(&image_dense)?;
    let text_norm = normalize_embeddings(&text_dense)?;

    // Compute similarity matrix: [batch_size, batch_size]
    let logits_image_to_text = compute_similarity_matrix(&image_norm, &text_norm)?;
    let logits_text_to_image = compute_similarity_matrix(&text_norm, &image_norm)?;

    // Scale by temperature
    let temp_tensor = Tensor::<B, DenseStorage<T>, T>::from_vec(
        vec![T::from(temperature).unwrap()],
        &[1],
    )?;
    let scaled_logits_i2t = &logits_image_to_text / &temp_tensor;
    let scaled_logits_t2i = &logits_text_to_image / &temp_tensor;

    // Compute cross-entropy loss for both directions
    let loss_i2t = cross_entropy_from_logits(&scaled_logits_i2t, batch_size)?;
    let loss_t2i = cross_entropy_from_logits(&scaled_logits_t2i, batch_size)?;

    // Average the two losses
    let total_loss = (&loss_i2t + &loss_t2i) / &T::from(2.0).unwrap();

    Ok(Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
        vec![total_loss],
        &[1],
        image_features.backend().clone(),
    )?)
}

/// InfoNCE Loss module for CLIP training
#[derive(Debug, Clone)]
pub struct InfoNCELoss<T> {
    temperature: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> InfoNCELoss<T>
where
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    /// Create new InfoNCE loss
    pub fn new(temperature: f64) -> Self {
        Self {
            temperature,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute InfoNCE loss for CLIP
    pub fn forward<B, S>(
        &self,
        image_features: &Tensor<B, S, T>,
        text_features: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, DenseStorage<T>, T>>
    where
        B: Backend<Data = T> + Clone + Default,
        S: Storage<T> + Clone + StorageToDense<T> + 'static,
    {
        info_nce_loss(image_features, text_features, self.temperature)
    }

    /// Get temperature parameter
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Set temperature parameter
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }
}

impl<T> Default for InfoNCELoss<T>
where
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    fn default() -> Self {
        Self::new(0.07) // CLIP default temperature
    }
}

impl<T> fmt::Display for InfoNCELoss<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InfoNCELoss(temperature={})", self.temperature)
    }
}

// Helper functions

/// Normalize embeddings to unit vectors for cosine similarity
fn normalize_embeddings<B, T>(
    embeddings: &Tensor<B, DenseStorage<T>, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone + Default,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    let shape = embeddings.shape().dims();
    let batch_size = shape[0];
    let embed_dim = shape[1];

    // Compute L2 norms: sqrt(sum(x^2)) for each embedding
    let mut norms = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let start = i * embed_dim;
        let end = start + embed_dim;
        let embedding: Vec<T> = embeddings.as_slice()[start..end].to_vec();

        let mut norm_sq = T::zero();
        for &val in &embedding {
            norm_sq = norm_sq + val * val;
        }
        norms.push(norm_sq.sqrt());
    }

    // Create norms tensor for element-wise division
    let norms_tensor = Tensor::<B, DenseStorage<T>, T>::from_vec_with_backend(
        norms,
        &[batch_size, 1],
        embeddings.backend().clone(),
    )?;

    // Broadcast divide embeddings by norms
    embeddings.div(&norms_tensor.expand(&[batch_size, embed_dim])?)
}

/// Compute similarity matrix between two sets of normalized embeddings
fn compute_similarity_matrix<B, T>(
    embeddings1: &Tensor<B, DenseStorage<T>, T>,
    embeddings2: &Tensor<B, DenseStorage<T>, T>,
) -> Result<Tensor<B, DenseStorage<T>, T>>
where
    B: Backend<Data = T> + Clone,
    T: DataType + FloatExt,
{
    // Compute dot product between all pairs: [batch, embed_dim] @ [embed_dim, batch] -> [batch, batch]
    let embeddings2_t = embeddings2.transpose(0, 1)?;
    embeddings1.matmul(&embeddings2_t)
}

/// Compute cross-entropy loss from logits matrix
fn cross_entropy_from_logits<B, T>(
    logits: &Tensor<B, DenseStorage<T>, T>,
    batch_size: usize,
) -> Result<T>
where
    B: Backend<Data = T> + Clone,
    T: DataType + FloatExt + std::ops::Neg<Output = T>,
{
    // Convert to dense data for computation
    let logits_data = logits.as_slice();

    let mut total_loss = T::zero();

    for i in 0..batch_size {
        // Get logits for row i (similarities with all j)
        let row_start = i * batch_size;
        let row_logits: Vec<T> = logits_data[row_start..row_start + batch_size].to_vec();

        // Find max for numerical stability
        let mut max_logit = *row_logits.iter().min().unwrap(); // Initialize with minimum
        for &val in &row_logits {
            if val > max_logit {
                max_logit = val;
            }
        }

        // Compute stable softmax: exp(logit - max_logit)
        let mut exp_logits = Vec::with_capacity(batch_size);
        let mut sum_exp = T::zero();

        for &logit in &row_logits {
            let exp_logit = (logit - max_logit).exp();
            exp_logits.push(exp_logit);
            sum_exp = sum_exp + exp_logit;
        }

        // Probability for correct pair (diagonal element)
        let correct_prob = exp_logits[i] / sum_exp;

        // Cross-entropy loss: -log(correct_prob)
        let epsilon = T::from(1e-10).unwrap();
        let safe_prob = correct_prob.max(epsilon);
        let loss = -(safe_prob.ln());

        total_loss = total_loss + loss;
    }

    // Return average loss
    Ok(total_loss / T::from(batch_size as f64).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;

    type TestBackend = CpuBackend<Float32>;
    type TestStorage = DenseStorage<Float32>;
    type TestTensor = Tensor<TestBackend, TestStorage, Float32>;

    #[test]
    fn test_info_nce_loss_basic() {
        let batch_size = 4;
        let embed_dim = 128;

        // Create random embeddings (normalized)
        let image_features = TestTensor::randn(&[batch_size, embed_dim]).unwrap();
        let text_features = TestTensor::randn(&[batch_size, embed_dim]).unwrap();

        let temperature = 0.07;
        let loss = info_nce_loss(&image_features, &text_features, temperature).unwrap();

        let loss_val = loss.as_slice()[0];
        assert!(loss_val >= 0.0, "Loss should be non-negative");

        // For random features, loss should be around log(batch_size)
        let expected_loss = (batch_size as f32).ln();
        assert!(loss_val >= expected_loss * 0.5, "Loss seems too low");
    }

    #[test]
    fn test_info_nce_perfect_match() {
        let batch_size = 2;
        let embed_dim = 64;

        // Create identical embeddings
        let embeddings = TestTensor::randn(&[batch_size, embed_dim]).unwrap();
        let image_features = embeddings.clone();
        let text_features = embeddings;

        let loss = info_nce_loss(&image_features, &text_features, 0.07).unwrap();
        let loss_val = loss.as_slice()[0];

        // With identical embeddings, loss should be very close to 0
        assert!(loss_val < 0.1, "Perfect match should have very low loss");
    }

    #[test]
    fn test_info_nce_module() {
        let loss_fn = InfoNCELoss::<Float32>::new(0.1);
        assert_eq!(loss_fn.temperature(), 0.1);

        let mut loss_fn = InfoNCELoss::<Float32>::default();
        assert_eq!(loss_fn.temperature(), 0.07);

        loss_fn.set_temperature(0.05);
        assert_eq!(loss_fn.temperature(), 0.05);
    }

    #[test]
    fn test_info_nce_invalid_input() {
        let single_batch = TestTensor::randn(&[1, 64]).unwrap();

        // Should fail with batch_size < 2
        let result = info_nce_loss(&single_batch, &single_batch, 0.07);
        assert!(result.is_err());

        // Should fail with mismatched shapes
        let b2 = TestTensor::randn(&[2, 64]).unwrap();
        let b3 = TestTensor::randn(&[3, 64]).unwrap();
        let result = info_nce_loss(&b2, &b3, 0.07);
        assert!(result.is_err());
    }
}
