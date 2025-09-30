//! Ranking loss functions
//!
//! Loss functions for learning relative ordering and ranking tasks,
//! commonly used in information retrieval, recommendation systems,
//! and metric learning applications.

use super::{utils, Module, Reduction};
use coeus_tensor::{Dtype, FloatDtype, Tensor, CpuBackend};

/// Margin Ranking Loss
///
/// Computes margin ranking loss for ranking tasks:
/// `MarginRanking(x1, x2, y) = max(0, -y * (x1 - x2) + margin)`
///
/// Where y = 1 if x1 should rank higher than x2, and y = -1 otherwise.
/// This is commonly used in learning-to-rank applications.
///
/// # Mathematical Properties
/// - Hinge loss variant for ranking
/// - Margin parameter controls the minimum separation
/// - Gradient: dL/dx1 = -y if loss > 0, else 0
/// - Gradient: dL/dx2 = y if loss > 0, else 0
///
/// # Example
/// ```rust
/// use coeus_nn::MarginRankingLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = MarginRankingLoss::new(1.0);
/// let input1 = Tensor::from_vec(CpuBackend::default(), vec![1.0).unwrap();
/// let input2 = Tensor::from_vec(CpuBackend::default(), vec![0.5).unwrap();
/// let target = Tensor::from_vec(CpuBackend::default(), vec![1.0).unwrap();
///
/// let loss = loss_fn.forward(&input1, &input2, &target).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MarginRankingLoss {
    /// Margin parameter (default: 0.0)
    pub margin: f64,
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl Default for MarginRankingLoss {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl MarginRankingLoss {
    /// Create a new margin ranking loss function
    pub fn new(margin: f64) -> Self {
        Self {
            margin,
            reduction: Reduction::Mean,
        }
    }

    /// Create with specified margin and reduction
    pub fn with_params(margin: f64, reduction: Reduction) -> Self {
        Self { margin, reduction }
    }

    /// Compute margin ranking loss
    pub fn forward<T: FloatDtype + std::iter::Sum>(
        &self,
        input1: &Tensor<T, CpuBackend>,
        input2: &Tensor<T, CpuBackend>,
        target: &Tensor<T, CpuBackend>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        if input1.shape() != input2.shape() || input1.shape() != target.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: input1.shape().to_vec(),
                actual: vec![
                    input1.shape().len(),
                    input2.shape().len(),
                    target.shape().len(),
                ],
            });
        }

        let input1_data = input1.data();
        let input2_data = input2.data();
        let target_data = target.data();

        let mut losses = Vec::with_capacity(input1.numel());
        let margin = T::from(self.margin).unwrap();

        for ((&x1, &x2), &y) in input1_data
            .iter()
            .zip(input2_data.iter())
            .zip(target_data.iter())
        {
            // MarginRanking = max(0, -y * (x1 - x2) + margin)
            let diff = x1 - x2;
            let loss_val = (-y * diff + margin).max(T::zero());
            losses.push(loss_val);
        }

        // CpuBackend has no Default impl; use new() and apply reduction
        let loss_tensor = Tensor::from_vec(CpuBackend::new(), losses, input1.shape().to_vec()).unwrap();
        utils::apply_reduction(&loss_tensor, self.reduction)
    }

    /// Compute analytical gradients w.r.t. input1 and input2
    ///
    /// Returns a tuple (grad_input1, grad_input2) shaped like inputs.
    pub fn backward<T: FloatDtype>(
        &self,
        input1: &Tensor<T, CpuBackend>,
        input2: &Tensor<T, CpuBackend>,
        target: &Tensor<T, CpuBackend>,
    ) -> crate::Result<(Tensor<T, CpuBackend>, Tensor<T, CpuBackend>)> {
        if input1.shape() != input2.shape() || input1.shape() != target.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: input1.shape().to_vec(),
                actual: vec! [
                    input1.shape().len(),
                    input2.shape().len(),
                    target.shape().len(),
                ],
            });
        }

        let input1_data = input1.data();
        let input2_data = input2.data();
        let target_data = target.data();
        let n = input1.numel();

        let mut grad1: Vec<T> = vec![T::zero(); n];
        let mut grad2: Vec<T> = vec![T::zero(); n];

        let margin_t = T::from(self.margin).unwrap();

        // Reduction scaling
        let scale = match self.reduction {
            Reduction::None => T::one(),
            Reduction::Sum => T::one(),
            Reduction::Mean => T::from(1.0 / (n as f64)).unwrap(),
        };

        for i in 0..n {
            let x1 = input1_data[i];
            let x2 = input2_data[i];
            let y = target_data[i];
            let diff = x1 - x2;
            let loss_val = (-y * diff + margin_t).max(T::zero());

            if loss_val > T::zero() {
                // dL/dx1 = -y, dL/dx2 = y, apply reduction scaling
                grad1[i] = (-y) * scale;
                grad2[i] = y * scale;
            } else {
                grad1[i] = T::zero();
                grad2[i] = T::zero();
            }
        }

        let g1 = Tensor::from_vec(CpuBackend::new(), grad1, input1.shape().to_vec()).unwrap();
        let g2 = Tensor::from_vec(CpuBackend::new(), grad2, input1.shape().to_vec()).unwrap();
        Ok((g1, g2))
    }
}

impl<T: FloatDtype> Module<T> for MarginRankingLoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(crate::NNError::InvalidInput {
            message: "MarginRankingLoss requires three inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

// Hinge Embedding Loss
///
/// Computes hinge embedding loss for learning embeddings:
/// ```rust,ignore
/// HingeEmbedding(x, y) = {
///     x                    if y = 1
///     max(0, margin - x)   if y = -1
/// }
/// ```
///
/// This is used for learning embeddings where similar pairs should have small distances
/// and dissimilar pairs should have distances larger than a margin.
#[derive(Debug, Clone, Copy)]
pub struct HingeEmbeddingLoss {
    /// Margin parameter (default: 1.0)
    pub margin: f64,
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl Default for HingeEmbeddingLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl HingeEmbeddingLoss {
    /// Create a new hinge embedding loss function
    pub fn new(margin: f64) -> Self {
        Self {
            margin,
            reduction: Reduction::Mean,
        }
    }

    /// Create with specified margin and reduction
    pub fn with_params(margin: f64, reduction: Reduction) -> Self {
        Self { margin, reduction }
    }

    /// Compute hinge embedding loss
    pub fn forward<T: FloatDtype + std::iter::Sum>(
        &self,
        input: &Tensor<T, CpuBackend>,
        target: &Tensor<T, CpuBackend>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        if input.shape() != target.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: input.shape().to_vec(),
                actual: target.shape().to_vec(),
            });
        }

        let input_data = input.data();
        let target_data = target.data();

        let mut losses = Vec::with_capacity(input.numel());
        let margin = T::from(self.margin).unwrap();
        let one = T::one();

        for (&x, &y) in input_data.iter().zip(target_data.iter()) {
            let loss_val = if y == one {
                // Similar pair: loss = x
                x
            } else {
                // Dissimilar pair: loss = max(0, margin - x)
                (margin - x).max(T::zero())
            };
            losses.push(loss_val);
        }

        let loss_tensor = Tensor::from_vec(CpuBackend::new(), losses, input.shape().to_vec()).unwrap();
        utils::apply_reduction(&loss_tensor, self.reduction)
    }
}

impl<T: FloatDtype> Module<T> for HingeEmbeddingLoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(crate::NNError::InvalidInput {
            message: "HingeEmbeddingLoss requires two inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

///
/// Triplet Margin Loss
///
/// Computes triplet margin loss for metric learning:
/// `TripletMargin = max(0, ||anchor - positive||_p - ||anchor - negative||_p + margin)`
///
/// This loss is used to learn embeddings where the distance between an anchor and a positive
/// example is smaller than the distance between the anchor and a negative example by at least
/// a margin.
#[derive(Debug, Clone, Copy)]
pub struct TripletMarginLoss {
    /// Margin parameter (default: 1.0)
    pub margin: f64,
    /// p-norm degree (default: 2.0 for Euclidean distance)
    pub p: f64,
    /// Small constant for numerical stability
    pub eps: f64,
    /// Whether to swap positive and negative if it violates ranking
    pub swap: bool,
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl Default for TripletMarginLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl TripletMarginLoss {
    /// Create a new triplet margin loss with default parameters
    pub fn new() -> Self {
        Self {
            margin: 1.0,
            p: 2.0,
            eps: 1e-6,
            swap: false,
            reduction: Reduction::Mean,
        }
    }

    /// Create with all parameters specified
    pub fn with_params(margin: f64, p: f64, eps: f64, swap: bool, reduction: Reduction) -> Self {
        Self {
            margin,
            p,
            eps,
            swap,
            reduction,
        }
    }

    /// Create with default parameters except margin and reduction
    pub fn with_margin_and_reduction(margin: f64, reduction: Reduction) -> Self {
        Self {
            margin,
            p: 2.0,
            eps: 1e-6,
            swap: false,
            reduction,
        }
    }

    /// Compute triplet margin loss
    pub fn forward<T: FloatDtype + std::iter::Sum>(
        &self,
        anchor: &Tensor<T, CpuBackend>,
        positive: &Tensor<T, CpuBackend>,
        negative: &Tensor<T, CpuBackend>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        if anchor.shape() != positive.shape() || anchor.shape() != negative.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: anchor.shape().to_vec(),
                actual: vec![
                    anchor.shape().len(),
                    positive.shape().len(),
                    negative.shape().len(),
                ],
            });
        }

        // For simplicity, implement L2 distance (p=2) case
        // More general p-norm would require additional tensor operations
        let anchor_data = anchor.data();
        let positive_data = positive.data();
        let negative_data = negative.data();

        let batch_size = anchor.shape()[0];
        let feature_size = anchor.numel() / batch_size;

        let mut losses = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let start_idx = i * feature_size;
            let end_idx = (i + 1) * feature_size;

            // Compute L2 distances
            let mut pos_dist_sq = T::zero();
            let mut neg_dist_sq = T::zero();

            for j in start_idx..end_idx {
                let anchor_val = anchor_data[j];
                let pos_val = positive_data[j];
                let neg_val = negative_data[j];

                let pos_diff = anchor_val - pos_val;
                let neg_diff = anchor_val - neg_val;

                pos_dist_sq = pos_dist_sq + pos_diff * pos_diff;
                neg_dist_sq = neg_dist_sq + neg_diff * neg_diff;
            }

            let pos_dist = pos_dist_sq.sqrt();
            let neg_dist = neg_dist_sq.sqrt();

            // TripletMargin = max(0, pos_dist - neg_dist + margin)
            let loss_val = (pos_dist - neg_dist + T::from(self.margin).unwrap()).max(T::zero());
            losses.push(loss_val);
        }

        let loss_tensor = Tensor::from_vec(CpuBackend::new(), losses, vec![batch_size]).unwrap();
        utils::apply_reduction(&loss_tensor, self.reduction)
    }
}

impl<T: FloatDtype> Module<T> for TripletMarginLoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(crate::NNError::InvalidInput {
            message: "TripletMarginLoss requires three inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

/// Multi-Margin Loss
///
/// Computes multi-margin loss for multi-class classification:
/// `MultiMarginLoss(x, y) = sum_{i≠y} max(0, (margin - x[y] + x[i])^p) / (batch_size * num_classes)`
///
/// This is a generalization of the hinge loss to multi-class classification.
/// The loss encourages the correct class score to be higher than all other class scores by at least a margin.
///
/// # Mathematical Properties
/// - Multi-class hinge loss variant
/// - Margin parameter controls the minimum separation
/// - p-norm exponent (typically 1 or 2)
/// - Encourages separation between correct and incorrect classes
///
/// # Example
/// ```rust
/// use coeus_nn::MultiMarginLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = MultiMarginLoss::new(1.0, 1);
/// let input = Tensor::from_vec(CpuBackend::default(), vec![0.2], vec![1]).unwrap();
/// let target = Tensor::from_vec(CpuBackend::default(), vec![1], vec![1]).unwrap(); // Class indices
///
/// let loss = loss_fn.forward(&input, &target).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MultiMarginLoss {
    /// Margin parameter (default: 1.0)
    pub margin: f64,
    /// p-norm degree (default: 1)
    pub p: i32,
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl Default for MultiMarginLoss {
    fn default() -> Self {
        Self::new(1.0, 1)
    }
}

impl MultiMarginLoss {
    /// Create a new multi-margin loss function
    pub fn new(margin: f64, p: i32) -> Self {
        Self {
            margin,
            p,
            reduction: Reduction::Mean,
        }
    }

    /// Create with specified margin, p-norm, and reduction
    pub fn with_params(margin: f64, p: i32, reduction: Reduction) -> Self {
        Self {
            margin,
            p,
            reduction,
        }
    }

    /// Compute multi-margin loss
    pub fn forward<T: FloatDtype, I: Dtype + num_traits::ToPrimitive>(
        &self,
        input: &Tensor<T, CpuBackend>,
        target: &Tensor<I, CpuBackend>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        if input.ndim() != 2 {
            return Err(crate::NNError::InvalidInput {
                message: "Input must be 2D tensor (batch_size, num_classes)".to_string(),
            });
        }

        if target.ndim() != 1 {
            return Err(crate::NNError::InvalidInput {
                message: "Target must be 1D tensor (batch_size,)".to_string(),
            });
        }

        if input.shape()[0] != target.shape()[0] {
            return Err(crate::NNError::ShapeMismatch {
                expected: vec![input.shape()[0]],
                actual: target.shape().to_vec(),
            });
        }

        let batch_size = input.shape()[0];
        let num_classes = input.shape()[1];
        let margin = T::from(self.margin).unwrap();

        let mut total_loss = T::zero();
        let input_data = input.data();
        let target_data = target.data();

        for batch_idx in 0..batch_size {
            let target_idx = target_data[batch_idx].to_usize().unwrap();
            if target_idx >= num_classes {
                return Err(crate::NNError::InvalidInput {
                    message: format!(
                        "Target index {} out of bounds for {} classes",
                        target_idx, num_classes
                    ),
                });
            }

            let correct_score = input_data[batch_idx * num_classes + target_idx];

            // Sum losses over all incorrect classes
            let mut sample_loss = T::zero();
            for class_idx in 0..num_classes {
                if class_idx == target_idx {
                    continue;
                }

                let incorrect_score = input_data[batch_idx * num_classes + class_idx];
                let diff = margin - correct_score + incorrect_score;

                // Apply p-norm
                let loss_term = if self.p == 1 {
                    diff.max(T::zero())
                } else {
                    // For p != 1, we need to handle the p-norm properly
                    let p_float = T::from(self.p as f64).unwrap();
                    if diff > T::zero() {
                        diff.powf(p_float)
                    } else {
                        T::zero()
                    }
                };

                sample_loss = sample_loss + loss_term;
            }

            total_loss = total_loss + sample_loss;
        }

        // Apply reduction
        let num_samples = T::from(batch_size as f64).unwrap();
        let _num_classes_float = T::from(num_classes as f64).unwrap();

        match self.reduction {
            Reduction::None => {
                // Return per-sample losses
                let mut losses = Vec::with_capacity(batch_size);
                for batch_idx in 0..batch_size {
                    let target_idx = target_data[batch_idx].to_usize().unwrap();
                    let correct_score = input_data[batch_idx * num_classes + target_idx];

                    let mut sample_loss = T::zero();
                    for class_idx in 0..num_classes {
                        if class_idx == target_idx {
                            continue;
                        }

                        let incorrect_score = input_data[batch_idx * num_classes + class_idx];
                        let diff = margin - correct_score + incorrect_score;

                        let loss_term = if self.p == 1 {
                            diff.max(T::zero())
                        } else {
                            let p_float = T::from(self.p as f64).unwrap();
                            if diff > T::zero() {
                                diff.powf(p_float)
                            } else {
                                T::zero()
                            }
                        };

                        sample_loss = sample_loss + loss_term;
                    }
                    losses.push(sample_loss);
                }
                // Construct per-sample loss tensor
                let losses = losses; // already computed in this branch
                let per_sample = Tensor::from_vec(CpuBackend::new(), losses, vec![batch_size]).unwrap();
                Ok(per_sample)
            }
            Reduction::Sum => Ok(Tensor::scalar(total_loss)),
            Reduction::Mean => {
                let mean_loss = total_loss / num_samples;
                Ok(Tensor::scalar(mean_loss))
            }
        }
    }
}

impl<T: FloatDtype> Module<T> for MultiMarginLoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(crate::NNError::InvalidInput {
            message: "MultiMarginLoss requires two inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

/// Multi-Label Margin Loss
///
/// Computes multi-label margin loss for multi-label classification:
/// `MultiLabelMarginLoss(x, y) = sum_{i} max(0, 1 - (x[y[i]] - x[y[j]])) for j≠i`
///
/// Where y is a list of target class indices for each sample.
/// This loss is used when each sample can belong to multiple classes simultaneously.
///
/// # Mathematical Properties
/// - Designed for multi-label classification
/// - Handles multiple correct labels per sample
/// - Margin-based loss encouraging separation
/// - Supports variable number of labels per sample
///
/// # Example
/// ```rust
/// use coeus_nn::MultiLabelMarginLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = MultiLabelMarginLoss::new();
/// let input = Tensor::from_vec(CpuBackend::default(), vec![0.2], vec![1]).unwrap();
/// // Multi-label targets: sample 0 has labels [0, 2], sample 1 has label [1]
/// let target = Tensor::from_vec(CpuBackend::default(), vec![0], vec![1]).unwrap();
///
/// let loss = loss_fn.forward(&input, &target).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MultiLabelMarginLoss {
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl Default for MultiLabelMarginLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiLabelMarginLoss {
    /// Create a new multi-label margin loss function
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Create with specified reduction
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute multi-label margin loss
    pub fn forward<T: FloatDtype, I: Dtype + num_traits::ToPrimitive + PartialOrd>(
        &self,
        input: &Tensor<T, CpuBackend>,
        target: &Tensor<I, CpuBackend>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        if input.ndim() != 2 {
            return Err(crate::NNError::InvalidInput {
                message: "Input must be 2D tensor (batch_size, num_classes)".to_string(),
            });
        }

        if target.ndim() != 2 {
            return Err(crate::NNError::InvalidInput {
                message: "Target must be 2D tensor (batch_size, num_classes)".to_string(),
            });
        }

        if input.shape() != target.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: input.shape().to_vec(),
                actual: target.shape().to_vec(),
            });
        }

        let batch_size = input.shape()[0];
        let num_classes = input.shape()[1];

        let mut total_loss = T::zero();
        let input_data = input.data();
        let target_data = target.data();

        for batch_idx in 0..batch_size {
            let mut sample_loss = T::zero();
            let mut num_positive_labels = 0;

            // Find all positive labels for this sample
            let mut positive_indices = Vec::new();
            for class_idx in 0..num_classes {
                let target_val = target_data[batch_idx * num_classes + class_idx];
                if Dtype::to_f64(&target_val).unwrap_or(-1.0) >= 0.0 {
                    positive_indices.push(class_idx);
                    num_positive_labels += 1;
                }
            }

            if num_positive_labels == 0 {
                continue; // No positive labels for this sample
            }

            // Compute loss for all pairs of positive and negative labels
            for &pos_idx in &positive_indices {
                let pos_score = input_data[batch_idx * num_classes + pos_idx];

                for neg_idx in 0..num_classes {
                    let target_val = target_data[batch_idx * num_classes + neg_idx];
                    if Dtype::to_f64(&target_val).unwrap_or(-1.0) >= 0.0 {
                        continue; // Skip if this is also a positive label
                    }

                    let neg_score = input_data[batch_idx * num_classes + neg_idx];
                    let diff = T::one() - (pos_score - neg_score);
                    sample_loss = sample_loss + diff.max(T::zero());
                }
            }

            total_loss = total_loss + sample_loss;
        }

        // Apply reduction
        let num_samples = T::from(batch_size as f64).unwrap();

        match self.reduction {
            Reduction::None => {
                // Return per-sample losses
                let mut losses = Vec::with_capacity(batch_size);
                for batch_idx in 0..batch_size {
                    let mut sample_loss = T::zero();

                    let mut positive_indices = Vec::new();
                    for class_idx in 0..num_classes {
                        let target_val = target_data[batch_idx * num_classes + class_idx];
                        if Dtype::to_f64(&target_val).unwrap_or(-1.0) >= 0.0 {
                            positive_indices.push(class_idx);
                        }
                    }

                    if positive_indices.is_empty() {
                        losses.push(T::zero());
                        continue;
                    }

                    for &pos_idx in &positive_indices {
                        let pos_score = input_data[batch_idx * num_classes + pos_idx];

                        for neg_idx in 0..num_classes {
                            let target_val = target_data[batch_idx * num_classes + neg_idx];
                            if Dtype::to_f64(&target_val).unwrap_or(-1.0) >= 0.0 {
                                continue;
                            }

                            let neg_score = input_data[batch_idx * num_classes + neg_idx];
                            let diff = T::one() - (pos_score - neg_score);
                            sample_loss = sample_loss + diff.max(T::zero());
                        }
                    }
                    losses.push(sample_loss);
                }
                // Per-sample return for MultiLabelMarginLoss
                let per_sample = Tensor::from_vec(CpuBackend::new(), losses, vec![batch_size]).unwrap();
                Ok(per_sample)
            }
            Reduction::Sum => Ok(Tensor::scalar(total_loss)),
            Reduction::Mean => {
                let mean_loss = total_loss / num_samples;
                Ok(Tensor::scalar(mean_loss))
            }
        }
    }
}

impl<T: FloatDtype> Module<T> for MultiLabelMarginLoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(crate::NNError::InvalidInput {
            message: "MultiLabelMarginLoss requires two inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use crate::validation::{numerical_gradient, validate_gradient_accuracy};
    use proptest::prelude::*;

    #[test]
    fn test_margin_ranking_loss_basic() {
        let loss_fn = MarginRankingLoss::new(1.0);
        let input1 = Tensor::from_vec(CpuBackend::new(), vec![1.0], vec![1]).unwrap();
        let input2 = Tensor::from_vec(CpuBackend::new(), vec![0.5], vec![1]).unwrap();
        let target = Tensor::from_vec(CpuBackend::new(), vec![1.0], vec![1]).unwrap();

        let loss = loss_fn.forward(&input1, &input2, &target).unwrap();

        // Expected: max(0, -1*(1-0.5)+1) = max(0, -0.5+1) = 0.5
        assert_relative_eq!(loss.item().unwrap(), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_margin_ranking_loss_zero_margin() {
        let loss_fn = MarginRankingLoss::new(0.0);
        let input1 = Tensor::from_vec(CpuBackend::new(), vec![2.0], vec![1]).unwrap();
        let input2 = Tensor::from_vec(CpuBackend::new(), vec![1.0], vec![1]).unwrap();
        let target = Tensor::from_vec(CpuBackend::new(), vec![1.0], vec![1]).unwrap(); // input1 should rank higher

        let loss = loss_fn.forward(&input1, &input2, &target).unwrap();

        // Expected: max(0, -1*(2-1)+0) = max(0, -1) = 0
        assert_relative_eq!(loss.item().unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hinge_embedding_loss_basic() {
        let loss_fn = HingeEmbeddingLoss::new(1.0);
        let input = Tensor::from_vec(CpuBackend::new(), vec![0.5], vec![1]).unwrap();
        let target = Tensor::from_vec(CpuBackend::new(), vec![1.0], vec![1]).unwrap();

        let loss = loss_fn.forward(&input, &target).unwrap();

        // For similar pair (y=1) loss = x = 0.5
        assert_relative_eq!(loss.item().unwrap(), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_hinge_embedding_loss_similar_pairs() {
        let loss_fn = HingeEmbeddingLoss::new(1.0);
        let input = Tensor::from_vec(CpuBackend::new(), vec![0.3], vec![1]).unwrap();
        let target = Tensor::from_vec(CpuBackend::new(), vec![1.0], vec![1]).unwrap(); // Both similar

        let loss = loss_fn.forward(&input, &target).unwrap();

        // Expected: loss = x = 0.3
        assert_relative_eq!(loss.item().unwrap(), 0.3, epsilon = 1e-6);
    }

    #[test]
    fn test_triplet_margin_loss_basic() {
        let loss_fn = TripletMarginLoss::new();

        // Simple 1D case
        let anchor = Tensor::from_vec(CpuBackend::new(), vec![0.0], vec![1]).unwrap();
        let positive = Tensor::from_vec(CpuBackend::new(), vec![1.0], vec![1]).unwrap(); // Distance = 1.0
        let negative = Tensor::from_vec(CpuBackend::new(), vec![3.0], vec![1]).unwrap(); // Distance = 3.0

        let loss = loss_fn.forward(&anchor, &positive, &negative).unwrap();

        // Expected: max(0, 1.0 - 3.0 + 1.0) = max(0, -1.0) = 0.0
        assert_relative_eq!(loss.item().unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_triplet_margin_loss_violation() {
        let loss_fn = TripletMarginLoss::new();

        // Case where positive is farther than negative (violation)
        let anchor = Tensor::from_vec(CpuBackend::new(), vec![0.0], vec![1]).unwrap();
        let positive = Tensor::from_vec(CpuBackend::new(), vec![3.0], vec![1]).unwrap(); // Distance = 3.0
        let negative = Tensor::from_vec(CpuBackend::new(), vec![1.0], vec![1]).unwrap(); // Distance = 1.0

        let loss = loss_fn.forward(&anchor, &positive, &negative).unwrap();

        // Expected: max(0, 3.0 - 1.0 + 1.0) = max(0, 3.0) = 3.0
        assert_relative_eq!(loss.item().unwrap(), 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_ranking_losses_reductions() {
        let input1 = Tensor::from_vec(CpuBackend::new(), vec![1.0, 2.0], vec![2]).unwrap();
        let input2 = Tensor::from_vec(CpuBackend::new(), vec![0.0, 1.5], vec![2]).unwrap();
        let target = Tensor::from_vec(CpuBackend::new(), vec![1.0, -1.0], vec![2]).unwrap();

        // Test different reductions for MarginRankingLoss
        let loss_none = MarginRankingLoss::with_params(0.0, Reduction::None);
        let loss_sum = MarginRankingLoss::with_params(0.0, Reduction::Sum);
        let loss_mean = MarginRankingLoss::with_params(0.0, Reduction::Mean);

        let result_none = loss_none.forward(&input1, &input2, &target).unwrap();
        let result_sum = loss_sum.forward(&input1, &input2, &target).unwrap();
        let result_mean = loss_mean.forward(&input1, &input2, &target).unwrap();

        // None should return per-element losses
        assert_eq!(result_none.shape(), &[2]);

        // Sum should be sum of individual losses
        let expected_sum = result_none.data()[0] + result_none.data()[1];
        assert_relative_eq!(result_sum.item().unwrap(), expected_sum, epsilon = 1e-6);

        // Mean should be average of individual losses
        let expected_mean = expected_sum / 2.0;
        assert_relative_eq!(result_mean.item().unwrap(), expected_mean, epsilon = 1e-6);
    }

    #[test]
    fn test_multi_margin_loss_basic() {
        let loss_fn = MultiMarginLoss::new(1.0, 1);
        // Batch size 1, 2 classes
        let input = Tensor::from_vec(CpuBackend::new(), vec![0.2, -0.1], vec![1, 2]).unwrap();
        let target = Tensor::from_vec(CpuBackend::default(), vec![0i32], vec![1]).unwrap();

        let loss = loss_fn.forward(&input, &target).unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_multi_margin_loss_perfect_classification() {
        let loss_fn = MultiMarginLoss::new(1.0, 1);
        // Perfect classification: correct class has much higher score
        let input = Tensor::from_vec(CpuBackend::new(), vec![10.0, 0.0], vec![1, 2]).unwrap();
        let target = Tensor::from_vec(CpuBackend::default(), vec![0i32], vec![1]).unwrap();

        let loss = loss_fn.forward(&input, &target).unwrap();
        // Should be very small since all margins are satisfied
        assert!(loss.item().unwrap() < 1e-6);
    }

    #[test]
    fn test_multi_margin_loss_violation() {
        let loss_fn = MultiMarginLoss::new(1.0, 1);
        // Violation: incorrect class has higher score than correct class
        let input = Tensor::from_vec(CpuBackend::new(), vec![0.0, 2.0], vec![1, 2]).unwrap();
        let target = Tensor::from_vec(CpuBackend::default(), vec![0i32], vec![1]).unwrap();

        let loss = loss_fn.forward(&input, &target).unwrap();
        // Should have positive loss due to margin violations
        assert!(loss.item().unwrap() > 0.0);
    }

    #[test]
    fn test_multi_margin_loss_reductions() {
        // Batch size 2, 2 classes
        let input = Tensor::from_vec(CpuBackend::default(), vec![0.2, -0.1, 0.0, 0.5], vec![2, 2]).unwrap();
        let target = Tensor::from_vec(CpuBackend::default(), vec![0i32, 1i32], vec![2]).unwrap();

        let loss_none = MultiMarginLoss::with_params(1.0, 1, Reduction::None);
        let loss_sum = MultiMarginLoss::with_params(1.0, 1, Reduction::Sum);
        let loss_mean = MultiMarginLoss::with_params(1.0, 1, Reduction::Mean);

        let result_none = loss_none.forward(&input, &target).unwrap();
        let result_sum = loss_sum.forward(&input, &target).unwrap();
        let result_mean = loss_mean.forward(&input, &target).unwrap();

        // None should return per-sample losses
        assert_eq!(result_none.shape(), &[2]);

        // Sum should be sum of individual losses
        let expected_sum = result_none.data().iter().copied().sum::<f64>();
        assert_relative_eq!(result_sum.item().unwrap(), expected_sum, epsilon = 1e-6);

        // Mean should be average of individual losses
        let expected_mean = expected_sum / 2.0;
        assert_relative_eq!(result_mean.item().unwrap(), expected_mean, epsilon = 1e-6);
    }

    #[test]
    fn test_multi_label_margin_loss_basic() {
        let loss_fn = MultiLabelMarginLoss::new();
        // Batch size 1, 2 classes
        let input = Tensor::from_vec(CpuBackend::new(), vec![0.2, 0.1], vec![1, 2]).unwrap();
        // Multi-label targets: sample 0 has label 0 positive, class 1 negative
        let target = Tensor::from_vec(CpuBackend::default(), vec![0i32, -1i32], vec![1, 2]).unwrap();

        let loss = loss_fn.forward(&input, &target).unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_multi_label_margin_loss_single_label() {
        let loss_fn = MultiLabelMarginLoss::new();
        let input = Tensor::from_vec(CpuBackend::new(), vec![0.2, 0.0], vec![1, 2]).unwrap();
        // Single label per sample: class 0 positive
        let target = Tensor::from_vec(CpuBackend::default(), vec![0i32, -1i32], vec![1, 2]).unwrap();

        let loss = loss_fn.forward(&input, &target).unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_multi_label_margin_loss_reductions() {
        // Batch size 2, 2 classes
        let input = Tensor::from_vec(CpuBackend::new(), vec![0.2, 0.0, 0.1, -0.1], vec![2, 2]).unwrap();
        let target = Tensor::from_vec(CpuBackend::default(), vec![0i32, -1i32, 0i32, 1i32], vec![2, 2]).unwrap();

        let loss_none = MultiLabelMarginLoss::with_reduction(Reduction::None);
        let loss_sum = MultiLabelMarginLoss::with_reduction(Reduction::Sum);
        let loss_mean = MultiLabelMarginLoss::with_reduction(Reduction::Mean);

        let result_none = loss_none.forward(&input, &target).unwrap();
        let result_sum = loss_sum.forward(&input, &target).unwrap();
        let result_mean = loss_mean.forward(&input, &target).unwrap();

        // None should return per-sample losses
        assert_eq!(result_none.shape(), &[2]);

        // Sum should be sum of individual losses
        let expected_sum = result_none.data().iter().copied().sum::<f64>();
        assert_relative_eq!(result_sum.item().unwrap(), expected_sum, epsilon = 1e-6);

        // Mean should be average of individual losses
        let expected_mean = expected_sum / 2.0;
        assert_relative_eq!(result_mean.item().unwrap(), expected_mean, epsilon = 1e-6);
    }

    #[test]
    fn test_margin_ranking_loss_gradient_validation() {
        // Validate analytical gradients against numerical finite differences for input1 and input2
        let loss_fn = MarginRankingLoss::new(0.5);
        let input1 = Tensor::from_vec(CpuBackend::new(), vec![0.2, -0.5, 1.0], vec![3]).unwrap();
        let input2 = Tensor::from_vec(CpuBackend::new(), vec![0.0, 0.3, 0.8], vec![3]).unwrap();
        let target = Tensor::from_vec(CpuBackend::new(), vec![1.0, -1.0, 1.0], vec![3]).unwrap();

        // Analytical gradients (both inputs)
        let (analytical_g1, analytical_g2) = loss_fn.backward(&input1, &input2, &target).unwrap();

        // Numerical gradient w.r.t input1
        let input2_clone = input2.clone();
        let target_clone = target.clone();
        let numerical_g1 = numerical_gradient(
            move |x: &Tensor<f64, CpuBackend>| -> Result<Tensor<f64, CpuBackend>, Box<dyn std::error::Error>> { Ok(loss_fn.forward(x, &input2_clone, &target_clone).unwrap()) },
            &input1,
            1e-6,
        ).unwrap();

        assert!(validate_gradient_accuracy(&analytical_g1, &numerical_g1, 1e-5));

        // Numerical gradient w.r.t input2
        let input1_clone = input1.clone();
        let target_clone2 = target.clone();
        let numerical_g2 = numerical_gradient(
            move |x: &Tensor<f64, CpuBackend>| -> Result<Tensor<f64, CpuBackend>, Box<dyn std::error::Error>> { Ok(loss_fn.forward(&input1_clone, x, &target_clone2).unwrap()) },
            &input2,
            1e-6,
        ).unwrap();

        assert!(validate_gradient_accuracy(&analytical_g2, &numerical_g2, 1e-5));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]
        fn prop_margin_ranking_random(
            x1 in proptest::collection::vec(-100.0f64..100.0, 1..8),
            x2 in proptest::collection::vec(-100.0f64..100.0, 1..8),
            y_vals in proptest::collection::vec(prop_oneof![Just(1.0f64), Just(-1.0f64)], 1..8),
            margin in -10.0f64..10.0,
        ) {
            prop_assume!(x1.len() == x2.len() && x1.len() == y_vals.len());

            let len = x1.len();
            let input1 = Tensor::from_vec(CpuBackend::default(), x1.clone(), vec![len]).unwrap();
            let input2 = Tensor::from_vec(CpuBackend::default(), x2.clone(), vec![len]).unwrap();
            let target = Tensor::from_vec(CpuBackend::default(), y_vals.clone(), vec![len]).unwrap();

            let loss_fn = MarginRankingLoss::with_params(margin, Reduction::Mean);
            let result = loss_fn.forward(&input1, &input2, &target).unwrap();

            // Compute expected per-element losses in plain f64
            let expected: Vec<f64> = x1.iter().zip(x2.iter()).zip(y_vals.iter())
                .map(|((&a, &b), &y)| {
                    let diff = a - b;
                    let val = (-y * diff + margin).max(0.0);
                    val
                })
                .collect();

            // Verify mean reduction matches
            let expected_mean = expected.iter().copied().sum::<f64>() / (expected.len() as f64);
            assert_relative_eq!(result.item().unwrap(), expected_mean, epsilon = 1e-6);

            // Also verify Reduction::None returns per-element vector
            let loss_none = MarginRankingLoss::with_params(margin, Reduction::None);
            let result_none = loss_none.forward(&input1, &input2, &target).unwrap();
            assert_eq!(result_none.shape(), &[len]);
            for i in 0..len {
                assert_relative_eq!(result_none.data()[i] as f64, expected[i], epsilon = 1e-6);
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]
        fn prop_hinge_embedding_random(
            xs in proptest::collection::vec(-100.0f64..100.0, 1..8),
            ys in proptest::collection::vec(prop_oneof![Just(1.0f64), Just(-1.0f64)], 1..8),
            margin in -10.0f64..10.0,
        ) {
            prop_assume!(xs.len() == ys.len());
            let len = xs.len();

            let input = Tensor::from_vec(CpuBackend::default(), xs.clone(), vec![len]).unwrap();
            let target = Tensor::from_vec(CpuBackend::default(), ys.clone(), vec![len]).unwrap();

            let loss_fn = HingeEmbeddingLoss::with_params(margin, Reduction::Mean);
            let result = loss_fn.forward(&input, &target).unwrap();

            let expected: Vec<f64> = xs.iter().zip(ys.iter()).map(|(&x, &y)| {
                if y == 1.0 {
                    x
                } else {
                    (margin - x).max(0.0)
                }
            }).collect();

            let expected_mean = expected.iter().copied().sum::<f64>() / (expected.len() as f64);
            assert_relative_eq!(result.item().unwrap(), expected_mean, epsilon = 1e-6);
        }
    }
}


