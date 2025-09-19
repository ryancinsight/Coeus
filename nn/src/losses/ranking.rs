//! Ranking loss functions
//!
//! Loss functions for learning relative ordering and ranking tasks,
//! commonly used in information retrieval, recommendation systems,
//! and metric learning applications.

use super::{utils, Module, Reduction};
use coeus_tensor::{Dtype, FloatDtype, Tensor};

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
/// let input1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
/// let input2 = Tensor::from_vec(vec![0.5, 1.5], vec![2]);
/// let target = Tensor::from_vec(vec![1.0, -1.0], vec![2]);
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
    pub fn forward<T: FloatDtype>(
        &self,
        input1: &Tensor<T>,
        input2: &Tensor<T>,
        target: &Tensor<T>,
    ) -> crate::Result<Tensor<T>> {
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

        let loss_tensor = Tensor::from_vec(losses, input1.shape().to_vec());
        utils::apply_reduction(&loss_tensor, self.reduction)
    }
}

impl<T: FloatDtype> Module<T> for MarginRankingLoss {
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Err(crate::NNError::InvalidInput {
            message: "MarginRankingLoss requires three inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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
    pub fn forward<T: FloatDtype>(
        &self,
        input: &Tensor<T>,
        target: &Tensor<T>,
    ) -> crate::Result<Tensor<T>> {
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

        let loss_tensor = Tensor::from_vec(losses, input.shape().to_vec());
        utils::apply_reduction(&loss_tensor, self.reduction)
    }
}

impl<T: FloatDtype> Module<T> for HingeEmbeddingLoss {
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Err(crate::NNError::InvalidInput {
            message: "HingeEmbeddingLoss requires two inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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
    pub fn forward<T: FloatDtype>(
        &self,
        anchor: &Tensor<T>,
        positive: &Tensor<T>,
        negative: &Tensor<T>,
    ) -> crate::Result<Tensor<T>> {
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

        let loss_tensor = Tensor::from_vec(losses, vec![batch_size]);
        utils::apply_reduction(&loss_tensor, self.reduction)
    }
}

impl<T: FloatDtype> Module<T> for TripletMarginLoss {
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Err(crate::NNError::InvalidInput {
            message: "TripletMarginLoss requires three inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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
/// let input = Tensor::from_vec(vec![0.2, 0.8, 0.1, 0.9, 0.3, 0.7], vec![2, 3]);
/// let target = Tensor::from_vec(vec![1, 2], vec![2]); // Class indices
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
        input: &Tensor<T>,
        target: &Tensor<I>,
    ) -> crate::Result<Tensor<T>> {
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
                Ok(Tensor::from_vec(losses, vec![batch_size]))
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
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Err(crate::NNError::InvalidInput {
            message: "MultiMarginLoss requires two inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
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
/// let input = Tensor::from_vec(vec![0.2, 0.8, 0.1, 0.9, 0.3, 0.7], vec![2, 3]);
/// // Multi-label targets: sample 0 has labels [0, 2], sample 1 has label [1]
/// let target = Tensor::from_vec(vec![0, -1, 2, 1, -1, -1], vec![2, 3]);
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
        input: &Tensor<T>,
        target: &Tensor<I>,
    ) -> crate::Result<Tensor<T>> {
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
                Ok(Tensor::from_vec(losses, vec![batch_size]))
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
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Err(crate::NNError::InvalidInput {
            message: "MultiLabelMarginLoss requires two inputs via forward() method".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_margin_ranking_loss_basic() {
        let loss_fn = MarginRankingLoss::new(1.0);
        let input1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let input2 = Tensor::from_vec(vec![0.5, 1.5], vec![2]);
        let target = Tensor::from_vec(vec![1.0, -1.0], vec![2]);

        let loss = loss_fn.forward(&input1, &input2, &target).unwrap();

        // Expected: (max(0, -1*(1-0.5)+1) + max(0, -(-1)*(2-1.5)+1)) / 2
        //         = (max(0, -0.5+1) + max(0, 0.5+1)) / 2
        //         = (0.5 + 1.5) / 2 = 1.0
        assert_relative_eq!(loss.item().unwrap(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_margin_ranking_loss_zero_margin() {
        let loss_fn = MarginRankingLoss::new(0.0);
        let input1 = Tensor::from_vec(vec![2.0], vec![1]);
        let input2 = Tensor::from_vec(vec![1.0], vec![1]);
        let target = Tensor::from_vec(vec![1.0], vec![1]); // input1 should rank higher

        let loss = loss_fn.forward(&input1, &input2, &target).unwrap();

        // Expected: max(0, -1*(2-1)+0) = max(0, -1) = 0
        assert_relative_eq!(loss.item().unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hinge_embedding_loss_basic() {
        let loss_fn = HingeEmbeddingLoss::new(1.0);
        let input = Tensor::from_vec(vec![0.5, 1.5, 0.2], vec![3]);
        let target = Tensor::from_vec(vec![1.0, -1.0, 1.0], vec![3]);

        let loss = loss_fn.forward(&input, &target).unwrap();

        // Expected: (0.5 + max(0, 1-1.5) + 0.2) / 3 = (0.5 + 0 + 0.2) / 3 = 0.7/3
        assert_relative_eq!(loss.item().unwrap(), 0.7 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hinge_embedding_loss_similar_pairs() {
        let loss_fn = HingeEmbeddingLoss::new(1.0);
        let input = Tensor::from_vec(vec![0.3, 0.8], vec![2]);
        let target = Tensor::from_vec(vec![1.0, 1.0], vec![2]); // Both similar

        let loss = loss_fn.forward(&input, &target).unwrap();

        // Expected: (0.3 + 0.8) / 2 = 0.55
        assert_relative_eq!(loss.item().unwrap(), 0.55, epsilon = 1e-6);
    }

    #[test]
    fn test_triplet_margin_loss_basic() {
        let loss_fn = TripletMarginLoss::new();

        // Simple 1D case
        let anchor = Tensor::from_vec(vec![0.0], vec![1, 1]);
        let positive = Tensor::from_vec(vec![1.0], vec![1, 1]); // Distance = 1.0
        let negative = Tensor::from_vec(vec![3.0], vec![1, 1]); // Distance = 3.0

        let loss = loss_fn.forward(&anchor, &positive, &negative).unwrap();

        // Expected: max(0, 1.0 - 3.0 + 1.0) = max(0, -1.0) = 0.0
        assert_relative_eq!(loss.item().unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_triplet_margin_loss_violation() {
        let loss_fn = TripletMarginLoss::new();

        // Case where positive is farther than negative (violation)
        let anchor = Tensor::from_vec(vec![0.0], vec![1, 1]);
        let positive = Tensor::from_vec(vec![3.0], vec![1, 1]); // Distance = 3.0
        let negative = Tensor::from_vec(vec![1.0], vec![1, 1]); // Distance = 1.0

        let loss = loss_fn.forward(&anchor, &positive, &negative).unwrap();

        // Expected: max(0, 3.0 - 1.0 + 1.0) = max(0, 3.0) = 3.0
        assert_relative_eq!(loss.item().unwrap(), 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_ranking_losses_reductions() {
        let input1 = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let input2 = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let target = Tensor::from_vec(vec![1.0, 1.0], vec![2]);

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
        let input = Tensor::from_vec(vec![0.2, 0.8, 0.1, 0.9, 0.3, 0.7], vec![2, 3]);
        let target = Tensor::from_vec(vec![1, 2], vec![2]);

        let loss = loss_fn.forward(&input, &target).unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_multi_margin_loss_perfect_classification() {
        let loss_fn = MultiMarginLoss::new(1.0, 1);
        // Perfect classification: correct class has much higher score
        let input = Tensor::from_vec(vec![0.0, 2.0, 0.0, 0.0, 0.0, 2.0], vec![2, 3]);
        let target = Tensor::from_vec(vec![1, 2], vec![2]);

        let loss = loss_fn.forward(&input, &target).unwrap();
        // Should be very small since all margins are satisfied
        assert!(loss.item().unwrap() < 0.1);
    }

    #[test]
    fn test_multi_margin_loss_violation() {
        let loss_fn = MultiMarginLoss::new(1.0, 1);
        // Violation: incorrect class has higher score than correct class
        let input = Tensor::from_vec(vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0], vec![2, 3]);
        let target = Tensor::from_vec(vec![1, 2], vec![2]);

        let loss = loss_fn.forward(&input, &target).unwrap();
        // Should have positive loss due to margin violations
        assert!(loss.item().unwrap() > 0.0);
    }

    #[test]
    fn test_multi_margin_loss_reductions() {
        let input = Tensor::from_vec(vec![0.2, 0.8, 0.1, 0.9, 0.3, 0.7], vec![2, 3]);
        let target = Tensor::from_vec(vec![1, 2], vec![2]);

        let loss_none = MultiMarginLoss::with_params(1.0, 1, Reduction::None);
        let loss_sum = MultiMarginLoss::with_params(1.0, 1, Reduction::Sum);
        let loss_mean = MultiMarginLoss::with_params(1.0, 1, Reduction::Mean);

        let result_none = loss_none.forward(&input, &target).unwrap();
        let result_sum = loss_sum.forward(&input, &target).unwrap();
        let result_mean = loss_mean.forward(&input, &target).unwrap();

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
    fn test_multi_label_margin_loss_basic() {
        let loss_fn = MultiLabelMarginLoss::new();
        let input = Tensor::from_vec(vec![0.2, 0.8, 0.1, 0.9, 0.3, 0.7], vec![2, 3]);
        // Multi-label targets: sample 0 has labels [0, 2], sample 1 has label [1]
        let target = Tensor::from_vec(vec![0, -1, 2, 1, -1, -1], vec![2, 3]);

        let loss = loss_fn.forward(&input, &target).unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_multi_label_margin_loss_single_label() {
        let loss_fn = MultiLabelMarginLoss::new();
        let input = Tensor::from_vec(vec![0.2, 0.8, 0.1, 0.9, 0.3, 0.7], vec![2, 3]);
        // Single label per sample
        let target = Tensor::from_vec(vec![1, -1, -1, -1, 2, -1], vec![2, 3]);

        let loss = loss_fn.forward(&input, &target).unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_multi_label_margin_loss_no_positive_labels() {
        let loss_fn = MultiLabelMarginLoss::new();
        let input = Tensor::from_vec(vec![0.2, 0.8, 0.1, 0.9, 0.3, 0.7], vec![2, 3]);
        // No positive labels for first sample, one positive label for second
        let target = Tensor::from_vec(vec![-1, -1, -1, 1, -1, -1], vec![2, 3]);

        let loss = loss_fn.forward(&input, &target).unwrap();
        // Loss should be finite and reasonable
        let loss_val: f64 = loss.item().unwrap();
        assert!(loss_val.is_finite());
    }

    #[test]
    fn test_multi_label_margin_loss_reductions() {
        let input = Tensor::from_vec(vec![0.2, 0.8, 0.1, 0.9, 0.3, 0.7], vec![2, 3]);
        let target = Tensor::from_vec(vec![0, -1, 2, 1, -1, -1], vec![2, 3]);

        let loss_none = MultiLabelMarginLoss::with_reduction(Reduction::None);
        let loss_sum = MultiLabelMarginLoss::with_reduction(Reduction::Sum);
        let loss_mean = MultiLabelMarginLoss::with_reduction(Reduction::Mean);

        let result_none = loss_none.forward(&input, &target).unwrap();
        let result_sum = loss_sum.forward(&input, &target).unwrap();
        let result_mean = loss_mean.forward(&input, &target).unwrap();

        // None should return per-element losses
        assert_eq!(result_none.shape(), &[2]);

        // Sum should be sum of individual losses
        let expected_sum = result_none.data()[0] + result_none.data()[1];
        assert_relative_eq!(result_sum.item().unwrap(), expected_sum, epsilon = 1e-6);

        // Mean should be average of individual losses
        let expected_mean = expected_sum / 2.0;
        assert_relative_eq!(result_mean.item().unwrap(), expected_mean, epsilon = 1e-6);
    }
}
