//! Regression loss functions
//!
//! Loss functions for continuous value prediction tasks, including
//! standard regression losses and their robust variants.

use super::{utils, Module, NNError, Reduction, Result};
use coeus_tensor::{FloatDtype, Tensor, CpuBackend};

/// Mean Squared Error (MSE) loss function
///
/// Computes the mean squared error between predictions and targets:
/// `MSE(y, ŷ) = (1/n) * Σ(yᵢ - ŷᵢ)²`
///
/// This is the most common loss function for regression tasks.
///
/// # Mathematical Properties
/// - Convex and differentiable everywhere
/// - Sensitive to outliers (quadratic penalty)
/// - Gradient: `∂MSE/∂ŷ = (2/n) * (ŷ - y)`
///
/// # Example
/// ```rust
/// use coeus_nn::MseLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = MseLoss::new();
/// let predictions = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
/// let targets = Tensor::from_vec(CpuBackend::default(), vec![1.1], vec![1]).unwrap();
///
/// let loss = loss_fn.forward(&predictions, &targets).unwrap();
/// assert!(loss.item().unwrap() > 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MseLoss {
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl MseLoss {
    /// Create a new MSE loss with mean reduction
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Create a new MSE loss with specified reduction
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute MSE loss between predictions and targets
    pub fn forward<T: FloatDtype + std::iter::Sum>(
        &self,
        predictions: &Tensor<T, CpuBackend>,
        targets: &Tensor<T, CpuBackend>,
    ) -> Result<Tensor<T, CpuBackend>> {
        if predictions.shape() != targets.shape() {
            return Err(NNError::ShapeMismatch {
                expected: predictions.shape().to_vec(),
                actual: targets.shape().to_vec(),
            });
        }

        // Compute element-wise squared differences
        let diff = (predictions - targets).map_err(|_| NNError::ForwardError {
            message: "Failed to compute prediction-target difference".to_string(),
        })?;

        let squared_diff = (&diff * &diff).map_err(|_| NNError::ForwardError {
            message: "Failed to compute squared differences".to_string(),
        })?;

        // Apply reduction
        utils::apply_reduction(&squared_diff, self.reduction)
    }

    /// Compute gradients of MSE loss with respect to predictions
    ///
    /// # Mathematical Derivation
    /// For MSE loss `L = (1/n) * Σ(ŷᵢ - yᵢ)²`:
    /// `∂L/∂ŷᵢ = (2/n) * (ŷᵢ - yᵢ)`
    pub fn backward<T: FloatDtype>(
        &self,
        predictions: &Tensor<T, CpuBackend>,
        targets: &Tensor<T, CpuBackend>,
    ) -> Result<Tensor<T, CpuBackend>> {
        if predictions.shape() != targets.shape() {
            return Err(NNError::ShapeMismatch {
                expected: predictions.shape().to_vec(),
                actual: targets.shape().to_vec(),
            });
        }

        // Gradient: ∂MSE/∂ŷ = (2/n) * (ŷ - y)
        let diff = (predictions - targets).map_err(|_| NNError::BackwardError {
            message: "Failed to compute prediction-target difference".to_string(),
        })?;

        let scale = match self.reduction {
            Reduction::None => T::from(2.0).unwrap(),
            Reduction::Sum => T::from(2.0).unwrap(),
            Reduction::Mean => {
                let n = T::from(predictions.numel()).unwrap();
                T::from(2.0).unwrap() / n
            }
        };

        (&diff * &Tensor::scalar(scale)).map_err(Into::into)
    }
}

impl<T: FloatDtype> Module<T> for MseLoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(crate::NNError::InvalidInput {
            message: "MSELoss should be used via forward() method with two inputs".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

/// Mean Absolute Error (MAE) loss function
///
/// Computes the mean absolute error between predictions and targets:
/// `MAE(y, ŷ) = (1/n) * Σ|yᵢ - ŷᵢ|`
///
/// Also known as L1 loss, this is more robust to outliers than MSE.
///
/// # Mathematical Properties
/// - Convex but not differentiable at zero
/// - Less sensitive to outliers than MSE (linear penalty)
/// - Gradient: `∂MAE/∂ŷ = (1/n) * sign(ŷ - y)`
///
/// # Example
/// ```rust
/// use coeus_nn::MaeLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = MaeLoss::new();
/// let predictions = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
/// let targets = Tensor::from_vec(CpuBackend::default(), vec![1.1], vec![1]).unwrap();
///
/// let loss = loss_fn.forward(&predictions, &targets).unwrap();
/// assert!(loss.item().unwrap() > 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct MaeLoss {
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl MaeLoss {
    /// Create a new MAE loss with mean reduction
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Create a new MAE loss with specified reduction
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute MAE loss between predictions and targets
    pub fn forward<T: FloatDtype + num_traits::Signed + std::iter::Sum>(
        &self,
        predictions: &Tensor<T, CpuBackend>,
        targets: &Tensor<T, CpuBackend>,
    ) -> Result<Tensor<T, CpuBackend>> {
        if predictions.shape() != targets.shape() {
            return Err(NNError::ShapeMismatch {
                expected: predictions.shape().to_vec(),
                actual: targets.shape().to_vec(),
            });
        }

        // Compute element-wise absolute differences
        let diff = (predictions - targets).map_err(|_| NNError::ForwardError {
            message: "Failed to compute prediction-target difference".to_string(),
        })?;

        let abs_diff = coeus_tensor::ops::arithmetic::abs(&diff);

        // Apply reduction
        utils::apply_reduction(&abs_diff, self.reduction)
    }

    /// Compute gradients of MAE loss with respect to predictions
    ///
    /// # Mathematical Derivation
    /// For MAE loss `L = (1/n) * Σ|ŷᵢ - yᵢ|`:
    /// `∂L/∂ŷᵢ = (1/n) * sign(ŷᵢ - yᵢ)`
    pub fn backward<T: FloatDtype + num_traits::Signed>(
        &self,
        predictions: &Tensor<T, CpuBackend>,
        targets: &Tensor<T, CpuBackend>,
    ) -> Result<Tensor<T, CpuBackend>> {
        if predictions.shape() != targets.shape() {
            return Err(NNError::ShapeMismatch {
                expected: predictions.shape().to_vec(),
                actual: targets.shape().to_vec(),
            });
        }

        // Gradient: ∂MAE/∂ŷ = (1/n) * sign(ŷ - y)
        let diff = (predictions - targets).map_err(|_| NNError::BackwardError {
            message: "Failed to compute prediction-target difference".to_string(),
        })?;

        let sign_diff = diff.map(|x| {
            if x > T::zero() {
                T::one()
            } else if x < T::zero() {
                -T::one()
            } else {
                T::zero()
            }
        });

        let scale = match self.reduction {
            Reduction::None => T::one(),
            Reduction::Sum => T::one(),
            Reduction::Mean => {
                let n = T::from(predictions.numel()).unwrap();
                T::one() / n
            }
        };

        (&sign_diff * &Tensor::scalar(scale)).map_err(Into::into)
    }
}

impl<T: FloatDtype> Module<T> for MaeLoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(crate::NNError::InvalidInput {
            message: "MAELoss should be used via forward() method with two inputs".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T, CpuBackend>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T, CpuBackend>> {
        vec![]
    }
}

/// Cosine Embedding Loss
///
/// Computes cosine embedding loss for learning embeddings:
/// `CosineEmbeddingLoss(x1, x2, y) = {
///     1 - cos(x1, x2)                    if y = 1 (similar)
///     max(0, cos(x1, x2) - margin)       if y = -1 (dissimilar)
/// }`
///
/// Where cos(x1, x2) is the cosine similarity between the two vectors.
/// This loss encourages similar pairs to have high cosine similarity and
/// dissimilar pairs to have low cosine similarity (below the margin).
///
/// # Mathematical Properties
/// - Uses cosine similarity to measure embedding similarity
/// - Margin parameter controls the minimum separation for dissimilar pairs
/// - Range: [0, 1 + margin] where lower values indicate better embeddings
/// - Gradient: depends on whether pairs are similar or dissimilar
///
/// # Example
/// ```rust
/// use coeus_nn::CosineEmbeddingLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = CosineEmbeddingLoss::new(0.5);
/// let input1 = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap(); // Unit vector along x-axis
/// let input2 = Tensor::from_vec(CpuBackend::default(), vec![0.0], vec![1]).unwrap(); // Unit vector along y-axis
/// let target = Tensor::from_vec(CpuBackend::default(), vec![-1.0], vec![1]).unwrap(); // Dissimilar pair
///
/// let loss = loss_fn.forward(&input1, &input2, &target).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct CosineEmbeddingLoss {
    /// Margin parameter for dissimilar pairs (default: 0.0)
    pub margin: f64,
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl Default for CosineEmbeddingLoss {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl CosineEmbeddingLoss {
    /// Create a new cosine embedding loss function
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

    /// Compute cosine similarity between two tensors
    fn cosine_similarity<T: FloatDtype>(x1: &Tensor<T, CpuBackend>, x2: &Tensor<T, CpuBackend>) -> crate::Result<T> {
        if x1.shape() != x2.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: x1.shape().to_vec(),
                actual: x2.shape().to_vec(),
            });
        }

        let x1_data = x1.data();
        let x2_data = x2.data();

        // Compute dot product: sum(x1 * x2)
        let mut dot_product = T::zero();
        for (&a, &b) in x1_data.iter().zip(x2_data.iter()) {
            dot_product = dot_product + a * b;
        }

        // Compute L2 norms: sqrt(sum(x1²)) and sqrt(sum(x2²))
        let mut x1_norm_sq = T::zero();
        let mut x2_norm_sq = T::zero();

        for (&a, &b) in x1_data.iter().zip(x2_data.iter()) {
            x1_norm_sq = x1_norm_sq + a * a;
            x2_norm_sq = x2_norm_sq + b * b;
        }

        let x1_norm = x1_norm_sq.sqrt();
        let x2_norm = x2_norm_sq.sqrt();

        // Cosine similarity: dot(x1, x2) / (||x1|| * ||x2||)
        let norm_product = x1_norm * x2_norm;

        if norm_product == T::zero() {
            // Handle zero norm case (avoid division by zero)
            return Ok(T::zero());
        }

        Ok(dot_product / norm_product)
    }

    /// Compute cosine embedding loss
    pub fn forward<T: FloatDtype + std::iter::Sum>(
        &self,
        input1: &Tensor<T, CpuBackend>,
        input2: &Tensor<T, CpuBackend>,
        target: &Tensor<T, CpuBackend>,
    ) -> crate::Result<Tensor<T, CpuBackend>> {
        if input1.shape() != input2.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: input1.shape().to_vec(),
                actual: input2.shape().to_vec(),
            });
        }

        if input1.ndim() != 2 || target.ndim() != 1 {
            return Err(crate::NNError::InvalidInput {
                message: "Input1 and input2 must be 2D tensors, target must be 1D".to_string(),
            });
        }

        if input1.shape()[0] != target.shape()[0] {
            return Err(crate::NNError::ShapeMismatch {
                expected: vec![input1.shape()[0]],
                actual: target.shape().to_vec(),
            });
        }

        let batch_size = input1.shape()[0];
        let feature_size = input1.shape()[1];
        let margin = T::from(self.margin).unwrap();

        let mut losses = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            // Extract embeddings for this sample
            let start_idx = batch_idx * feature_size;
            let end_idx = (batch_idx + 1) * feature_size;

            let x1_slice = input1.data()[start_idx..end_idx].to_vec();
            let x2_slice = input2.data()[start_idx..end_idx].to_vec();

            let x1_sample = Tensor::from_vec(CpuBackend::default(), x1_slice, vec![feature_size]).unwrap();
            let x2_sample = Tensor::from_vec(CpuBackend::default(), x2_slice, vec![feature_size]).unwrap();

            // Compute cosine similarity
            let cos_sim = Self::cosine_similarity(&x1_sample, &x2_sample)?;

            let target_val = target.data()[batch_idx];

            // Compute loss based on target
            let loss_val = if target_val == T::one() {
                // Similar pair: loss = 1 - cos(x1, x2)
                T::one() - cos_sim
            } else {
                // Dissimilar pair: loss = max(0, cos(x1, x2) - margin)
                (cos_sim - margin).max(T::zero())
            };

            losses.push(loss_val);
        }

        let loss_tensor = Tensor::from_vec(CpuBackend::default(), losses, vec![batch_size]).unwrap();
        utils::apply_reduction(&loss_tensor, self.reduction)
    }
}

impl<T: FloatDtype> Module<T> for CosineEmbeddingLoss {
    fn forward(&self, _input: &Tensor<T, CpuBackend>) -> crate::Result<Tensor<T, CpuBackend>> {
        Err(crate::NNError::InvalidInput {
            message: "CosineEmbeddingLoss requires three inputs via forward() method".to_string(),
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

    #[test]
    fn test_mse_loss_basic() {
        let loss_fn = MseLoss::new();
        let pred = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
        let target = Tensor::from_vec(CpuBackend::default(), vec![1.1], vec![1]).unwrap();

        let loss = loss_fn.forward(&pred, &target).unwrap();
        assert!(loss.item().unwrap() > 0.0);

        // Test backward pass
        let grad = loss_fn.backward(&pred, &target).unwrap();
        assert_eq!(grad.shape(), pred.shape());
    }

    #[test]
    fn test_mse_loss_perfect_prediction() {
        let loss_fn = MseLoss::new();
        let pred = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
        let target = pred.clone();

        let loss = loss_fn.forward(&pred, &target).unwrap();
        assert_relative_eq!(loss.item().unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mse_loss_reductions() {
        let pred = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
        let target = Tensor::from_vec(CpuBackend::default(), vec![0.0], vec![1]).unwrap();

        // Test different reductions
        let loss_none = MseLoss::with_reduction(Reduction::None);
        let loss_sum = MseLoss::with_reduction(Reduction::Sum);
        let loss_mean = MseLoss::with_reduction(Reduction::Mean);

        let result_none = loss_none.forward(&pred, &target).unwrap();
        let result_sum = loss_sum.forward(&pred, &target).unwrap();
        let result_mean = loss_mean.forward(&pred, &target).unwrap();

        // None should return [1.0, 4.0]
        assert_eq!(result_none.shape(), &[2]);
        assert_relative_eq!(result_none.data()[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result_none.data()[1], 4.0, epsilon = 1e-6);

        // Sum should return 5.0
        assert_relative_eq!(result_sum.item().unwrap(), 5.0, epsilon = 1e-6);

        // Mean should return 2.5
        assert_relative_eq!(result_mean.item().unwrap(), 2.5, epsilon = 1e-6);
    }

    #[test]
    fn test_mae_loss_basic() {
        let loss_fn = MaeLoss::new();
        let pred = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
        let target = Tensor::from_vec(CpuBackend::default(), vec![1.1], vec![1]).unwrap();

        let loss = loss_fn.forward(&pred, &target).unwrap();
        assert!(loss.item().unwrap() > 0.0);

        // Test backward pass
        let grad = loss_fn.backward(&pred, &target).unwrap();
        assert_eq!(grad.shape(), pred.shape());
    }

    #[test]
    fn test_mae_loss_perfect_prediction() {
        let loss_fn = MaeLoss::new();
        let pred = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
        let target = pred.clone();

        let loss = loss_fn.forward(&pred, &target).unwrap();
        assert_relative_eq!(loss.item().unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mae_vs_mse_outlier_robustness() {
        // MAE should be less sensitive to outliers than MSE
        let pred = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap(); // Large outlier
        let target = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();

        let mse_loss = MseLoss::new();
        let mae_loss = MaeLoss::new();

        let mse_result = mse_loss.forward(&pred, &target).unwrap();
        let mae_result = mae_loss.forward(&pred, &target).unwrap();

        // MSE should be much larger due to quadratic penalty on outlier
        assert!(mse_result.item().unwrap() > mae_result.item().unwrap());
    }

    #[test]
    fn test_cosine_embedding_loss_similar_pairs() {
        let loss_fn = CosineEmbeddingLoss::new(0.5);
        // Two identical unit vectors (perfect similarity, cos = 1.0)
        let input1 = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
        let input2 = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
        let target = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap(); // Similar pair

        let loss = loss_fn.forward(&input1, &input2, &target).unwrap();

        // Expected: 1 - 1.0 = 0.0
        assert_relative_eq!(loss.item().unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_embedding_loss_dissimilar_pairs() {
        let loss_fn = CosineEmbeddingLoss::new(0.5);
        // Orthogonal unit vectors (cos = 0.0)
        let input1 = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
        let input2 = Tensor::from_vec(CpuBackend::default(), vec![0.0], vec![1]).unwrap();
        let target = Tensor::from_vec(CpuBackend::default(), vec![-1.0], vec![1]).unwrap(); // Dissimilar pair

        let loss = loss_fn.forward(&input1, &input2, &target).unwrap();

        // Expected: max(0, 0.0 - 0.5) = 0.0
        assert_relative_eq!(loss.item().unwrap(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_embedding_loss_dissimilar_violation() {
        let loss_fn = CosineEmbeddingLoss::new(0.5);
        // Vectors with high similarity (cos ≈ 0.8)
        let input1 = Tensor::from_vec(CpuBackend::default(), vec![2.0], vec![1]).unwrap();
        let input2 = Tensor::from_vec(CpuBackend::default(), vec![2.0], vec![1]).unwrap();
        let target = Tensor::from_vec(CpuBackend::default(), vec![-1.0], vec![1]).unwrap(); // Dissimilar pair

        let loss = loss_fn.forward(&input1, &input2, &target).unwrap();

        // Should have positive loss due to high similarity
        assert!(loss.item().unwrap() > 0.0);
    }

    #[test]
    fn test_cosine_embedding_loss_zero_vectors() {
        let loss_fn = CosineEmbeddingLoss::new(0.5);
        // Zero vectors (undefined cosine similarity)
        let input1 = Tensor::from_vec(CpuBackend::default(), vec![0.0], vec![1]).unwrap();
        let input2 = Tensor::from_vec(CpuBackend::default(), vec![0.0], vec![1]).unwrap();
        let target = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap(); // Similar pair

        let loss = loss_fn.forward(&input1, &input2, &target).unwrap();

        // Should handle zero vectors gracefully (cosine similarity = 0)
        assert_relative_eq!(loss.item().unwrap(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_embedding_loss_reductions() {
        let loss_fn_none = CosineEmbeddingLoss::with_params(0.5, Reduction::None);
        let loss_fn_sum = CosineEmbeddingLoss::with_params(0.5, Reduction::Sum);
        let loss_fn_mean = CosineEmbeddingLoss::with_params(0.5, Reduction::Mean);

        let input1 = Tensor::from_vec(CpuBackend::default(), vec![1.0], vec![1]).unwrap();
        let input2 = Tensor::from_vec(CpuBackend::default(), vec![0.0], vec![1]).unwrap();
        let target = Tensor::from_vec(CpuBackend::default(), vec![-1.0], vec![1]).unwrap();

        let result_none = loss_fn_none.forward(&input1, &input2, &target).unwrap();
        let result_sum = loss_fn_sum.forward(&input1, &input2, &target).unwrap();
        let result_mean = loss_fn_mean.forward(&input1, &input2, &target).unwrap();

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


