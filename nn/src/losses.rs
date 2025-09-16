//! Loss functions for neural networks
//!
//! This module provides common loss functions used in machine learning,
//! including MSE loss, cross-entropy loss, and other specialized losses.
//!
//! ## Mathematical Foundation
//!
//! ### Mean Squared Error (MSE)
//! ```math
//! MSE(y, ŷ) = (1/n) * Σ(yᵢ - ŷᵢ)²
//!
//! ∂MSE/∂ŷᵢ = (2/n) * (ŷᵢ - yᵢ)
//! ```
//!
//! ### Cross-Entropy Loss
//! ```math
//! CE(y, ŷ) = -Σ yᵢ * log(ŷᵢ)
//!
//! ∂CE/∂ŷᵢ = -yᵢ / ŷᵢ
//! ```
//!
//! ## References
//!
//! - [Deep Learning Book - Loss Functions](https://www.deeplearningbook.org/contents/ml.html)
//! - [Goodfellow et al., 2016 - Deep Learning](https://www.deeplearningbook.org/)

use crate::{Module, NNError, Result};
use coeus_tensor::{Dtype, FloatDtype, Tensor};

/// Mean Squared Error (MSE) loss function
///
/// Computes the mean squared error between predictions and targets:
/// `MSE(y, ŷ) = (1/n) * Σ(yᵢ - ŷᵢ)²`
///
/// This is commonly used for regression tasks.
#[derive(Debug, Clone, Copy, Default)]
pub struct MseLoss;

impl MseLoss {
    /// Create a new MSE loss function
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::MseLoss;
    ///
    /// let loss_fn = MseLoss::new();
    /// ```
    pub fn new() -> Self {
        Self
    }

    /// Compute MSE loss between predictions and targets
    ///
    /// # Arguments
    /// * `predictions` - Predicted values tensor
    /// * `targets` - Ground truth values tensor
    ///
    /// # Returns
    /// MSE loss as a scalar tensor
    ///
    /// # Errors
    /// Returns `NNError::ShapeMismatch` if shapes don't match
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::MseLoss;
    /// use coeus_tensor::Tensor;
    ///
    /// let loss_fn = MseLoss::new();
    /// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    /// let target = Tensor::from_vec(vec![1.1, 1.9, 3.2], vec![3]);
    ///
    /// let loss = loss_fn.forward(&pred, &target).unwrap();
    /// assert!(loss.item() > 0.0);
    /// ```
    pub fn forward<T: FloatDtype>(
        &self,
        predictions: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> Result<Tensor<T>> {
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

        // Compute sum of squared differences
        let sum_value: T = squared_diff.data().iter().cloned().sum();
        let n = T::from(predictions.numel()).unwrap();

        // Return mean squared error
        Ok(Tensor::scalar(sum_value / n))
    }

    /// Compute gradients of MSE loss with respect to predictions
    ///
    /// # Arguments
    /// * `predictions` - Predicted values tensor
    /// * `targets` - Ground truth values tensor
    ///
    /// # Returns
    /// Gradient tensor with same shape as predictions
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::MseLoss;
    /// use coeus_tensor::Tensor;
    ///
    /// let loss_fn = MseLoss::new();
    /// let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
    /// let target = Tensor::from_vec(vec![1.1, 1.9, 3.2], vec![3]);
    ///
    /// let grad = loss_fn.backward(&pred, &target).unwrap();
    /// assert_eq!(grad.shape(), pred.shape());
    /// ```
    pub fn backward<T: FloatDtype>(
        &self,
        predictions: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> Result<Tensor<T>> {
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

        let n = T::from(predictions.numel()).unwrap();
        let two = T::from(2.0).unwrap();

        // Compute gradient: (2/n) * (ŷ - y)
        let scale = two / n;
        Ok(diff.map(|x| *x * scale))
    }
}

impl<T: FloatDtype> Module<T> for MseLoss {
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // MSE loss expects two inputs: predictions and targets
        // For Module trait compatibility, we expect a concatenated tensor
        // This is a limitation - MSE is typically not used as a standalone module
        Err(crate::NNError::InvalidInput {
            message: "MSELoss should be used via forward() method, not Module::forward()"
                .to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![] // MSE loss has no learnable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![] // MSE loss has no learnable parameters
    }
}

/// Cross-Entropy Loss for classification tasks
///
/// Computes cross-entropy loss between predictions and targets:
/// `CE(y, ŷ) = -Σ yᵢ * log(ŷᵢ)`
///
/// This is commonly used for classification tasks with softmax outputs.
#[derive(Debug, Clone, Copy, Default)]
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Create a new cross-entropy loss function
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::CrossEntropyLoss;
    ///
    /// let loss_fn = CrossEntropyLoss::new();
    /// ```
    pub fn new() -> Self {
        Self
    }

    /// Compute cross-entropy loss between predictions and targets
    ///
    /// # Arguments
    /// * `predictions` - Predicted logits (before softmax) of shape (batch_size, num_classes)
    /// * `targets` - Ground truth class indices of shape (batch_size,)
    ///
    /// # Returns
    /// Cross-entropy loss as a scalar tensor
    ///
    /// # Errors
    /// Returns `NNError::ShapeMismatch` if shapes are incompatible
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::CrossEntropyLoss;
    /// use coeus_tensor::Tensor;
    ///
    /// let loss_fn = CrossEntropyLoss::new();
    /// let logits = Tensor::from_vec(vec![1.0, 2.0, 0.5, 0.1, 1.5, 2.1], vec![2, 3]);
    /// let targets = Tensor::from_vec(vec![1, 2], vec![2]); // Class indices
    ///
    /// let loss = loss_fn.forward(&logits, &targets).unwrap();
    /// assert!(loss.item() >= 0.0);
    /// ```
    pub fn forward<T: FloatDtype, I: Dtype + num_traits::ToPrimitive>(
        &self,
        predictions: &Tensor<T>,
        targets: &Tensor<I>,
    ) -> Result<Tensor<T>> {
        if predictions.ndim() != 2 {
            return Err(NNError::InvalidInput {
                message: "Predictions must be 2D tensor (batch_size, num_classes)".to_string(),
            });
        }

        if targets.ndim() != 1 {
            return Err(NNError::InvalidInput {
                message: "Targets must be 1D tensor (batch_size,)".to_string(),
            });
        }

        if predictions.shape()[0] != targets.shape()[0] {
            return Err(NNError::ShapeMismatch {
                expected: vec![predictions.shape()[0]],
                actual: targets.shape().to_vec(),
            });
        }

        // Apply log-softmax for numerical stability
        let log_probs = self.log_softmax(predictions)?;
        let batch_size = predictions.shape()[0];

        // Gather log probabilities for target classes
        let mut loss_sum = T::zero();
        for i in 0..batch_size {
            let target_idx = targets.data()[i]
                .to_usize()
                .ok_or_else(|| NNError::InvalidInput {
                    message: "Target indices must be valid usize values".to_string(),
                })?;
            loss_sum = loss_sum - log_probs.data()[i * predictions.shape()[1] + target_idx];
        }

        // Return mean loss
        let batch_size_float = T::from(batch_size).unwrap();
        Ok(Tensor::scalar(loss_sum / batch_size_float))
    }

    /// Compute log-softmax for numerical stability
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (batch_size, num_classes)
    ///
    /// # Returns
    /// Log-softmax probabilities of same shape
    fn log_softmax<T: FloatDtype>(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        // Find max for numerical stability
        let mut max_vals = Vec::new();
        let batch_size = x.shape()[0];
        let num_classes = x.shape()[1];

        for i in 0..batch_size {
            let mut max_val = x.data()[i * num_classes];
            for j in 1..num_classes {
                if x.data()[i * num_classes + j] > max_val {
                    max_val = x.data()[i * num_classes + j];
                }
            }
            max_vals.push(max_val);
        }

        // Compute log-softmax: log(exp(x - max) / sum(exp(x - max)))
        let mut log_probs = Vec::new();

        #[allow(clippy::needless_range_loop)]
        for i in 0..batch_size {
            let mut sum_exp = T::zero();
            for j in 0..num_classes {
                let shifted = x.data()[i * num_classes + j] - max_vals[i];
                sum_exp = sum_exp + shifted.exp();
            }

            for j in 0..num_classes {
                let shifted = x.data()[i * num_classes + j] - max_vals[i];
                let log_prob = shifted - sum_exp.ln();
                log_probs.push(log_prob);
            }
        }

        Ok(Tensor::from_vec(log_probs, x.shape().to_vec()))
    }

    /// Compute gradients of cross-entropy loss with respect to predictions
    ///
    /// # Arguments
    /// * `predictions` - Predicted logits of shape (batch_size, num_classes)
    /// * `targets` - Ground truth class indices of shape (batch_size,)
    ///
    /// # Returns
    /// Gradient tensor with same shape as predictions
    ///
    /// # Errors
    /// Returns `NNError::ShapeMismatch` if shapes are incompatible
    pub fn backward<T: FloatDtype>(
        &self,
        predictions: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        if predictions.shape().len() != 2 {
            return Err(NNError::InvalidInput {
                message: "Predictions must be 2D tensor (batch_size, num_classes)".to_string(),
            });
        }

        if targets.shape().len() != 1 {
            return Err(NNError::InvalidInput {
                message: "Targets must be 1D tensor (batch_size,)".to_string(),
            });
        }

        if predictions.shape()[0] != targets.shape()[0] {
            return Err(NNError::ShapeMismatch {
                expected: vec![predictions.shape()[0]],
                actual: targets.shape().to_vec(),
            });
        }

        // Compute softmax probabilities
        let softmax = self.compute_softmax(predictions)?;

        // Create gradient tensor (same shape as predictions)
        let mut grad_data = softmax.data().to_vec();

        // For each sample, subtract 1 from the target class probability
        let batch_size = predictions.shape()[0];
        let num_classes = predictions.shape()[1];

        for i in 0..batch_size {
            let target_idx = targets.data()[i]
                .to_usize()
                .ok_or_else(|| NNError::InvalidInput {
                    message: "Target indices must be valid usize values".to_string(),
                })?;

            // Gradient of cross-entropy w.r.t. softmax input is (softmax - one_hot_target)
            let idx = i * num_classes + target_idx;
            grad_data[idx] = grad_data[idx] - T::one();
        }

        // Normalize by batch size
        let batch_size_float = T::from(batch_size).unwrap();
        let grad_data_normalized: Vec<T> =
            grad_data.iter().map(|&x| x / batch_size_float).collect();

        Ok(Tensor::from_vec(
            grad_data_normalized,
            predictions.shape().to_vec(),
        ))
    }

    /// Compute softmax probabilities
    fn compute_softmax<T: FloatDtype>(&self, x: &Tensor<T>) -> Result<Tensor<T>> {
        // Find max for numerical stability
        let mut max_vals = Vec::new();
        let batch_size = x.shape()[0];
        let num_classes = x.shape()[1];

        for i in 0..batch_size {
            let mut max_val = x.data()[i * num_classes];
            for j in 1..num_classes {
                if x.data()[i * num_classes + j] > max_val {
                    max_val = x.data()[i * num_classes + j];
                }
            }
            max_vals.push(max_val);
        }

        // Compute softmax: exp(x - max) / sum(exp(x - max))
        let mut softmax_data = Vec::new();

        #[allow(clippy::needless_range_loop)]
        for i in 0..batch_size {
            let mut sum_exp = T::zero();
            for j in 0..num_classes {
                let shifted = x.data()[i * num_classes + j] - max_vals[i];
                sum_exp = sum_exp + shifted.exp();
            }

            for j in 0..num_classes {
                let shifted = x.data()[i * num_classes + j] - max_vals[i];
                let prob = shifted.exp() / sum_exp;
                softmax_data.push(prob);
            }
        }

        Ok(Tensor::from_vec(softmax_data, x.shape().to_vec()))
    }
}

impl<T: FloatDtype> Module<T> for CrossEntropyLoss {
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        // Cross-entropy loss expects two inputs: predictions and targets
        // For Module trait compatibility, we expect a concatenated tensor
        // This is a limitation - cross-entropy is typically not used as a standalone module
        Err(crate::NNError::InvalidInput {
            message: "CrossEntropyLoss should be used via forward() method, not Module::forward()"
                .to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![] // Cross-entropy loss has no learnable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![] // Cross-entropy loss has no learnable parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mse_loss_basic() {
        let loss_fn = MseLoss::new();

        // Simple case: perfect predictions
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let loss = loss_fn.forward(&pred, &target).unwrap();
        assert_relative_eq!(loss.item(), 0.0, epsilon = 1e-6);

        // Simple case: some error
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let target = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]);
        let loss = loss_fn.forward(&pred, &target).unwrap();
        // Expected: (1² + 1² + 1²) / 3 = 1.0
        assert_relative_eq!(loss.item(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mse_loss_gradients() {
        let loss_fn = MseLoss::new();

        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let target = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]);

        let grad = loss_fn.backward(&pred, &target).unwrap();

        // Expected gradient: (2/3) * (pred - target) = (2/3) * [-1, -1, -1]
        assert_eq!(grad.shape(), &[3]);
        assert_relative_eq!(grad.data()[0], -2.0 / 3.0, epsilon = 1e-6);
        assert_relative_eq!(grad.data()[1], -2.0 / 3.0, epsilon = 1e-6);
        assert_relative_eq!(grad.data()[2], -2.0 / 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mse_loss_shape_mismatch() {
        let loss_fn = MseLoss::new();

        let pred = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        let result = loss_fn.forward(&pred, &target);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_entropy_loss_basic() {
        let loss_fn = CrossEntropyLoss::new();

        // Simple 2-class, 1-sample case
        let logits = Tensor::from_vec(vec![0.0f64, 1.0f64], vec![1, 2]); // exp(0) ≈ 1, exp(1) ≈ 2.718
        let targets = Tensor::from_vec(vec![1i32], vec![1]); // Target class 1

        let loss = loss_fn.forward(&logits, &targets).unwrap();

        // Softmax probabilities: [1/(1+2.718), 2.718/(1+2.718)] ≈ [0.269, 0.731]
        // Cross-entropy: -log(0.731) ≈ -log(0.731) ≈ 0.315
        assert!(loss.item() > 0.0);
        assert!(loss.item() < 1.0);
    }

    #[test]
    fn test_cross_entropy_loss_perfect_prediction() {
        let loss_fn = CrossEntropyLoss::new();

        // Case where prediction is perfect (very high logit for target class)
        let logits = Tensor::from_vec(vec![-100.0f64, 100.0f64], vec![1, 2]);
        let targets = Tensor::from_vec(vec![1i32], vec![1]);

        let loss = loss_fn.forward(&logits, &targets).unwrap();

        // Loss should be very close to 0
        assert_relative_eq!(loss.item(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cross_entropy_loss_batch() {
        let loss_fn = CrossEntropyLoss::new();

        // 2 samples, 3 classes
        let logits = Tensor::from_vec(
            vec![1.0f64, 2.0f64, 0.5f64, 0.1f64, 1.5f64, 2.1f64],
            vec![2, 3],
        );
        let targets = Tensor::from_vec(vec![1i32, 2i32], vec![2]);

        let loss = loss_fn.forward(&logits, &targets).unwrap();

        // Should compute mean loss across batch
        assert!(loss.item() > 0.0);
        assert!(loss.item() < 3.0); // Reasonable upper bound
    }
}
