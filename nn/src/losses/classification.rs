//! Classification loss functions
//!
//! Loss functions for discrete class prediction tasks, including
//! multi-class and binary classification losses.

use super::{utils, Module, NNError, Reduction, Result};
use coeus_tensor::{Dtype, FloatDtype, Tensor};

/// Cross-Entropy Loss for multi-class classification
///
/// Computes cross-entropy loss between predicted logits and target class indices:
/// `CE(y, ŷ) = -Σ yᵢ * log(softmax(ŷᵢ))`
///
/// This is the standard loss function for multi-class classification tasks.
///
/// # Mathematical Properties
/// - Combines softmax activation with negative log-likelihood
/// - Numerically stable implementation using log-sum-exp trick
/// - Gradient: `∂CE/∂ŷ = softmax(ŷ) - one_hot(y)`
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
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CrossEntropyLoss {
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl CrossEntropyLoss {
    /// Create a new cross-entropy loss with mean reduction
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Create a new cross-entropy loss with specified reduction
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute cross-entropy loss between logits and target class indices
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

        // Compute log-softmax for numerical stability
        let log_probs = utils::log_softmax(predictions)?;
        let batch_size = predictions.shape()[0];
        let num_classes = predictions.shape()[1];

        // Gather log probabilities for target classes
        let mut losses = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let target_idx = targets.data()[i]
                .to_usize()
                .ok_or_else(|| NNError::InvalidInput {
                    message: "Target indices must be valid usize values".to_string(),
                })?;

            if target_idx >= num_classes {
                return Err(NNError::InvalidInput {
                    message: format!(
                        "Target index {} out of bounds for {} classes",
                        target_idx, num_classes
                    ),
                });
            }

            let log_prob = log_probs.data()[i * num_classes + target_idx];
            losses.push(-log_prob);
        }

        let loss_tensor = Tensor::from_vec(losses, vec![batch_size]);
        utils::apply_reduction(&loss_tensor, self.reduction)
    }

    /// Compute gradients of cross-entropy loss with respect to logits
    ///
    /// # Mathematical Derivation
    /// For cross-entropy loss with softmax:
    /// `∂CE/∂ŷᵢ = softmax(ŷᵢ) - δᵢⱼ` where δᵢⱼ is 1 if i=j (target class), 0 otherwise
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
        let softmax = utils::softmax(predictions)?;
        let mut grad_data = softmax.data().to_vec();

        // Subtract 1 from the target class probability
        let batch_size = predictions.shape()[0];
        let num_classes = predictions.shape()[1];

        for i in 0..batch_size {
            let target_idx = targets.data()[i]
                .to_usize()
                .ok_or_else(|| NNError::InvalidInput {
                    message: "Target indices must be valid usize values".to_string(),
                })?;

            let idx = i * num_classes + target_idx;
            grad_data[idx] = grad_data[idx] - T::one();
        }

        // Apply reduction scaling
        let scale = match self.reduction {
            Reduction::None => T::one(),
            Reduction::Sum => T::one(),
            Reduction::Mean => {
                let batch_size_float = T::from(batch_size).unwrap();
                T::one() / batch_size_float
            }
        };

        let grad_data_scaled: Vec<T> = grad_data.iter().map(|&x| x * scale).collect();

        Ok(Tensor::from_vec(
            grad_data_scaled,
            predictions.shape().to_vec(),
        ))
    }
}

impl<T: FloatDtype> Module<T> for CrossEntropyLoss {
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Err(crate::NNError::InvalidInput {
            message: "CrossEntropyLoss should be used via forward() method with two inputs"
                .to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

/// Negative Log Likelihood Loss for classification
///
/// Computes negative log likelihood loss between log-probabilities and targets:
/// `NLL(y, ŷ) = -Σ yᵢ * ŷᵢ` where ŷ are log-probabilities
///
/// This is commonly used when the model outputs log-probabilities directly.
///
/// # Mathematical Properties
/// - Expects log-probabilities as input (not raw logits)
/// - More numerically stable than CrossEntropyLoss for some use cases
/// - Gradient: `∂NLL/∂ŷᵢ = -δᵢⱼ` where δᵢⱼ is 1 if i=j (target class)
///
/// # Example
/// ```rust
/// use coeus_nn::NLLLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = NLLLoss::new();
/// let log_probs = Tensor::from_vec(vec![-1.0, -0.5, -2.0, -0.1, -1.5, -0.8], vec![2, 3]);
/// let targets = Tensor::from_vec(vec![1, 2], vec![2]); // Class indices
///
/// let loss = loss_fn.forward(&log_probs, &targets).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct NLLLoss {
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl NLLLoss {
    /// Create a new NLL loss with mean reduction
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Create a new NLL loss with specified reduction
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute negative log likelihood loss between log-probabilities and targets
    pub fn forward<T: FloatDtype, I: Dtype + num_traits::ToPrimitive>(
        &self,
        log_probs: &Tensor<T>,
        targets: &Tensor<I>,
    ) -> crate::Result<Tensor<T>> {
        if log_probs.ndim() != 2 {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "Log probabilities must be 2D (batch_size, num_classes), got shape {:?}",
                    log_probs.shape()
                ),
            });
        }

        if targets.ndim() != 1 {
            return Err(crate::NNError::InvalidInput {
                message: format!(
                    "Targets must be 1D (batch_size,), got shape {:?}",
                    targets.shape()
                ),
            });
        }

        if log_probs.shape()[0] != targets.shape()[0] {
            return Err(crate::NNError::ShapeMismatch {
                expected: vec![targets.shape()[0], log_probs.shape()[1]],
                actual: log_probs.shape().to_vec(),
            });
        }

        let batch_size = targets.shape()[0];
        let num_classes = log_probs.shape()[1];
        let targets_data = targets.data();

        // Compute NLL loss: -Σ log_probs[batch_idx, target_idx]
        let mut losses = Vec::with_capacity(batch_size);

        #[allow(clippy::needless_range_loop)]
        for batch_idx in 0..batch_size {
            let target_idx = targets_data[batch_idx].to_usize().unwrap();

            if target_idx >= num_classes {
                return Err(crate::NNError::InvalidInput {
                    message: format!(
                        "Target index {} out of bounds for {} classes",
                        target_idx, num_classes
                    ),
                });
            }

            let linear_idx = batch_idx * num_classes + target_idx;
            let log_prob = log_probs.data()[linear_idx];
            losses.push(-log_prob);
        }

        let loss_tensor = Tensor::from_vec(losses, vec![batch_size]);
        utils::apply_reduction(&loss_tensor, self.reduction)
    }
}

impl<T: FloatDtype> Module<T> for NLLLoss {
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Err(crate::NNError::InvalidInput {
            message: "NLLLoss should be used via forward() method with two inputs".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

/// Binary Cross Entropy Loss for binary classification
///
/// Computes binary cross entropy loss between predictions and targets:
/// `BCE(y, ŷ) = -[y * log(ŷ) + (1-y) * log(1-ŷ)]`
///
/// This is used for binary classification tasks where predictions are probabilities.
///
/// # Mathematical Properties
/// - Expects probabilities in range [0, 1] as input
/// - Numerically stable with epsilon clamping
/// - Gradient: `∂BCE/∂ŷ = -(y/ŷ - (1-y)/(1-ŷ))`
///
/// # Example
/// ```rust
/// use coeus_nn::BCELoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = BCELoss::new();
/// let predictions = Tensor::from_vec(vec![0.9, 0.3, 0.8], vec![3]);
/// let targets = Tensor::from_vec(vec![1.0, 0.0, 1.0], vec![3]);
///
/// let loss = loss_fn.forward(&predictions, &targets).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BCELoss {
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl BCELoss {
    /// Create a new BCE loss with mean reduction
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Create a new BCE loss with specified reduction
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute binary cross entropy loss between predictions and targets
    pub fn forward<T: FloatDtype>(
        &self,
        predictions: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> crate::Result<Tensor<T>> {
        if predictions.shape() != targets.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: targets.shape().to_vec(),
                actual: predictions.shape().to_vec(),
            });
        }

        let pred_data = predictions.data();
        let target_data = targets.data();

        let mut losses = Vec::with_capacity(predictions.numel());

        for (&pred, &target) in pred_data.iter().zip(target_data.iter()) {
            // Clamp predictions to avoid log(0)
            let pred_clamped = utils::clamp_for_log(pred);
            let one_minus_pred_clamped = utils::clamp_for_log(T::one() - pred);

            // BCE = -[y * log(ŷ) + (1-y) * log(1-ŷ)]
            let term1 = target * pred_clamped.ln();
            let term2 = (T::one() - target) * one_minus_pred_clamped.ln();
            losses.push(-(term1 + term2));
        }

        let loss_tensor = Tensor::from_vec(losses, predictions.shape().to_vec());
        utils::apply_reduction(&loss_tensor, self.reduction)
    }
}

impl<T: FloatDtype> Module<T> for BCELoss {
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Err(crate::NNError::InvalidInput {
            message: "BCELoss should be used via forward() method with two inputs".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

/// Binary Cross Entropy Loss with Logits
///
/// Computes BCE loss with sigmoid applied to logits internally:
/// `BCEWithLogits(y, x) = -[y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x))]`
///
/// This is more numerically stable than applying sigmoid manually then using BCELoss.
///
/// # Mathematical Properties
/// - Combines sigmoid activation with BCE loss
/// - More numerically stable than separate sigmoid + BCE
/// - Gradient: `∂BCE/∂x = sigmoid(x) - y`
///
/// # Example
/// ```rust
/// use coeus_nn::BCEWithLogitsLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = BCEWithLogitsLoss::new();
/// let logits = Tensor::from_vec(vec![2.0, -1.5, 1.8], vec![3]);
/// let targets = Tensor::from_vec(vec![1.0, 0.0, 1.0], vec![3]);
///
/// let loss = loss_fn.forward(&logits, &targets).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct BCEWithLogitsLoss {
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl BCEWithLogitsLoss {
    /// Create a new BCE with logits loss with mean reduction
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Create a new BCE with logits loss with specified reduction
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute BCE loss with sigmoid applied to logits
    pub fn forward<T: FloatDtype>(
        &self,
        logits: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> crate::Result<Tensor<T>> {
        if logits.shape() != targets.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: targets.shape().to_vec(),
                actual: logits.shape().to_vec(),
            });
        }

        let logits_data = logits.data();
        let targets_data = targets.data();

        let mut losses = Vec::with_capacity(logits.numel());

        for (&logit, &target) in logits_data.iter().zip(targets_data.iter()) {
            // Compute sigmoid(logit) = 1 / (1 + exp(-logit))
            let sigmoid = T::one() / (T::one() + (-logit).exp());

            // Clamp sigmoid to avoid log(0)
            let sigmoid_clamped = utils::clamp_for_log(sigmoid);
            let one_minus_sigmoid_clamped = utils::clamp_for_log(T::one() - sigmoid);

            // BCE = -[y * log(sigmoid) + (1-y) * log(1-sigmoid)]
            let term1 = target * sigmoid_clamped.ln();
            let term2 = (T::one() - target) * one_minus_sigmoid_clamped.ln();
            losses.push(-(term1 + term2));
        }

        let loss_tensor = Tensor::from_vec(losses, logits.shape().to_vec());
        utils::apply_reduction(&loss_tensor, self.reduction)
    }
}

impl<T: FloatDtype> Module<T> for BCEWithLogitsLoss {
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Err(crate::NNError::InvalidInput {
            message: "BCEWithLogitsLoss should be used via forward() method with two inputs"
                .to_string(),
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
    fn test_cross_entropy_basic() {
        let loss_fn = CrossEntropyLoss::new();
        let logits = Tensor::from_vec(vec![1.0, 2.0, 0.5, 0.1, 1.5, 2.1], vec![2, 3]);
        let targets = Tensor::from_vec(vec![1, 2], vec![2]);

        let loss = loss_fn.forward(&logits, &targets).unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_cross_entropy_perfect_prediction() {
        let loss_fn = CrossEntropyLoss::new();
        // Perfect prediction: very high logit for correct class
        let logits = Tensor::from_vec(vec![-10.0, 10.0, -10.0], vec![1, 3]);
        let targets = Tensor::from_vec(vec![1], vec![1]);

        let loss = loss_fn.forward(&logits, &targets).unwrap();
        assert!(loss.item().unwrap() < 1e-6); // Should be very small
    }

    #[test]
    fn test_nll_loss_basic() {
        let loss_fn = NLLLoss::new();
        let log_probs = Tensor::from_vec(vec![-1.0, -0.5, -2.0, -0.1, -1.5, -0.8], vec![2, 3]);
        let targets = Tensor::from_vec(vec![1, 2], vec![2]);

        let loss = loss_fn.forward(&log_probs, &targets).unwrap();
        assert_relative_eq!(loss.item().unwrap(), 0.65, epsilon = 1e-6);
    }

    #[test]
    fn test_bce_loss_basic() {
        let loss_fn = BCELoss::new();
        let predictions = Tensor::from_vec(vec![0.9, 0.3, 0.8], vec![3]);
        let targets = Tensor::from_vec(vec![1.0, 0.0, 1.0], vec![3]);

        let loss = loss_fn.forward(&predictions, &targets).unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_bce_loss_perfect_prediction() {
        let loss_fn = BCELoss::new();
        let predictions = Tensor::from_vec(vec![1.0, 0.0, 1.0], vec![3]);
        let targets = Tensor::from_vec(vec![1.0, 0.0, 1.0], vec![3]);

        let loss = loss_fn.forward(&predictions, &targets).unwrap();
        // Should be very small due to epsilon clamping
        assert!(loss.item().unwrap() < 1e-10);
    }

    #[test]
    fn test_bce_with_logits_basic() {
        let loss_fn = BCEWithLogitsLoss::new();
        let logits = Tensor::from_vec(vec![2.0, -1.5, 1.8], vec![3]);
        let targets = Tensor::from_vec(vec![1.0, 0.0, 1.0], vec![3]);

        let loss = loss_fn.forward(&logits, &targets).unwrap();
        assert!(loss.item().unwrap() >= 0.0);
    }

    #[test]
    fn test_classification_losses_reductions() {
        let logits = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let targets = Tensor::from_vec(vec![0, 1], vec![2]);

        // Test different reductions for CrossEntropyLoss
        let loss_none = CrossEntropyLoss::with_reduction(Reduction::None);
        let loss_sum = CrossEntropyLoss::with_reduction(Reduction::Sum);
        let loss_mean = CrossEntropyLoss::with_reduction(Reduction::Mean);

        let result_none = loss_none.forward(&logits, &targets).unwrap();
        let result_sum = loss_sum.forward(&logits, &targets).unwrap();
        let result_mean = loss_mean.forward(&logits, &targets).unwrap();

        // None should return per-sample losses
        assert_eq!(result_none.shape(), &[2]);

        // Sum should be sum of individual losses
        let expected_sum = result_none.data()[0] + result_none.data()[1];
        assert_relative_eq!(result_sum.item().unwrap(), expected_sum, epsilon = 1e-6);

        // Mean should be average of individual losses
        let expected_mean = expected_sum / 2.0;
        assert_relative_eq!(result_mean.item().unwrap(), expected_mean, epsilon = 1e-6);
    }
}
