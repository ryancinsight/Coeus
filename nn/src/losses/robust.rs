//! Robust loss functions
//!
//! Loss functions that are robust to outliers and provide
//! better stability in the presence of noisy data.

use super::{utils, Module, Reduction};
use coeus_tensor::{FloatDtype, Tensor};

/// Smooth L1 Loss (Huber Loss)
///
/// Computes smooth L1 loss between predictions and targets:
/// ```rust,ignore
/// SmoothL1(x) = {
///     0.5 * x^2     if |x| < beta
///     |x| - 0.5*beta  otherwise
/// }
/// ```
///
/// This loss is less sensitive to outliers than MSE loss while maintaining
/// differentiability. Commonly used in object detection (e.g., R-CNN, Fast R-CNN).
///
/// # Mathematical Properties
/// - Quadratic for small errors (smooth gradient)
/// - Linear for large errors (robust to outliers)
/// - Differentiable everywhere
/// - Parameter beta controls the transition point (default: 1.0)
///
/// # Example
/// ```rust
/// use coeus_nn::SmoothL1Loss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = SmoothL1Loss::new();
/// let predictions = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
/// let targets = Tensor::from_vec(vec![1.1, 1.5, 4.0], vec![3]);
///
/// let loss = loss_fn.forward(&predictions, &targets).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SmoothL1Loss {
    /// Reduction mode for the loss
    pub reduction: Reduction,
    /// Transition point between quadratic and linear regions
    pub beta: f64,
}

impl Default for SmoothL1Loss {
    fn default() -> Self {
        Self::new()
    }
}

impl SmoothL1Loss {
    /// Create a new Smooth L1 loss with default parameters (beta=1.0, mean reduction)
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
            beta: 1.0,
        }
    }

    /// Create a new Smooth L1 loss with specified beta parameter
    pub fn with_beta(beta: f64) -> Self {
        Self {
            reduction: Reduction::Mean,
            beta,
        }
    }

    /// Create a new Smooth L1 loss with specified reduction and beta
    pub fn with_params(reduction: Reduction, beta: f64) -> Self {
        Self { reduction, beta }
    }

    /// Compute Smooth L1 loss between predictions and targets
    pub fn forward<T: FloatDtype>(
        &self,
        predictions: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> crate::Result<Tensor<T>> {
        if predictions.shape() != targets.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: predictions.shape().to_vec(),
                actual: targets.shape().to_vec(),
            });
        }

        let pred_data = predictions.data();
        let target_data = targets.data();

        let mut losses = Vec::with_capacity(predictions.numel());
        let beta = T::from(self.beta).unwrap();
        let half = T::from(0.5).unwrap();

        for (&pred, &target) in pred_data.iter().zip(target_data.iter()) {
            let diff = pred - target;
            let abs_diff = if diff >= T::zero() { diff } else { -diff };

            let loss_val = if abs_diff < beta {
                // Quadratic region: 0.5 * x^2
                half * diff * diff
            } else {
                // Linear region: |x| - 0.5 * beta
                abs_diff - half * beta
            };

            losses.push(loss_val);
        }

        let loss_tensor = Tensor::from_vec(losses, predictions.shape().to_vec());
        utils::apply_reduction(&loss_tensor, self.reduction)
    }

    /// Compute gradients of Smooth L1 loss with respect to predictions
    ///
    /// # Mathematical Derivation
    /// For Smooth L1 loss with parameter beta:
    /// ```rust,ignore
    /// dSmoothL1/dx = {
    ///     x        if |x| < beta
    ///     sign(x)  if |x| >= beta
    /// }
    /// ```
    pub fn backward<T: FloatDtype>(
        &self,
        predictions: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> crate::Result<Tensor<T>> {
        if predictions.shape() != targets.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: predictions.shape().to_vec(),
                actual: targets.shape().to_vec(),
            });
        }

        let pred_data = predictions.data();
        let target_data = targets.data();

        let mut gradients = Vec::with_capacity(predictions.numel());
        let beta = T::from(self.beta).unwrap();

        for (&pred, &target) in pred_data.iter().zip(target_data.iter()) {
            let diff = pred - target;
            let abs_diff = if diff >= T::zero() { diff } else { -diff };

            let grad_val = if abs_diff < beta {
                // Quadratic region: gradient = x
                diff
            } else {
                // Linear region: gradient = sign(x)
                if diff >= T::zero() {
                    T::one()
                } else {
                    -T::one()
                }
            };

            gradients.push(grad_val);
        }

        let grad_tensor = Tensor::from_vec(gradients, predictions.shape().to_vec());

        // Apply reduction scaling
        let scale = match self.reduction {
            Reduction::None => T::one(),
            Reduction::Sum => T::one(),
            Reduction::Mean => {
                let n = T::from(predictions.numel()).unwrap();
                T::one() / n
            }
        };

        Ok(grad_tensor.map(|x| *x * scale))
    }
}

impl<T: FloatDtype> Module<T> for SmoothL1Loss {
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Err(crate::NNError::InvalidInput {
            message: "SmoothL1Loss should be used via forward() method with two inputs".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![]
    }
}

/// Huber Loss (alternative name for Smooth L1 Loss)
///
/// This is an alias for SmoothL1Loss, commonly used in reinforcement learning
/// and robust regression. The Huber loss provides a balance between MSE and MAE.
pub type HuberLoss = SmoothL1Loss;

/// Log-Cosh Loss
///
/// Computes the logarithm of the hyperbolic cosine of the prediction error:
/// `LogCosh(y, ŷ) = log(cosh(ŷ - y))`
///
/// This loss function is smooth everywhere and approximately equal to
/// `(x^2)/2` for small x and `|x| - log(2)` for large x.
///
/// # Mathematical Properties
/// - Smooth and twice differentiable everywhere
/// - Approximately quadratic for small errors
/// - Approximately linear for large errors
/// - More robust to outliers than MSE
///
/// # Example
/// ```rust
/// use coeus_nn::LogCoshLoss;
/// use coeus_tensor::Tensor;
///
/// let loss_fn = LogCoshLoss::new();
/// let predictions = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
/// let targets = Tensor::from_vec(vec![1.1, 1.9, 3.2], vec![3]);
///
/// let loss = loss_fn.forward(&predictions, &targets).unwrap();
/// assert!(loss.item().unwrap() >= 0.0);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct LogCoshLoss {
    /// Reduction mode for the loss
    pub reduction: Reduction,
}

impl LogCoshLoss {
    /// Create a new Log-Cosh loss with mean reduction
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }

    /// Create a new Log-Cosh loss with specified reduction
    pub fn with_reduction(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute Log-Cosh loss between predictions and targets
    pub fn forward<T: FloatDtype>(
        &self,
        predictions: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> crate::Result<Tensor<T>> {
        if predictions.shape() != targets.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: predictions.shape().to_vec(),
                actual: targets.shape().to_vec(),
            });
        }

        let pred_data = predictions.data();
        let target_data = targets.data();

        let mut losses = Vec::with_capacity(predictions.numel());

        for (&pred, &target) in pred_data.iter().zip(target_data.iter()) {
            let diff = pred - target;

            // log(cosh(x)) = log((exp(x) + exp(-x))/2)
            // For numerical stability, use: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
            let abs_diff = if diff >= T::zero() { diff } else { -diff };
            let two = T::from(2.0).unwrap();
            let log_2 = two.ln();

            let log_cosh = if abs_diff > T::from(10.0).unwrap() {
                // For large |x|, log(cosh(x)) ≈ |x| - log(2)
                abs_diff - log_2
            } else {
                // For small |x|, use the full formula
                let exp_neg_2x = (-two * abs_diff).exp();
                abs_diff + (T::one() + exp_neg_2x).ln() - log_2
            };

            losses.push(log_cosh);
        }

        let loss_tensor = Tensor::from_vec(losses, predictions.shape().to_vec());
        utils::apply_reduction(&loss_tensor, self.reduction)
    }

    /// Compute gradients of Log-Cosh loss with respect to predictions
    ///
    /// # Mathematical Derivation
    /// For Log-Cosh loss `L = log(cosh(x))`:
    /// `dL/dx = tanh(x)`
    pub fn backward<T: FloatDtype>(
        &self,
        predictions: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> crate::Result<Tensor<T>> {
        if predictions.shape() != targets.shape() {
            return Err(crate::NNError::ShapeMismatch {
                expected: predictions.shape().to_vec(),
                actual: targets.shape().to_vec(),
            });
        }

        let pred_data = predictions.data();
        let target_data = targets.data();

        let mut gradients = Vec::with_capacity(predictions.numel());

        for (&pred, &target) in pred_data.iter().zip(target_data.iter()) {
            let diff = pred - target;
            // Gradient of log(cosh(x)) is tanh(x)
            let grad_val = diff.tanh();
            gradients.push(grad_val);
        }

        let grad_tensor = Tensor::from_vec(gradients, predictions.shape().to_vec());

        // Apply reduction scaling
        let scale = match self.reduction {
            Reduction::None => T::one(),
            Reduction::Sum => T::one(),
            Reduction::Mean => {
                let n = T::from(predictions.numel()).unwrap();
                T::one() / n
            }
        };

        Ok(grad_tensor.map(|x| *x * scale))
    }
}

impl<T: FloatDtype> Module<T> for LogCoshLoss {
    fn forward(&self, _input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        Err(crate::NNError::InvalidInput {
            message: "LogCoshLoss should be used via forward() method with two inputs".to_string(),
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
    fn test_smooth_l1_loss_quadratic_region() {
        let loss_fn = SmoothL1Loss::new();

        // Test quadratic region (|x| < 1)
        let pred = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let target = Tensor::from_vec(vec![1.5, 2.3], vec![2]); // Differences: -0.5, -0.3

        let loss = loss_fn.forward(&pred, &target).unwrap();

        // Expected: (0.5 * 0.5² + 0.5 * 0.3²) / 2 = (0.125 + 0.045) / 2 = 0.085
        assert_relative_eq!(loss.item().unwrap(), 0.085, epsilon = 1e-6);
    }

    #[test]
    fn test_smooth_l1_loss_linear_region() {
        let loss_fn = SmoothL1Loss::new();

        // Test linear region (|x| >= 1)
        let pred = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let target = Tensor::from_vec(vec![3.0, 0.5], vec![2]); // Differences: -2.0, 1.5

        let loss = loss_fn.forward(&pred, &target).unwrap();

        // Expected: (|2.0| - 0.5 + |1.5| - 0.5) / 2 = (1.5 + 1.0) / 2 = 1.25
        assert_relative_eq!(loss.item().unwrap(), 1.25, epsilon = 1e-6);
    }

    #[test]
    fn test_smooth_l1_loss_custom_beta() {
        let loss_fn = SmoothL1Loss::with_beta(2.0);

        // With beta=2.0, differences of 1.0 should be in quadratic region
        let pred = Tensor::from_vec(vec![1.0], vec![1]);
        let target = Tensor::from_vec(vec![2.0], vec![1]); // Difference: -1.0

        let loss = loss_fn.forward(&pred, &target).unwrap();

        // Expected: 0.5 * 1.0² = 0.5 (quadratic region since |1.0| < 2.0)
        assert_relative_eq!(loss.item().unwrap(), 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_smooth_l1_loss_backward() {
        let loss_fn = SmoothL1Loss::new();
        let pred = Tensor::from_vec(vec![1.0, 3.0], vec![2]);
        let target = Tensor::from_vec(vec![1.5, 1.0], vec![2]); // Differences: -0.5, 2.0

        let grad = loss_fn.backward(&pred, &target).unwrap();

        // Expected gradients:
        // For diff = -0.5 (quadratic): grad = -0.5 / 2 = -0.25
        // For diff = 2.0 (linear): grad = sign(2.0) / 2 = 0.5
        assert_relative_eq!(grad.data()[0], -0.25, epsilon = 1e-6);
        assert_relative_eq!(grad.data()[1], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_log_cosh_loss_basic() {
        let loss_fn = LogCoshLoss::new();
        let pred = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let target = Tensor::from_vec(vec![1.1, 1.9, 3.2], vec![3]);

        let loss = loss_fn.forward(&pred, &target).unwrap();
        assert!(loss.item().unwrap() >= 0.0);

        // Test backward pass
        let grad = loss_fn.backward(&pred, &target).unwrap();
        assert_eq!(grad.shape(), pred.shape());
    }

    #[test]
    fn test_log_cosh_loss_small_errors() {
        let loss_fn = LogCoshLoss::new();
        let pred = Tensor::from_vec(vec![1.0], vec![1]);
        let target = Tensor::from_vec(vec![1.01], vec![1]); // Small error: -0.01

        let loss = loss_fn.forward(&pred, &target).unwrap();

        // For small x, log(cosh(x)) ≈ x^2/2
        let expected = 0.01 * 0.01 / 2.0;
        assert_relative_eq!(loss.item().unwrap(), expected, epsilon = 1e-6);
    }

    #[test]
    fn test_log_cosh_loss_backward() {
        let loss_fn = LogCoshLoss::new();
        let pred = Tensor::from_vec(vec![1.0], vec![1]);
        let target = Tensor::from_vec(vec![0.0], vec![1]); // Difference: 1.0

        let grad = loss_fn.backward(&pred, &target).unwrap();

        // Gradient of log(cosh(x)) is tanh(x)
        let expected_grad = 1.0f32.tanh();
        assert_relative_eq!(grad.data()[0], expected_grad, epsilon = 1e-6);
    }

    #[test]
    fn test_robust_losses_outlier_handling() {
        // Compare how different losses handle outliers
        let pred = Tensor::from_vec(vec![1.0, 2.0, 100.0], vec![3]); // Large outlier
        let target = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);

        let smooth_l1 = SmoothL1Loss::new();
        let log_cosh = LogCoshLoss::new();

        let smooth_l1_loss = smooth_l1.forward(&pred, &target).unwrap();
        let log_cosh_loss = log_cosh.forward(&pred, &target).unwrap();

        // Both should be much smaller than MSE would be due to outlier robustness
        assert!(smooth_l1_loss.item().unwrap() < 100.0); // Much less than MSE would give
        assert!(log_cosh_loss.item().unwrap() < 100.0);
    }

    #[test]
    fn test_robust_losses_reductions() {
        let pred = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let target = Tensor::from_vec(vec![0.0, 0.0], vec![2]);

        // Test different reductions for SmoothL1Loss
        let loss_none = SmoothL1Loss::with_params(Reduction::None, 1.0);
        let loss_sum = SmoothL1Loss::with_params(Reduction::Sum, 1.0);
        let loss_mean = SmoothL1Loss::with_params(Reduction::Mean, 1.0);

        let result_none = loss_none.forward(&pred, &target).unwrap();
        let result_sum = loss_sum.forward(&pred, &target).unwrap();
        let result_mean = loss_mean.forward(&pred, &target).unwrap();

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
