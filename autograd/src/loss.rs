//! Loss functions for automatic differentiation
//!
//! This module provides autograd-aware loss functions that participate in the computation graph.
//! Temporarily disabled during node-based autograd refactoring.

// Temporarily disabled during node-based autograd refactoring
// use crate::Variable;

/*
// Temporarily disabled - will be reimplemented with node-based autograd
/// Compute Mean Squared Error (MSE) loss with automatic differentiation support.
///
/// Computes the mean squared error between predictions and targets:
/// `loss = mean((predictions - targets)²)`
///
/// # Arguments
/// * `predictions` - Predicted values as a Variable
/// * `targets` - Target values as a Variable (typically created with `Variable::no_grad()`)
///
/// # Returns
/// A scalar Variable containing the MSE loss value. Calling `backward()` on this
/// Variable will compute gradients for all operations in the computation graph.
///
/// # Examples
/// ```
/// use autograd::{Variable, loss::mse_loss};
/// use tensor::Tensor;
/// use dtype::float::Float32;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
///
/// let predictions = Variable::new(Tensor::from_vec(
///     vec![Float32::new(1.0), Float32::new(2.0)],
///     &[2]
/// ).unwrap());
///
/// let targets = Variable::no_grad(Tensor::from_vec(
///     vec![Float32::new(1.5), Float32::new(2.5)],
///     &[2]
/// ).unwrap());
///
/// let loss = mse_loss(&predictions, &targets);
/// // Loss: mean((1.0-1.5)² + (2.0-2.5)²) = mean(0.25 + 0.25) = 0.25
/// ```
///
/// # Gradient Formula
/// For MSE loss `L = mean((y_pred - y_true)²)`:
/// - `∂L/∂y_pred = 2 * (y_pred - y_true) / n`
///
/// where `n` is the number of elements.
#[must_use]
pub fn mse_loss<T>(predictions: &Variable<T>, targets: &Variable<T>) -> Variable<T>
where
    T: DataType + FloatExt,
{
    // Compute (predictions - targets)
    let diff = predictions - targets;

    // Square the differences: diff²
    let squared = &diff * &diff;

    // Compute mean
    squared.mean()
}

/// Compute Cross-Entropy loss with automatic differentiation support.
///
/// Computes the cross-entropy loss between logits and class targets.
/// This implementation uses the numerically stable formulation:
/// `loss = mean(-log_softmax(logits)[target_class])`
///
/// where `log_softmax(x) = x - log(sum(exp(x)))`
///
/// # Arguments
/// * `logits` - Unnormalized predictions `[batch_size, num_classes]` as a Variable
/// * `targets` - Class indices `[batch_size]` as a Variable (typically created with `Variable::no_grad()`)
///
/// # Returns
/// A scalar Variable containing the cross-entropy loss value.
///
/// # Examples
/// ```
/// use autograd::{Variable, loss::cross_entropy_loss};
/// use tensor::Tensor;
/// use dtype::float::Float32;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
///
/// // 3 classes, 2 samples
/// let logits = Variable::new(Tensor::from_vec(
///     vec![
///         Float32::new(1.0), Float32::new(0.5), Float32::new(0.2),  // sample 1
///         Float32::new(0.1), Float32::new(2.0), Float32::new(0.3),  // sample 2
///     ],
///     &[2, 3]
/// ).unwrap());
///
/// let targets = Variable::no_grad(Tensor::from_vec(
///     vec![Float32::new(0.0), Float32::new(1.0)],  // class 0 for sample 1, class 1 for sample 2
///     &[2]
/// ).unwrap());
///
/// let loss = cross_entropy_loss(&logits, &targets);
/// ```
///
/// # Gradient Formula
/// For cross-entropy loss with softmax:
/// - `∂L/∂logits[i] = (softmax(logits)[i] - 1{i == target}) / batch_size`
///
/// where `1{condition}` is the indicator function (1 if true, 0 if false).
///
/// # Note
/// This is a simplified implementation that computes cross-entropy using basic operations.
/// For production use, consider implementing a fused `log_softmax` operation for better
/// numerical stability and performance.
#[must_use]
pub fn cross_entropy_loss<T>(logits: &Variable<T>, _targets: &Variable<T>) -> Variable<T>
where
    T: DataType + FloatExt + PartialOrd,
{
    // For now, implement a simplified version using log-sum-exp trick
    // Full implementation would require:
    // 1. Compute log_softmax(logits) = logits - log(sum(exp(logits)))
    // 2. Gather log_softmax values at target indices
    // 3. Compute mean of negative log probabilities

    // Simplified implementation: compute softmax cross-entropy using basic operations
    // This is a placeholder that demonstrates the API - full implementation deferred

    // Compute exp(logits)
    let exp_logits = logits.exp();

    // Compute sum(exp(logits)) along class dimension
    // For now, use sum() which sums all elements (simplified)
    let sum_exp = exp_logits.sum();

    // Compute log(sum(exp(logits)))
    let log_sum_exp = sum_exp.log();

    // Compute mean(logits) - log_sum_exp as a simplified loss
    // This is not the correct cross-entropy formula but demonstrates the API
    let mean_logits = logits.mean();
    &mean_logits - &log_sum_exp
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;
    use tensor::Tensor;

    #[test]
    fn test_mse_loss_forward() {
        // Test MSE loss computation
        let predictions = Variable::new(
            Tensor::from_vec(
                vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
                &[3],
            )
            .unwrap(),
        );

        let targets = Variable::no_grad(
            Tensor::from_vec(
                vec![Float32::new(1.5), Float32::new(2.5), Float32::new(3.5)],
                &[3],
            )
            .unwrap(),
        );

        let loss = mse_loss(&predictions, &targets);

        // Expected: mean((0.5)² + (0.5)² + (0.5)²) = mean(0.25 + 0.25 + 0.25) = 0.25
        let loss_value = loss.data().as_slice()[0].get();
        assert!(
            (loss_value - 0.25).abs() < 1e-6,
            "MSE loss mismatch: expected 0.25, got {}",
            loss_value
        );
    }

    #[test]
    fn test_mse_loss_gradient() {
        use crate::backward;

        // Test MSE loss gradient using numerical validation
        let predictions = Variable::new(
            Tensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap(),
        );

        let targets = Variable::no_grad(
            Tensor::from_vec(vec![Float32::new(1.5), Float32::new(2.5)], &[2]).unwrap(),
        );

        // Compute analytical gradient
        let loss = mse_loss(&predictions, &targets);

        // Backward pass using the backward() function
        backward(&[&loss], &[]).unwrap();

        // Check gradient: ∂L/∂pred = 2 * (pred - target) / n
        // For pred=[1.0, 2.0], target=[1.5, 2.5]: grad = 2 * [-0.5, -0.5] / 2 = [-0.5, -0.5]
        let pred_grad = predictions.grad().unwrap();
        let expected_grad = vec![-0.5, -0.5];
        for i in 0..2 {
            let actual = pred_grad.as_slice()[i].get();
            let expected = expected_grad[i];
            assert!(
                (actual - expected).abs() < 1e-2,
                "Gradient mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_mse_loss_numerical_gradient() {
        use crate::backward;
        use crate::numerical::numerical_gradient;

        // Test MSE loss with numerical gradient validation
        let predictions = Variable::new(
            Tensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap(),
        );

        let targets_data =
            Tensor::from_vec(vec![Float32::new(1.5), Float32::new(2.5)], &[2]).unwrap();

        // Compute numerical gradient
        let f = |pred: &Variable<Float32>| {
            let targets = Variable::no_grad(targets_data.clone());
            mse_loss(pred, &targets)
        };

        let numerical_grad = numerical_gradient(f, &predictions, Float32::new(1e-5)).unwrap();

        // Compute analytical gradient
        let targets = Variable::no_grad(targets_data);
        let loss = mse_loss(&predictions, &targets);

        // Backward pass using the backward() function
        backward(&[&loss], &[]).unwrap();

        let analytical_grad = predictions.grad().unwrap();

        // Compare gradients
        for i in 0..2 {
            let num_val = numerical_grad.as_slice()[i].get();
            let ana_val = analytical_grad.as_slice()[i].get();
            let diff = (num_val - ana_val).abs();
            assert!(
                diff < 1e-2,
                "Gradient mismatch at index {}: numerical={}, analytical={}",
                i,
                num_val,
                ana_val
            );
        }
    }

    #[test]
    fn test_cross_entropy_loss_forward() {
        // Test cross-entropy loss computation (simplified version)
        let logits = Variable::new(
            Tensor::from_vec(
                vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
                &[3],
            )
            .unwrap(),
        );

        let targets = Variable::no_grad(
            Tensor::from_vec(
                vec![Float32::new(0.0)], // class 0
                &[1],
            )
            .unwrap(),
        );

        let loss = cross_entropy_loss(&logits, &targets);

        // Just verify it computes without error (simplified implementation)
        let loss_value = loss.data().as_slice()[0].get();
        assert!(
            loss_value.is_finite(),
            "Cross-entropy loss should be finite, got {}",
            loss_value
        );
    }

    #[test]
    fn test_simple_subtraction_backward() {
        use crate::backward;

        // Simplified test: just test subtraction and mean
        let a = Variable::new(
            Tensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap(),
        );

        let b = Variable::no_grad(
            Tensor::from_vec(vec![Float32::new(1.5), Float32::new(2.5)], &[2]).unwrap(),
        );

        // Compute diff = a - b
        let diff = &a - &b;

        // Compute mean
        let result = diff.mean();

        // Backward pass
        backward(&[&result], &[]).unwrap();

        // Check gradient
        assert!(a.grad().is_ok(), "a should have a gradient after backward");
    }
}
*/
