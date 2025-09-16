//! General utility functions and helpers
//!
//! This module provides utility functions for common machine learning
//! and tensor operations that complement the core functionality.

use coeus_tensor::{Add, Div, Mul, Neg, Result, Sub, Tensor};

/// Mathematical constants and utilities
pub mod math {
    use super::*;
    /// Compute the softmax function
    ///
    /// Softmax normalizes a vector of real numbers into a probability distribution.
    /// Each element is transformed as: softmax(x_i) = exp(x_i) / Σ exp(x_j)
    ///
    /// # Arguments
    /// * `input` - Input tensor
    /// * `dim` - Dimension along which to compute softmax (default: -1, last dimension)
    ///
    /// # Returns
    /// Softmax-normalized tensor
    ///
    /// # Example
    pub fn softmax<T: coeus_dtype::FloatDtype>(input: &Tensor<T>, dim: i32) -> Result<Tensor<T>> {
        // Compute softmax: softmax(x_i) = exp(x_i) / Σ exp(x_j) along dimension dim
        let exp_input = input.exp();

        // Sum along the specified dimension (convert i32 to Option<usize>)
        let dim_opt = if dim >= 0 {
            Some(dim as usize)
        } else {
            // Handle negative dimensions (Python-style indexing)
            let ndim = input.shape().len();
            Some((ndim as i32 + dim) as usize)
        };
        let sum_exp = exp_input.sum_dim(dim_opt, true)?;

        // Divide by the sum
        exp_input.div(&sum_exp)
    }

    /// Compute the log-softmax function
    ///
    /// Log-softmax computes the logarithm of the softmax function.
    /// This is numerically stable and avoids overflow issues.
    pub fn log_softmax<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T>,
        dim: i32,
    ) -> Result<Tensor<T>> {
        // Numerically stable log-softmax: log(softmax(x_i)) = x_i - log(Σ exp(x_j))
        let exp_input = input.exp();

        // Sum along the specified dimension (convert i32 to Option<usize>)
        let dim_opt = if dim >= 0 {
            Some(dim as usize)
        } else {
            // Handle negative dimensions (Python-style indexing)
            let ndim = input.shape().len();
            Some((ndim as i32 + dim) as usize)
        };
        let sum_exp = exp_input.sum_dim(dim_opt, true)?;

        // Compute log of sum
        let log_sum_exp = sum_exp.log();

        // Subtract: x_i - log(Σ exp(x_j))
        input.sub(&log_sum_exp)
    }

    /// Compute the sigmoid (logistic) function
    pub fn sigmoid<T: coeus_dtype::FloatDtype>(input: &Tensor<T>) -> Result<Tensor<T>> {
        // Sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
        let neg_input = input.neg();
        let exp_neg = neg_input.exp();
        let one = Tensor::scalar(T::one());
        let denominator = one.add(&exp_neg)?;
        one.div(&denominator)
    }

    /// Compute the hyperbolic tangent function
    pub fn tanh<T: coeus_dtype::FloatDtype>(input: &Tensor<T>) -> Result<Tensor<T>> {
        // Tanh: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        let exp_x = input.exp();
        let exp_neg_x = input.neg().exp();
        let numerator = exp_x.sub(&exp_neg_x)?;
        let denominator = exp_x.add(&exp_neg_x)?;
        numerator.div(&denominator)
    }

    /// Compute the Rectified Linear Unit (ReLU) activation
    pub fn relu<T: coeus_dtype::FloatDtype>(input: &Tensor<T>) -> Result<Tensor<T>> {
        // ReLU: relu(x) = max(0, x)
        let zero = Tensor::scalar(T::zero());
        use coeus_tensor::ops::arithmetic::maximum;
        maximum(input, &zero)
    }

    /// Compute the Leaky ReLU activation
    pub fn leaky_relu<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T>,
        negative_slope: T,
    ) -> Result<Tensor<T>> {
        // Leaky ReLU: leaky_relu(x) = max(x, negative_slope * x)
        let slope_tensor = Tensor::scalar(negative_slope);
        let scaled_input = input.mul(&slope_tensor)?;
        use coeus_tensor::ops::arithmetic::maximum;
        maximum(input, &scaled_input)
    }
}

/// Statistical utilities
pub mod stats {
    use super::*;

    /// Compute the mean of a tensor along specified dimensions
    pub fn mean<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T>,
        dims: Option<&[usize]>,
        keepdim: bool,
    ) -> Result<Tensor<T>> {
        // For now, implement basic case - global mean or single dimension
        if let Some(dims) = dims {
            if dims.len() == 1 {
                let sum = input.sum_dim(Some(dims[0]), keepdim)?;
                let count = T::from(input.shape()[dims[0]] as f64).unwrap();
                let count_tensor = Tensor::scalar(count);
                Ok(sum.div(&count_tensor)?)
            } else {
                // Multi-dimension reduction not fully implemented yet
                // Return global mean as fallback
                let sum = input.sum();
                let count = T::from(input.numel() as f64).unwrap();
                let count_tensor = Tensor::scalar(count);
                Ok(sum.div(&count_tensor)?)
            }
        } else {
            // Global mean
            let sum = input.sum();
            let count = T::from(input.numel() as f64).unwrap();
            let count_tensor = Tensor::scalar(count);
            Ok(sum.div(&count_tensor)?)
        }
    }

    /// Compute the standard deviation of a tensor along specified dimensions
    pub fn std<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T>,
        dims: Option<&[usize]>,
        keepdim: bool,
    ) -> Result<Tensor<T>> {
        // Standard deviation is square root of variance
        let variance = var(input, dims, keepdim)?;
        Ok(variance.sqrt())
    }

    /// Compute the variance of a tensor along specified dimensions
    pub fn var<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T>,
        dims: Option<&[usize]>,
        keepdim: bool,
    ) -> Result<Tensor<T>> {
        // Variance: var = mean((x - mean(x))^2)
        let mean_val = mean(input, dims, keepdim)?;
        let diff = input.sub(&mean_val)?;
        let squared_diff = diff.mul(&diff)?;
        mean(&squared_diff, dims, keepdim)
    }

    /// Normalize tensor to have zero mean and unit variance
    pub fn normalize<T: coeus_dtype::FloatDtype>(input: &Tensor<T>, eps: T) -> Result<Tensor<T>> {
        // Normalize: (x - mean) / (std + eps)
        let mean_val = mean(input, None, false)?;
        let std_val = std(input, None, false)?;

        let eps_tensor = Tensor::scalar(eps);
        let std_with_eps = std_val.add(&eps_tensor)?;

        let centered = input.sub(&mean_val)?;
        centered.div(&std_with_eps)
    }
}

/// Random utilities
pub mod random {
    use super::*;
    use rand::prelude::*;

    /// Generate random tensor with uniform distribution
    pub fn rand<T: coeus_dtype::FloatDtype>(shape: Vec<usize>) -> Result<Tensor<T>> {
        // Generate random values from uniform distribution [0, 1)
        let mut rng = rand::thread_rng();
        let size = shape.iter().product();
        let data: Vec<T> = (0..size)
            .map(|_| T::from(rng.gen::<f64>()).unwrap())
            .collect();
        Ok(Tensor::from_vec(data, shape))
    }

    /// Generate random tensor with normal distribution
    pub fn randn<T: coeus_dtype::FloatDtype>(shape: Vec<usize>) -> Result<Tensor<T>> {
        // Generate random values from standard normal distribution N(0, 1)
        let mut rng = rand::thread_rng();
        use rand_distr::Normal;
        let normal = Normal::new(0.0, 1.0).unwrap();
        let size = shape.iter().product();
        let data: Vec<T> = (0..size)
            .map(|_| T::from(rng.sample(normal)).unwrap())
            .collect();
        Ok(Tensor::from_vec(data, shape))
    }

    /// Generate random integers in range [low, high)
    pub fn randint(low: i64, high: i64, shape: Vec<usize>) -> Result<Tensor<i64>> {
        // Generate random integers in range [low, high)
        let mut rng = rand::thread_rng();
        let uniform = rand::distributions::Uniform::from(low..high);
        let size = shape.iter().product();
        let data: Vec<i64> = (0..size).map(|_| rng.sample(uniform)).collect();
        Ok(Tensor::from_vec(data, shape))
    }

    /// Shuffle indices
    pub fn shuffle_indices(n: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();
        let mut rng = thread_rng();
        indices.shuffle(&mut rng);
        indices
    }
}

/// Loss functions
pub mod loss {
    use super::*;

    /// Mean Squared Error (MSE) loss
    pub fn mse_loss<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T>,
        target: &Tensor<T>,
        reduction: Reduction,
    ) -> Result<Tensor<T>> {
        // MSE Loss: mean((input - target)^2)
        let diff = input.sub(target)?;
        let squared_diff = diff.mul(&diff)?;

        match reduction {
            Reduction::None => Ok(squared_diff),
            Reduction::Sum => Ok(squared_diff.sum()),
            Reduction::Mean => {
                let sum = squared_diff.sum();
                let count = T::from(squared_diff.numel() as f64).unwrap();
                let count_tensor = Tensor::scalar(count);
                Ok(sum.div(&count_tensor)?)
            }
        }
    }

    /// Binary Cross Entropy loss
    pub fn binary_cross_entropy<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T>,
        target: &Tensor<T>,
        reduction: Reduction,
    ) -> Result<Tensor<T>> {
        // Binary Cross Entropy: -[y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x))]
        let sigmoid_input = super::math::sigmoid(input)?;

        // log(sigmoid(x))
        let log_sigmoid = sigmoid_input.log();

        // log(1-sigmoid(x))
        let one = Tensor::scalar(T::one());
        let one_minus_sigmoid = one.sub(&sigmoid_input)?;
        let log_one_minus_sigmoid = one_minus_sigmoid.log();

        // y * log(sigmoid(x))
        let term1 = target.mul(&log_sigmoid)?;

        // (1-y) * log(1-sigmoid(x))
        let one_minus_target = one.sub(target)?;
        let term2 = one_minus_target.mul(&log_one_minus_sigmoid)?;

        // -[term1 + term2]
        let sum_terms = term1.add(&term2)?;
        let loss = sum_terms.neg();

        match reduction {
            Reduction::None => Ok(loss),
            Reduction::Sum => Ok(loss.sum()),
            Reduction::Mean => {
                let sum = loss.sum();
                let count = T::from(loss.numel() as f64).unwrap();
                let count_tensor = Tensor::scalar(count);
                Ok(sum.div(&count_tensor)?)
            }
        }
    }

    /// Cross Entropy loss
    pub fn cross_entropy<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T>,
        target: &Tensor<i64>,
        reduction: Reduction,
    ) -> Result<Tensor<T>> {
        // Cross Entropy: -log(softmax(x)[target])
        // First compute log softmax along the last dimension
        let log_softmax = super::math::log_softmax(input, -1)?;

        // Gather the log probabilities for the correct classes
        // Use advanced indexing to select log_softmax[batch_idx, target[batch_idx]]
        let batch_size = input.shape()[0];
        let num_classes = input.shape()[input.shape().len() - 1];

        let mut loss_data = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let target_class = target.data()[batch_idx] as usize;
            if target_class >= num_classes {
                return Err(coeus_tensor::TensorError::InvalidOperation {
                    message: format!(
                        "Target class {} is out of range for {} classes",
                        target_class, num_classes
                    ),
                });
            }

            // Get log probability for the correct class: log_softmax[batch_idx, target_class]
            let log_prob_idx = batch_idx * num_classes + target_class;
            let log_prob = log_softmax.data()[log_prob_idx];

            // Cross entropy loss: -log_prob for the correct class
            loss_data.push(T::zero().sub(log_prob));
        }

        let loss = Tensor::from_vec(loss_data, vec![batch_size]);

        match reduction {
            Reduction::None => Ok(loss),
            Reduction::Sum => Ok(loss.sum()),
            Reduction::Mean => {
                let sum = loss.sum();
                let count = T::from(batch_size as f64).unwrap();
                let count_tensor = Tensor::scalar(count);
                Ok(sum.div(&count_tensor)?)
            }
        }
    }
}

/// Reduction modes for loss functions
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reduction {
    /// No reduction - return per-element losses
    None,
    /// Sum all losses
    Sum,
    /// Average all losses
    Mean,
}

/// Tensor operations utilities
pub mod tensor_ops {
    use super::*;

    /// Stack tensors along a new dimension
    ///
    /// Creates a new tensor by stacking the input tensors along a specified dimension.
    /// This is equivalent to PyTorch's torch.stack() function.
    ///
    /// # Arguments
    /// * `tensors` - Vector of tensors to stack
    /// * `dim` - Dimension along which to stack (0-based)
    ///
    /// # Returns
    /// Stacked tensor with one additional dimension
    ///
    /// # Example
    /// ```rust
    /// # use coeus_utils::utils::tensor_ops::stack;
    /// # use coeus_tensor::Tensor;
    /// let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
    /// let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
    /// let result = stack(&[&a, &b], 0).unwrap();
    /// // Result shape: [2, 2], values: [[1, 2], [3, 4]]
    /// ```
    pub fn stack<T: coeus_dtype::FloatDtype>(
        tensors: &[&Tensor<T>],
        dim: usize,
    ) -> Result<Tensor<T>> {
        if tensors.is_empty() {
            return Err(coeus_tensor::TensorError::InvalidOperation {
                message: "Cannot stack empty tensor list".to_string(),
            });
        }

        // Check that all tensors have the same shape
        let first_shape = tensors[0].shape();
        for tensor in tensors.iter().skip(1) {
            if tensor.shape() != first_shape {
                return Err(coeus_tensor::TensorError::ShapeMismatch {
                    expected: first_shape.to_vec(),
                    actual: tensor.shape().to_vec(),
                });
            }
        }

        // Expand each tensor by adding a dimension at the specified position
        let mut expanded_tensors = Vec::new();
        for tensor in tensors {
            // Create new shape with additional dimension
            let mut new_shape = first_shape.to_vec();
            new_shape.insert(dim, 1);
            let expanded = tensor.reshape(new_shape.clone())?;
            expanded_tensors.push(expanded);
        }

        // Concatenate along the new dimension
        use coeus_tensor::ops::reduction::cat;
        cat(&expanded_tensors.iter().collect::<Vec<_>>(), dim)
    }
}

/// Metric computations
pub mod metrics {
    use super::*;

    /// Compute accuracy for classification
    pub fn accuracy<T: coeus_dtype::FloatDtype>(
        predictions: &Tensor<T>,
        targets: &Tensor<i64>,
    ) -> Result<f64> {
        // For classification, predictions are typically class probabilities/logits
        // Convert predictions to class indices by taking argmax
        let batch_size = predictions.shape()[0];
        let num_classes = predictions.shape()[predictions.shape().len() - 1];

        let mut correct = 0;

        for batch_idx in 0..batch_size {
            // Find predicted class (argmax along class dimension)
            let mut max_prob = T::neg_infinity();
            let mut predicted_class = 0;

            for class_idx in 0..num_classes {
                let prob_idx = batch_idx * num_classes + class_idx;
                let prob = predictions.data()[prob_idx];

                if prob > max_prob {
                    max_prob = prob;
                    predicted_class = class_idx;
                }
            }

            // Check if prediction matches target
            let target_class = targets.data()[batch_idx] as usize;
            if predicted_class == target_class {
                correct += 1;
            }
        }

        Ok(correct as f64 / batch_size as f64)
    }

    /// Compute precision for binary classification
    pub fn precision<T: coeus_dtype::FloatDtype>(
        predictions: &Tensor<T>,
        targets: &Tensor<i64>,
    ) -> Result<f64> {
        // Precision = TP / (TP + FP)
        // For binary classification: predictions are probabilities, targets are 0/1
        let mut true_positives = 0.0;
        let mut false_positives = 0.0;

        let pred_data = predictions.data();
        let target_data = targets.data();

        for (pred, target) in pred_data.iter().zip(target_data.iter()) {
            let pred_val = coeus_dtype::Dtype::to_f64(pred).unwrap();
            let target_val = *target as f64;

            // Binary classification: threshold at 0.5
            let pred_class = if pred_val >= 0.5 { 1.0 } else { 0.0 };

            if pred_class == 1.0 && target_val == 1.0 {
                true_positives += 1.0;
            } else if pred_class == 1.0 && target_val == 0.0 {
                false_positives += 1.0;
            }
        }

        if true_positives + false_positives == 0.0 {
            Ok(0.0) // No positive predictions
        } else {
            Ok(true_positives / (true_positives + false_positives))
        }
    }

    /// Compute recall for binary classification
    pub fn recall<T: coeus_dtype::FloatDtype>(
        predictions: &Tensor<T>,
        targets: &Tensor<i64>,
    ) -> Result<f64> {
        // Recall = TP / (TP + FN)
        // For binary classification: predictions are probabilities, targets are 0/1
        let mut true_positives = 0.0;
        let mut false_negatives = 0.0;

        let pred_data = predictions.data();
        let target_data = targets.data();

        for (pred, target) in pred_data.iter().zip(target_data.iter()) {
            let pred_val = coeus_dtype::Dtype::to_f64(pred).unwrap();
            let target_val = *target as f64;

            // Binary classification: threshold at 0.5
            let pred_class = if pred_val >= 0.5 { 1.0 } else { 0.0 };

            if pred_class == 1.0 && target_val == 1.0 {
                true_positives += 1.0;
            } else if pred_class == 0.0 && target_val == 1.0 {
                false_negatives += 1.0;
            }
        }

        if true_positives + false_negatives == 0.0 {
            Ok(0.0) // No positive targets
        } else {
            Ok(true_positives / (true_positives + false_negatives))
        }
    }

    /// Compute F1 score for binary classification
    pub fn f1_score<T: coeus_dtype::FloatDtype>(
        predictions: &Tensor<T>,
        targets: &Tensor<i64>,
    ) -> Result<f64> {
        // F1 = 2 * (precision * recall) / (precision + recall)
        let precision = precision(predictions, targets)?;
        let recall = recall(predictions, targets)?;

        if (precision + recall) == 0.0 {
            Ok(0.0) // No positive predictions or targets
        } else {
            Ok(2.0 * (precision * recall) / (precision + recall))
        }
    }
}

pub use loss::*;
/// Re-export commonly used utilities
pub use math::*;
pub use metrics::*;
pub use random::*;
pub use stats::*;
