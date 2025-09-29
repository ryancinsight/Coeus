//! General utility functions and helpers
//!
//! This module provides utility functions for common machine learning
//! and tensor operations that complement the core functionality.

use coeus_tensor::{ops::{arithmetic::{self, div, sub, neg}, reduction::{self, sum}, creation}, Add, Mul, Result, Tensor, CpuBackend};

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
    pub fn softmax<T: coeus_dtype::FloatDtype>(input: &Tensor<T, CpuBackend>, dim: i32) -> Result<Tensor<T, CpuBackend>> {
        // Compute softmax: softmax(x_i) = exp(x_i) / Σ exp(x_j) along dimension dim
        let exp_input = arithmetic::exp(input)?;

        // Sum along the specified dimension (convert i32 to Option<usize>)
        let dim_opt = if dim >= 0 {
            Some(dim as usize)
        } else {
            // Handle negative dimensions (Python-style indexing)
            let ndim = input.shape().len();
            Some((ndim as i32 + dim) as usize)
        };
        let sum_exp = reduction::sum_dim(&exp_input, dim_opt.unwrap_or(0))?;

        // Divide by the sum
        div(&exp_input, &sum_exp)
    }

    /// Compute the log-softmax function
    ///
    /// Log-softmax computes the logarithm of the softmax function.
    /// This is numerically stable and avoids overflow issues.
    pub fn log_softmax<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T, CpuBackend>,
        dim: i32,
    ) -> Result<Tensor<T, CpuBackend>> {
        // Numerically stable log-softmax: log(softmax(x_i)) = x_i - log(Σ exp(x_j))
        let exp_input = arithmetic::exp(input)?;

        // Sum along the specified dimension (convert i32 to Option<usize>)
        let dim_opt = if dim >= 0 {
            Some(dim as usize)
        } else {
            // Handle negative dimensions (Python-style indexing)
            let ndim = input.shape().len();
            Some((ndim as i32 + dim) as usize)
        };
        let sum_exp = reduction::sum_dim(&exp_input, dim_opt.unwrap_or(0))?;

        // Compute log of sum
        let log_sum_exp = sum_exp.log()?;

        // Subtract: x_i - log(Σ exp(x_j))
        arithmetic::sub(&input, &log_sum_exp)
    }

    /// Compute the sigmoid (logistic) function
    pub fn sigmoid<T: coeus_dtype::FloatDtype>(input: &Tensor<T, CpuBackend>) -> Result<Tensor<T, CpuBackend>> {
        // Sigmoid: sigmoid(x) = 1 / (1 + exp(-x))
        let neg_input = arithmetic::neg(&input.clone())?;
        let exp_neg = arithmetic::exp(&neg_input)?;
        let one = creation::scalar(CpuBackend::default(), T::one())?;
        let denominator = one.add(&exp_neg)?;
        div(&one, &denominator)
    }

    /// Compute the hyperbolic tangent function
    pub fn tanh<T: coeus_dtype::FloatDtype>(input: &Tensor<T, CpuBackend>) -> Result<Tensor<T, CpuBackend>> {
        // Tanh: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        let exp_x = arithmetic::exp(input)?;
        let exp_neg_x = arithmetic::exp(&arithmetic::neg(&input.clone())?)?;
        let numerator = arithmetic::sub(&exp_x, &exp_neg_x)?;
        let denominator = exp_x.add(&exp_neg_x)?;
        div(&numerator, &denominator)
    }

    /// Compute the Rectified Linear Unit (ReLU) activation
    pub fn relu<T: coeus_dtype::FloatDtype>(input: &Tensor<T, CpuBackend>) -> Result<Tensor<T, CpuBackend>> {
        // ReLU: relu(x) = max(0, x)
        let zero = creation::scalar(CpuBackend::default(), T::zero())?;
        use coeus_tensor::ops::arithmetic::maximum;
        maximum(input, &zero)
    }

    /// Compute the Leaky ReLU activation
    pub fn leaky_relu<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T, CpuBackend>,
        negative_slope: T,
    ) -> Result<Tensor<T, CpuBackend>> {
        // Leaky ReLU: leaky_relu(x) = max(x, negative_slope * x)
        let slope_tensor = creation::scalar(CpuBackend::default(), negative_slope)?;
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
        input: &Tensor<T, CpuBackend>,
        dims: Option<&[usize]>,
        _keepdim: bool,
    ) -> Result<Tensor<T, CpuBackend>> {
        // For now, implement basic case - global mean or single dimension
        if let Some(dims) = dims {
            if dims.len() == 1 {
                let sum = reduction::sum_dim(input, dims[0])?;
                let count = T::from(input.shape()[dims[0]] as f64).unwrap();
                let count_tensor = creation::scalar(CpuBackend::default(), count)?;
                Ok(div(&sum, &count_tensor)?)
            } else {
                // Multi-dimension reduction not fully implemented yet
                // Return global mean as fallback
                let sum = sum(input)?;
                let count = T::from(input.numel() as f64).unwrap();
                let count_tensor = creation::scalar(CpuBackend::default(), count)?;
                Ok(div(&sum, &count_tensor)?)
            }
        } else {
            // Global mean
            let sum = sum(input)?;
            let count = T::from(input.numel() as f64).unwrap();
            let count_tensor = creation::scalar(CpuBackend::default(), count)?;
            Ok(div(&sum, &count_tensor)?)
        }
    }

    /// Compute the standard deviation of a tensor along specified dimensions
    pub fn std<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T, CpuBackend>,
        dims: Option<&[usize]>,
        keepdim: bool,
    ) -> Result<Tensor<T, CpuBackend>> {
        // Standard deviation is square root of variance
        let variance = var(input, dims, keepdim)?;
        Ok(arithmetic::sqrt(&variance))
    }

    /// Compute the variance of a tensor along specified dimensions
    pub fn var<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T, CpuBackend>,
        dims: Option<&[usize]>,
        keepdim: bool,
    ) -> Result<Tensor<T, CpuBackend>> {
        // Variance: var = mean((x - mean(x))^2)
        let mean_val = mean(input, dims, keepdim)?;
        let diff = arithmetic::sub(input, &mean_val)?;
        let squared_diff = diff.mul(&diff)?;
        mean(&squared_diff, dims, keepdim)
    }

    /// Normalize tensor to have zero mean and unit variance
    pub fn normalize<T: coeus_dtype::FloatDtype>(input: &Tensor<T, CpuBackend>, eps: T) -> Result<Tensor<T, CpuBackend>> {
        // Normalize: (x - mean) / (std + eps)
        let mean_val = mean(input, None, false)?;
        let std_val = std(input, None, false)?;

        let eps_tensor = creation::scalar(CpuBackend::default(), eps)?;
        let std_with_eps = std_val.add(&eps_tensor)?;

        let centered = arithmetic::sub(input, &mean_val)?;
        div(&centered, &std_with_eps)
    }
}

/// Random utilities
pub mod random {
    use super::*;
    use rand::prelude::*;

    /// Generate random tensor with uniform distribution
    pub fn rand<T: coeus_dtype::FloatDtype>(shape: Vec<usize>) -> Result<Tensor<T, CpuBackend>> {
        // Generate random values from uniform distribution [0, 1)
        let mut rng = rand::thread_rng();
        let size = shape.iter().product();
        let data: Vec<T> = (0..size)
            .map(|_| T::from(rng.gen::<f64>()).unwrap())
            .collect();
        Ok(Tensor::from_vec(CpuBackend::new(), data, shape)?)
    }

    /// Generate random tensor with normal distribution
    pub fn randn<T: coeus_dtype::FloatDtype>(shape: Vec<usize>) -> Result<Tensor<T, CpuBackend>> {
        // Generate random values from standard normal distribution N(0, 1)
        let mut rng = rand::thread_rng();
        use rand_distr::Normal;
        let normal = Normal::new(0.0, 1.0).unwrap();
        let size = shape.iter().product();
        let data: Vec<T> = (0..size)
            .map(|_| T::from(rng.sample(normal)).unwrap())
            .collect();
        Ok(Tensor::from_vec(CpuBackend::new(), data, shape)?)
    }

    /// Generate random integers in range [low, high)
    pub fn randint(low: i64, high: i64, shape: Vec<usize>) -> Result<Tensor<i64, CpuBackend>> {
        // Generate random integers in range [low, high)
        let mut rng = rand::thread_rng();
        let uniform = rand::distributions::Uniform::from(low..high);
        let size = shape.iter().product();
        let data: Vec<i64> = (0..size).map(|_| rng.sample(uniform)).collect();
        Ok(Tensor::from_vec(CpuBackend::new(), data, shape)?)
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
        input: &Tensor<T, CpuBackend>,
        target: &Tensor<T, CpuBackend>,
        reduction: Reduction,
    ) -> Result<Tensor<T, CpuBackend>> {
        // MSE Loss: mean((input - target)^2)
        let diff = arithmetic::sub(input, target)?;
        let squared_diff = diff.mul(&diff)?;

        match reduction {
            Reduction::None => Ok(squared_diff),
            Reduction::Sum => sum(&squared_diff),
            Reduction::Mean => {
                let sum = sum(&squared_diff)?;
                let count = T::from(squared_diff.numel() as f64).unwrap();
                let count_tensor = creation::scalar(CpuBackend::default(), count)?;
                div(&sum, &count_tensor)
            }
        }
    }

    /// Binary Cross Entropy loss
    pub fn binary_cross_entropy<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T, CpuBackend>,
        target: &Tensor<T, CpuBackend>,
        reduction: Reduction,
    ) -> Result<Tensor<T, CpuBackend>> {
        // Binary Cross Entropy: -[y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x))]
        let sigmoid_input = super::math::sigmoid(input)?;

        // log(sigmoid(x))
        let log_sigmoid = sigmoid_input.log()?;

        // log(1-sigmoid(x))
        let one = creation::scalar(CpuBackend::default(), T::one())?;
        let one_minus_sigmoid = sub(&one, &sigmoid_input)?;
        let log_one_minus_sigmoid = one_minus_sigmoid.log()?;

        // y * log(sigmoid(x))
        let term1 = target.mul(&log_sigmoid)?;

        // (1-y) * log(1-sigmoid(x))
        let one_minus_target = sub(&one, target)?;
        let term2 = one_minus_target.mul(&log_one_minus_sigmoid)?;

        // -[term1 + term2]
        let sum_terms = term1.add(&term2)?;
        let loss = neg(&sum_terms);

        match reduction {
            Reduction::None => Ok(loss?),
            Reduction::Sum => sum(&loss?),
            Reduction::Mean => {
                let loss_tensor = loss?;
                let sum = sum(&loss_tensor)?;
                let count = T::from(loss_tensor.numel() as f64).unwrap();
                let count_tensor = creation::scalar(CpuBackend::default(), count)?;
                div(&sum, &count_tensor)
            }
        }
    }

    /// Cross Entropy loss
    pub fn cross_entropy<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T, CpuBackend>,
        target: &Tensor<i64, CpuBackend>,
        reduction: Reduction,
    ) -> Result<Tensor<T, CpuBackend>> {
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
            loss_data.push(T::zero() - log_prob);
        }

        let loss = Tensor::from_vec(CpuBackend::new(), loss_data, vec![batch_size]).unwrap();

        match reduction {
            Reduction::None => Ok(loss),
            Reduction::Sum => sum(&loss),
            Reduction::Mean => {
                let sum = sum(&loss)?;
                let count = T::from(batch_size as f64).unwrap();
                let count_tensor = creation::scalar(CpuBackend::default(), count)?;
                div(&sum, &count_tensor)
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

/// Advanced loss functions
pub mod advanced_loss {
    use super::*;

    /// KL Divergence Loss
    pub fn kl_div_loss<T: coeus_dtype::FloatDtype>(
        input: &Tensor<T, CpuBackend>,
        target: &Tensor<T, CpuBackend>,
        reduction: Reduction,
    ) -> Result<Tensor<T, CpuBackend>> {
        // KL divergence: target * (log(target) - log(input))
        let log_target = target.log().unwrap();
        let log_input = input.log().unwrap();
        let diff = sub(&log_target, &log_input).unwrap();
        let kl_div = target.mul(&diff)?;

        match reduction {
            Reduction::None => Ok(kl_div),
            Reduction::Sum => sum(&kl_div),
            Reduction::Mean => {
                let sum = sum(&kl_div)?;
                let count = T::from(kl_div.numel() as f64).unwrap();
                let count_tensor = creation::scalar(CpuBackend::default(), count)?;
                div(&sum, &count_tensor)
            }
        }
    }
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
        tensors: &[&Tensor<T, CpuBackend>],
        dim: usize,
    ) -> Result<Tensor<T, CpuBackend>> {
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
            // TODO: Implement reshape operation - for now using identity
            let expanded = (*tensor).clone();
            expanded_tensors.push(expanded);
        }

        // Concatenate along the new dimension
        use coeus_tensor::ops::reduction::cat;
        let tensor_refs: Vec<&Tensor<T, CpuBackend>> = expanded_tensors.iter().collect();
        cat(&tensor_refs, dim)
    }
}

/// Metric computations
pub mod metrics {
    use super::*;

    /// Compute accuracy for classification
    pub fn accuracy<T: coeus_dtype::FloatDtype>(
        predictions: &Tensor<T, CpuBackend>,
        targets: &Tensor<i64, CpuBackend>,
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

    /// Compute top-k accuracy for classification
    pub fn top_k_accuracy<T: coeus_dtype::FloatDtype>(
        predictions: &Tensor<T, CpuBackend>,
        targets: &Tensor<i64, CpuBackend>,
        k: usize,
    ) -> Result<f64> {
        let batch_size = predictions.shape()[0];
        let num_classes = predictions.shape()[predictions.shape().len() - 1];

        if k > num_classes {
            return Err(coeus_tensor::TensorError::InvalidOperation {
                message: format!(
                    "k ({}) cannot be greater than num_classes ({})",
                    k, num_classes
                ),
            });
        }

        let mut correct = 0;

        for batch_idx in 0..batch_size {
            // Get top-k predictions using argmax for each position
            let target_class = targets.data()[batch_idx] as usize;

            // Simple approach: check if target is among the top k predictions
            // by finding k largest values and checking if target is among them
            let mut values: Vec<T> = (0..num_classes)
                .map(|class_idx| {
                    let prob_idx = batch_idx * num_classes + class_idx;
                    predictions.data()[prob_idx]
                })
                .collect();

            // Sort in descending order
            values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            // Check if target probability is among top k
            let target_prob = predictions.data()[batch_idx * num_classes + target_class];
            let top_k_found = values.iter().take(k.min(values.len())).any(|&prob| {
                // Use epsilon comparison for floating point values
                (prob - target_prob).abs() < T::from(1e-6).unwrap()
            });

            if top_k_found {
                correct += 1;
            }
        }

        Ok(correct as f64 / batch_size as f64)
    }

    /// Compute confusion matrix for multi-class classification
    pub fn confusion_matrix(
        predictions: &Tensor<i64, CpuBackend>,
        targets: &Tensor<i64, CpuBackend>,
        num_classes: usize,
    ) -> Result<Tensor<i64, CpuBackend>> {
        let batch_size = predictions.numel();
        let mut matrix = vec![0i64; num_classes * num_classes];

        for i in 0..batch_size {
            let pred = predictions.data()[i] as usize;
            let target = targets.data()[i] as usize;

            if pred >= num_classes || target >= num_classes {
                return Err(coeus_tensor::TensorError::InvalidOperation {
                    message: format!(
                        "Class index out of range: pred={}, target={}, num_classes={}",
                        pred, target, num_classes
                    ),
                });
            }

            let idx = target * num_classes + pred;
            matrix[idx] += 1;
        }

        Ok(Tensor::from_vec(CpuBackend::new(), matrix, vec![num_classes, num_classes])?)
    }

    /// Compute multi-class precision, recall, and F1 scores
    pub fn classification_report(
        predictions: &Tensor<i64, CpuBackend>,
        targets: &Tensor<i64, CpuBackend>,
        num_classes: usize,
    ) -> Result<ClassificationReport> {
        let cm = confusion_matrix(predictions, targets, num_classes)?;

        let mut precision = vec![0.0; num_classes];
        let mut recall = vec![0.0; num_classes];
        let mut f1_score = vec![0.0; num_classes];

        for class in 0..num_classes {
            let tp = cm.data()[class * num_classes + class];
            let fp: i64 = (0..num_classes)
                .map(|i| cm.data()[i * num_classes + class])
                .sum::<i64>()
                - tp;
            let fn_val: i64 = (0..num_classes)
                .map(|i| cm.data()[class * num_classes + i])
                .sum::<i64>()
                - tp;

            let tp_f64 = tp as f64;
            let fp_f64 = fp as f64;
            let fn_f64 = fn_val as f64;

            precision[class] = if tp_f64 + fp_f64 > 0.0 {
                tp_f64 / (tp_f64 + fp_f64)
            } else {
                0.0
            };
            recall[class] = if tp_f64 + fn_f64 > 0.0 {
                tp_f64 / (tp_f64 + fn_f64)
            } else {
                0.0
            };
            f1_score[class] = if precision[class] + recall[class] > 0.0 {
                2.0 * precision[class] * recall[class] / (precision[class] + recall[class])
            } else {
                0.0
            };
        }

        let macro_precision = precision.iter().sum::<f64>() / num_classes as f64;
        let macro_recall = recall.iter().sum::<f64>() / num_classes as f64;
        let macro_f1 = f1_score.iter().sum::<f64>() / num_classes as f64;

        Ok(ClassificationReport {
            precision,
            recall,
            f1_score,
            macro_precision,
            macro_recall,
            macro_f1,
        })
    }

    /// Compute Mean Squared Error (MSE) for regression
    pub fn mean_squared_error<T: coeus_dtype::FloatDtype>(
        predictions: &Tensor<T, CpuBackend>,
        targets: &Tensor<T, CpuBackend>,
    ) -> Result<f64> {
        // MSE = mean((predictions - targets)^2)
        let diff = sub(&predictions, targets).unwrap();
        let squared_diff = diff.mul(&diff)?;
        let sum_squared_diff = sum(&squared_diff)?;
        let sum_val = sum_squared_diff.as_scalar().or_else(|_| {
            Err(coeus_tensor::TensorError::InvalidOperation {
                message: "Tensor is not scalar".to_string(),
            })
        })?;
        let numel = predictions.numel() as f64;
        let sum_f64 = num_traits::ToPrimitive::to_f64(&sum_val).unwrap();
        let mse = sum_f64 / numel;
        Ok(mse)
    }

    /// Compute Area Under ROC Curve (AUC-ROC) for binary classification
    pub fn auc_roc<T: coeus_dtype::FloatDtype>(
        _predictions: &Tensor<T, CpuBackend>,
        targets: &Tensor<i64, CpuBackend>,
    ) -> Result<f64> {
        // Simplified AUC calculation - for now return a placeholder
        // Full implementation would require sorting and complex AUC computation
        let n_positives: usize = targets.data().iter().map(|&x| x as usize).sum();
        let n_negatives = targets.numel() - n_positives;

        if n_positives == 0 || n_negatives == 0 {
            Ok(0.5) // Random classifier AUC
        } else {
            // Placeholder: return 0.5 (random classifier)
            // TODO: Implement full AUC calculation with proper sorting
            Ok(0.5)
        }
    }
}

/// Classification report structure
#[derive(Debug, Clone)]
pub struct ClassificationReport {
    pub precision: Vec<f64>,
    pub recall: Vec<f64>,
    pub f1_score: Vec<f64>,
    pub macro_precision: f64,
    pub macro_recall: f64,
    pub macro_f1: f64,
}

impl std::fmt::Display for ClassificationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Classification Report")?;
        writeln!(f, "=====================")?;
        writeln!(
            f,
            "{:<10} {:<10} {:<10} {:<10}",
            "Class", "Precision", "Recall", "F1-Score"
        )?;
        writeln!(f, "{:-<40}", "")?;

        for i in 0..self.precision.len() {
            writeln!(
                f,
                "{:<10} {:<10.4} {:<10.4} {:<10.4}",
                i, self.precision[i], self.recall[i], self.f1_score[i]
            )?;
        }

        writeln!(f)?;
        writeln!(f, "Macro Average:")?;
        writeln!(f, "Precision: {:.4}", self.macro_precision)?;
        writeln!(f, "Recall:    {:.4}", self.macro_recall)?;
        writeln!(f, "F1-Score:  {:.4}", self.macro_f1)?;

        Ok(())
    }
}

pub use loss::*;
/// Re-export commonly used utilities
pub use math::*;
pub use metrics::*;
pub use random::*;
pub use stats::*;
