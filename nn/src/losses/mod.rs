//! Loss functions for neural networks
//!
//! This module provides common loss functions used in machine learning,
//! organized by category for better maintainability and discoverability.
//!
//! ## Categories
//!
//! - **Regression**: Loss functions for continuous value prediction
//! - **Classification**: Loss functions for discrete class prediction  
//! - **Ranking**: Loss functions for learning relative ordering
//! - **Distribution**: Loss functions for probability distribution matching
//! - **Robust**: Loss functions robust to outliers
//! - **Specialized**: Domain-specific loss functions (computer vision, NLP, etc.)

pub mod classification;
pub mod distribution;
pub mod ranking;
pub mod regression;
pub mod robust;
pub mod specialized;

// Re-export all loss functions to maintain API compatibility
#[allow(unused_imports)]
pub use classification::*;
#[allow(unused_imports)]
pub use distribution::*;
#[allow(unused_imports)]
pub use ranking::*;
#[allow(unused_imports)]
pub use regression::*;
#[allow(unused_imports)]
pub use robust::*;
#[allow(unused_imports)]
pub use specialized::*;

// Re-export specific loss functions for convenience
pub use classification::{BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, NLLLoss};
pub use ranking::{
    HingeEmbeddingLoss, MarginRankingLoss, MultiLabelMarginLoss, MultiMarginLoss, TripletMarginLoss,
};
pub use regression::{CosineEmbeddingLoss, MaeLoss, MseLoss};
pub use specialized::{
    CTCLoss, DiceLoss, FocalLoss, GaussianNLLLoss, IoULoss, PoissonNLLLoss, SoftMarginLoss,
};

use crate::{Module, NNError, Result};
use coeus_tensor::{FloatDtype, Tensor};
use std::ops::Div;

/// Reduction modes for loss functions
///
/// Specifies how to reduce the loss across batch dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Reduction {
    /// No reduction - return loss for each element
    None,
    /// Sum all losses
    Sum,
    /// Mean of all losses (default for most losses)
    #[default]
    Mean,
}

/// Common utilities for loss function implementations
pub(crate) mod utils {
    use super::*;

    /// Apply reduction to a tensor of losses
    pub fn apply_reduction<T: FloatDtype>(
        losses: &Tensor<T>,
        reduction: Reduction,
    ) -> crate::Result<Tensor<T>> {
        match reduction {
            Reduction::None => Ok(losses.clone()),
            Reduction::Sum => Ok(losses.sum()),
            Reduction::Mean => {
                let sum = losses.sum();
                let count = T::from(losses.numel() as f64).unwrap();
                let count_tensor = Tensor::scalar(count);
                sum.div(&count_tensor).map_err(|_| NNError::ForwardError {
                    message: "Failed to compute mean reduction".to_string(),
                })
            }
        }
    }

    /// Compute numerically stable log-softmax
    pub fn log_softmax<T: FloatDtype>(x: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if x.ndim() != 2 {
            return Err(NNError::InvalidInput {
                message: "log_softmax requires 2D input (batch_size, num_classes)".to_string(),
            });
        }

        let batch_size = x.shape()[0];
        let num_classes = x.shape()[1];
        let mut max_vals = Vec::with_capacity(batch_size);

        // Find max values for numerical stability
        for i in 0..batch_size {
            let mut max_val = x.data()[i * num_classes];
            for j in 1..num_classes {
                if x.data()[i * num_classes + j] > max_val {
                    max_val = x.data()[i * num_classes + j];
                }
            }
            max_vals.push(max_val);
        }

        let mut log_probs = Vec::with_capacity(x.numel());

        // Compute log-softmax: log(exp(x - max) / sum(exp(x - max)))
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

    /// Compute numerically stable softmax
    pub fn softmax<T: FloatDtype>(x: &Tensor<T>) -> crate::Result<Tensor<T>> {
        if x.ndim() != 2 {
            return Err(NNError::InvalidInput {
                message: "softmax requires 2D input (batch_size, num_classes)".to_string(),
            });
        }

        let batch_size = x.shape()[0];
        let num_classes = x.shape()[1];
        let mut max_vals = Vec::with_capacity(batch_size);

        // Find max values for numerical stability
        for i in 0..batch_size {
            let mut max_val = x.data()[i * num_classes];
            for j in 1..num_classes {
                if x.data()[i * num_classes + j] > max_val {
                    max_val = x.data()[i * num_classes + j];
                }
            }
            max_vals.push(max_val);
        }

        let mut softmax_data = Vec::with_capacity(x.numel());

        // Compute softmax: exp(x - max) / sum(exp(x - max))
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

    /// Clamp values to avoid numerical instability
    pub fn clamp_for_log<T: FloatDtype>(value: T) -> T {
        let epsilon = T::from(1e-12).unwrap();
        if value < epsilon {
            epsilon
        } else {
            value
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduction_enum() {
        assert_eq!(Reduction::default(), Reduction::Mean);
    }

    #[test]
    fn test_utils_log_softmax() {
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = utils::log_softmax(&input).unwrap();

        // Check that each row sums to approximately 1 when exponentiated
        let batch_size = 2;
        let num_classes = 3;

        for i in 0..batch_size {
            let mut sum = 0.0f32;
            for j in 0..num_classes {
                sum += result.data()[i * num_classes + j].exp();
            }
            assert!((sum - 1.0f32).abs() < 1e-6);
        }
    }

    #[test]
    fn test_utils_softmax() {
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let result = utils::softmax(&input).unwrap();

        // Check that each row sums to 1
        let batch_size = 2;
        let num_classes = 3;

        for i in 0..batch_size {
            let mut sum = 0.0f32;
            for j in 0..num_classes {
                sum += result.data()[i * num_classes + j];
            }
            assert!((sum - 1.0f32).abs() < 1e-6);
        }
    }
}
