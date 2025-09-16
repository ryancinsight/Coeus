//! Dropout regularization
//!
//! This module provides dropout layers for regularization
//! during neural network training.
//!
//! ## Mathematical Foundation
//!
//! ### Dropout
//! ```math
//! yᵢ = {
//!     xᵢ / (1-p)    if random() > p
//!     0             otherwise
//! }
//! ```
//!
//! Where:
//! - `p` is the dropout probability
//! - During training: randomly zero out elements with probability `p`
//! - During evaluation: scale by `(1-p)` to maintain expected value
//!
//! ## References
//!
//! - [Srivastava et al., 2014 - Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html)
//! - [Deep Learning Book - Regularization](https://www.deeplearningbook.org/contents/regularization.html)

use crate::{Module, NNError, Result};
use coeus_tensor::{FloatDtype, Tensor};
use rand::prelude::*;

/// Dropout layer for regularization
///
/// Randomly zeros out elements during training to prevent overfitting.
/// During evaluation, scales the output by `(1-p)` to maintain expected values.
#[derive(Debug, Clone)]
pub struct Dropout<T: FloatDtype> {
    /// Dropout probability (probability of dropping an element)
    pub p: T,
    /// Whether the layer is in training mode
    training: bool,
    /// Random number generator
    rng: Option<StdRng>,
}

impl<T: FloatDtype> Dropout<T> {
    /// Create a new dropout layer
    ///
    /// # Arguments
    /// * `p` - Dropout probability (0.0 = no dropout, 1.0 = drop everything)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Dropout;
    ///
    /// let dropout = Dropout::new(0.5); // 50% dropout probability
    /// ```
    pub fn new(p: T) -> Self {
        Self {
            p,
            training: true,
            rng: Some(StdRng::from_entropy()),
        }
    }

    /// Set the training mode
    ///
    /// # Arguments
    /// * `training` - Whether the layer is in training mode
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Dropout;
    ///
    /// let mut dropout = Dropout::new(0.5);
    /// dropout.set_training(false); // Evaluation mode
    /// ```
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Apply dropout to input tensor
    ///
    /// # Arguments
    /// * `input` - Input tensor
    ///
    /// # Returns
    /// Output tensor after dropout
    ///
    /// # Errors
    /// Returns error if dropout operation fails
    pub fn forward_impl(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if !self.training {
            // During evaluation, just scale by (1-p)
            let scale = T::one() - self.p;
            return Ok(input.map(|x| *x * scale));
        }

        // During training, randomly drop elements
        let mut rng = self
            .rng
            .as_ref()
            .ok_or_else(|| crate::NNError::InvalidInput {
                message: "Dropout RNG not initialized".to_string(),
            })?
            .clone();
        let mut output_data = Vec::with_capacity(input.numel());

        for &val in input.data() {
            if rng.gen::<f64>()
                < num_traits::ToPrimitive::to_f64(&self.p).ok_or_else(|| {
                    crate::NNError::InvalidInput {
                        message: format!(
                            "Failed to convert dropout probability {:?} to f64",
                            self.p
                        ),
                    }
                })?
            {
                // Drop this element
                output_data.push(T::zero());
            } else {
                // Keep this element and scale by (1-p)
                let scale = T::one() - self.p;
                output_data.push(val * scale);
            }
        }

        Ok(Tensor::from_vec(output_data, input.shape().to_vec()))
    }
}

impl<T: FloatDtype> Module<T> for Dropout<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        self.forward_impl(input)
            .map_err(|e| crate::NNError::InvalidInput {
                message: format!("Dropout forward pass failed: {}", e),
            })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![] // Dropout has no learnable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![] // Dropout has no learnable parameters
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

/// 2D Dropout layer (spatial dropout)
///
/// Applies dropout to entire feature maps (channels) instead of individual elements.
/// This helps preserve spatial correlations in convolutional networks.
#[derive(Debug, Clone)]
pub struct Dropout2d<T: FloatDtype> {
    /// Dropout probability for feature maps
    pub p: T,
    /// Whether the layer is in training mode
    training: bool,
    /// Random number generator
    rng: Option<StdRng>,
}

impl<T: FloatDtype> Dropout2d<T> {
    /// Create a new 2D dropout layer
    ///
    /// # Arguments
    /// * `p` - Dropout probability for feature maps
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::Dropout2d;
    ///
    /// let dropout = Dropout2d::new(0.2); // 20% of feature maps will be dropped
    /// ```
    pub fn new(p: T) -> Self {
        Self {
            p,
            training: true,
            rng: Some(StdRng::from_entropy()),
        }
    }

    /// Set the training mode
    ///
    /// # Arguments
    /// * `training` - Whether the layer is in training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Apply 2D dropout to input tensor
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (batch_size, height, width, channels)
    ///
    /// # Returns
    /// Output tensor after 2D dropout
    ///
    /// # Errors
    /// Returns error if input shape is invalid or dropout operation fails
    pub fn forward_impl(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        if input.shape().len() != 4 {
            return Err(NNError::InvalidInput {
                message: "Dropout2d expects 4D input tensor (batch_size, height, width, channels)"
                    .to_string(),
            });
        }

        if !self.training {
            // During evaluation, just scale by (1-p)
            let scale = T::one() - self.p;
            return Ok(input.map(|x| *x * scale));
        }

        let batch_size = input.shape()[0];
        let height = input.shape()[1];
        let width = input.shape()[2];
        let channels = input.shape()[3];

        let mut rng = self
            .rng
            .as_ref()
            .ok_or_else(|| crate::NNError::InvalidInput {
                message: "Dropout2d RNG not initialized".to_string(),
            })?
            .clone();
        let mut output_data = input.data().to_vec();

        // For each channel, decide whether to drop the entire feature map
        for c in 0..channels {
            if rng.gen::<f64>()
                < num_traits::ToPrimitive::to_f64(&self.p).ok_or_else(|| {
                    crate::NNError::InvalidInput {
                        message: format!(
                            "Failed to convert dropout2d probability {:?} to f64",
                            self.p
                        ),
                    }
                })?
            {
                // Drop this entire channel/feature map
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * height + h) * width + w) * channels + c;
                            output_data[idx] = T::zero();
                        }
                    }
                }
            } else {
                // Keep this channel and scale by (1-p)
                let scale = T::one() - self.p;
                for b in 0..batch_size {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = ((b * height + h) * width + w) * channels + c;
                            output_data[idx] = output_data[idx] * scale;
                        }
                    }
                }
            }
        }

        Ok(Tensor::from_vec(output_data, input.shape().to_vec()))
    }
}

impl<T: FloatDtype> Module<T> for Dropout2d<T> {
    fn forward(&self, input: &Tensor<T>) -> crate::Result<Tensor<T>> {
        self.forward_impl(input)
            .map_err(|e| crate::NNError::InvalidInput {
                message: format!("Dropout2d forward pass failed: {}", e),
            })
    }

    fn parameters(&self) -> Vec<&Tensor<T>> {
        vec![] // Dropout2d has no learnable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor<T>> {
        vec![] // Dropout2d has no learnable parameters
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dropout_creation() {
        let dropout = Dropout::new(0.5);
        assert_eq!(dropout.p, 0.5);
        assert!(dropout.training);
    }

    #[test]
    fn test_dropout_eval_mode() {
        let mut dropout = Dropout::new(0.5);
        dropout.set_training(false);

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let output = dropout
            .forward(&input)
            .expect("Dropout eval mode forward should succeed");

        // In eval mode, should scale by (1-p) = 0.5
        let expected = Tensor::from_vec(vec![0.5, 1.0, 1.5, 2.0], vec![4]);
        for i in 0..4 {
            assert_relative_eq!(output.data()[i], expected.data()[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dropout2d_creation() {
        let dropout = Dropout2d::new(0.2);
        assert_eq!(dropout.p, 0.2);
        assert!(dropout.training);
    }

    #[test]
    fn test_dropout2d_eval_mode() {
        let mut dropout = Dropout2d::new(0.5);
        dropout.set_training(false);

        // Input: 2x2x2 (batch_size=1, height=2, width=2, channels=2)
        let input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 2, 2, 2],
        );
        let output = dropout
            .forward(&input)
            .expect("Dropout2d eval mode forward should succeed");

        // In eval mode, should scale all elements by (1-p) = 0.5
        let expected = Tensor::from_vec(
            vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            vec![1, 2, 2, 2],
        );
        for i in 0..8 {
            assert_relative_eq!(output.data()[i], expected.data()[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dropout_no_parameters() {
        let dropout = Dropout::new(0.5);
        assert_eq!(dropout.parameters().len(), 0);

        let mut dropout_mut = Dropout::new(0.5);
        assert_eq!(dropout_mut.parameters_mut().len(), 0);
    }

    #[test]
    fn test_dropout2d_no_parameters() {
        let dropout = Dropout2d::new(0.2);
        assert_eq!(dropout.parameters().len(), 0);

        let mut dropout_mut = Dropout2d::new(0.2);
        assert_eq!(dropout_mut.parameters_mut().len(), 0);
    }

    #[test]
    fn test_dropout2d_invalid_input() {
        let dropout = Dropout2d::new(0.5);
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]); // 1D tensor

        let result = dropout.forward_impl(&input);
        assert!(result.is_err());
    }
}
