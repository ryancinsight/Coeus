//! Weight initialization methods
//!
//! This module provides various weight initialization techniques
//! for neural network parameters.
//!
//! ## Mathematical Foundation
//!
//! ### Xavier (Glorot) Initialization
//! ```math
//! W ~ U(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
//! ```
//!
//! ### Kaiming (He) Initialization
//! ```math
//! W ~ N(0, 2/n)  // for ReLU activation
//! W ~ N(0, 1/n)  // for other activations
//! ```
//!
//! ## References
//!
//! - [Glorot & Bengio, 2010 - Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html)
//! - [He et al., 2015 - Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

use crate::{NNError, Result};
use coeus_tensor::{Dtype, Tensor, CpuBackend};
use rand::prelude::*;

/// Helper function to create tensor from vec and return it directly
/// This unwraps the Result from Tensor::from_vec for convenience in initialization
pub(crate) fn tensor_from_vec_ok<T: Dtype>(backend: CpuBackend, data: Vec<T>, shape: Vec<usize>) -> crate::Result<Tensor<T, CpuBackend>> {
    // Propagate TensorError as NNError via `?` (NNError implements `#[from] TensorError`).
    let t = Tensor::from_vec(backend, data, shape)?;
    Ok(t)
}

/// Xavier (Glorot) weight initialization
///
/// Initializes weights using Xavier initialization, which is designed
/// to keep the variance of activations roughly the same across layers.
///
/// This initialization works well with sigmoid and tanh activations.
#[derive(Debug, Clone, Default)]
pub struct Xavier;

impl Xavier {
    /// Create a new Xavier initializer
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::init::Xavier;
    ///
    /// let init = Xavier::new();
    /// ```
    pub fn new() -> Self {
        Self
    }

    /// Initialize a tensor using Xavier initialization
    ///
    /// # Arguments
    /// * `shape` - Shape of the tensor to initialize
    ///
    /// # Returns
    /// Initialized tensor with Xavier initialization
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::init::Xavier;
    ///
    /// let init = Xavier::new();
    /// let weights = init.initialize::<f32>(&[64, 32]);
    /// assert_eq!(weights.unwrap().shape(), &[64, 32]);
    /// ```
    pub fn initialize<T: Dtype + num_traits::Float + num_traits::FromPrimitive>(
        &self,
        shape: &[usize],
    ) -> Result<Tensor<T, CpuBackend>> {
        if shape.len() != 2 {
            return Err(NNError::InvalidInput {
                message: "Xavier initialization requires 2D tensors (weight matrices)".to_string(),
            });
        }

        let fan_in = shape[1] as f64;
        let fan_out = shape[0] as f64;
        let limit = (6.0 / (fan_in + fan_out)).sqrt();

        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(shape.iter().product());

        for _ in 0..shape.iter().product::<usize>() {
            // Uniform distribution: [-limit, limit]
            let value: f64 = rng.gen_range(-limit..=limit);
            data.push(T::from(value).unwrap());
        }

        crate::tensor_from_vec_ok(CpuBackend::default(), data, shape.to_vec())
    }

    /// Initialize weights for linear layer
    ///
    /// # Arguments
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    ///
    /// # Returns
    /// Weight matrix of shape (out_features, in_features)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::init::Xavier;
    ///
    /// let init = Xavier::new();
    /// let weights = init.initialize_linear::<f32>(784, 128);
    /// assert_eq!(weights.unwrap().shape(), &[128, 784]);
    /// ```
    pub fn initialize_linear<T: Dtype + num_traits::Float + num_traits::FromPrimitive>(
        &self,
        in_features: usize,
        out_features: usize,
    ) -> Result<Tensor<T, CpuBackend>> {
        self.initialize(&[out_features, in_features])
    }
}

/// Kaiming (He) weight initialization
///
/// Initializes weights using Kaiming initialization, which is specifically
/// designed for layers followed by ReLU activation functions.
///
/// This initialization provides better performance than Xavier for deep networks
/// with ReLU activations.
#[derive(Debug, Clone, Default)]
pub struct Kaiming;

impl Kaiming {
    /// Create a new Kaiming initializer
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::init::Kaiming;
    ///
    /// let init = Kaiming::new();
    /// ```
    pub fn new() -> Self {
        Self
    }

    /// Initialize a tensor using Kaiming initialization for ReLU
    ///
    /// # Arguments
    /// * `shape` - Shape of the tensor to initialize
    ///
    /// # Returns
    /// Initialized tensor with Kaiming initialization for ReLU
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::init::Kaiming;
    ///
    /// let init = Kaiming::new();
    /// let weights = init.initialize_relu::<f32>(&[64, 32]);
    /// assert_eq!(weights.unwrap().shape(), &[64, 32]);
    /// ```
    pub fn initialize_relu<T: Dtype + num_traits::Float + num_traits::FromPrimitive>(
        &self,
        shape: &[usize],
    ) -> Result<Tensor<T, CpuBackend>> {
        if shape.len() != 2 {
            return Err(NNError::InvalidInput {
                message: "Kaiming initialization requires 2D tensors (weight matrices)".to_string(),
            });
        }

        let fan_in = shape[1] as f64;
        let std = (2.0 / fan_in).sqrt();

        self.initialize_normal(shape, T::zero(), T::from(std).unwrap())
    }

    /// Initialize a tensor using Kaiming initialization for other activations
    ///
    /// # Arguments
    /// * `shape` - Shape of the tensor to initialize
    ///
    /// # Returns
    /// Initialized tensor with Kaiming initialization for non-ReLU activations
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::init::Kaiming;
    ///
    /// let init = Kaiming::new();
    /// let weights = init.initialize_default::<f32>(&[64, 32]);
    /// assert_eq!(weights.unwrap().shape(), &[64, 32]);
    /// ```
    pub fn initialize_default<T: Dtype + num_traits::Float + num_traits::FromPrimitive>(
        &self,
        shape: &[usize],
    ) -> Result<Tensor<T, CpuBackend>> {
        if shape.len() != 2 {
            return Err(NNError::InvalidInput {
                message: "Kaiming initialization requires 2D tensors (weight matrices)".to_string(),
            });
        }

        let fan_in = shape[1] as f64;
        let std = (1.0 / fan_in).sqrt();

        self.initialize_normal(shape, T::zero(), T::from(std).unwrap())
    }

    /// Initialize weights for linear layer with ReLU activation
    ///
    /// # Arguments
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    ///
    /// # Returns
    /// Weight matrix of shape (out_features, in_features)
    ///
    /// # Example
    /// ```rust
    /// use coeus_nn::init::Kaiming;
    ///
    /// let init = Kaiming::new();
    /// let weights = init.initialize_linear_relu::<f32>(784, 128);
    /// assert_eq!(weights.unwrap().shape(), &[128, 784]);
    /// ```
    pub fn initialize_linear_relu<T: Dtype + num_traits::Float + num_traits::FromPrimitive>(
        &self,
        in_features: usize,
        out_features: usize,
    ) -> Result<Tensor<T, CpuBackend>> {
        self.initialize_relu(&[out_features, in_features])
    }

    /// Initialize tensor with normal distribution
    ///
    /// # Arguments
    /// * `shape` - Shape of the tensor
    /// * `mean` - Mean of the normal distribution
    /// * `std` - Standard deviation of the normal distribution
    ///
    /// # Returns
    /// Tensor initialized with normal distribution
    fn initialize_normal<T: Dtype + num_traits::Float + num_traits::FromPrimitive>(
        &self,
        shape: &[usize],
        mean: T,
        std: T,
    ) -> Result<Tensor<T, CpuBackend>> {
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(shape.iter().product());

        for _ in 0..shape.iter().product::<usize>() {
            // Normal distribution: N(mean, std²)
            let value: f64 = rng.sample(
                rand_distr::Normal::new(
                    num_traits::ToPrimitive::to_f64(&mean).unwrap(),
                    num_traits::ToPrimitive::to_f64(&std).unwrap(),
                )
                .unwrap(),
            );
            data.push(T::from(value).unwrap());
        }

        crate::tensor_from_vec_ok(CpuBackend::default(), data, shape.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_initialization() {
        let init = Xavier::new();
        let tensor: Tensor<f64, CpuBackend> = init.initialize(&[10, 5]).unwrap();

        assert_eq!(tensor.shape(), &[10, 5]);
        assert_eq!(tensor.numel(), 50);

        // Check that values are within expected range
        let limit = (6.0f64 / (5.0f64 + 10.0f64)).sqrt();
        for &val in tensor.data() {
            let val_f64 = val;
            assert!(val_f64 >= -limit && val_f64 <= limit);
        }
    }

    #[test]
    fn test_xavier_linear_initialization() {
        let init = Xavier::new();
        let tensor: Tensor<f64, CpuBackend> = init.initialize_linear(5, 10).unwrap();

        assert_eq!(tensor.shape(), &[10, 5]);
        assert_eq!(tensor.numel(), 50);
    }

    #[test]
    fn test_kaiming_relu_initialization() {
        let init = Kaiming::new();
        let tensor: Tensor<f64, CpuBackend> = init.initialize_relu(&[10, 5]).unwrap();

        assert_eq!(tensor.shape(), &[10, 5]);
        assert_eq!(tensor.numel(), 50);

        // Values should follow normal distribution with std = sqrt(2/fan_in)
        let expected_std = (2.0f64 / 5.0f64).sqrt();
        let mean = tensor.data().iter().copied().sum::<f64>() / tensor.numel() as f64;
        let variance = tensor
            .data()
            .iter()
            .map(|x: &f64| {
                let diff = *x - mean;
                diff * diff
            })
            .sum::<f64>()
            / tensor.numel() as f64;
        let std = variance.sqrt();

        // Check that computed std is reasonably close to expected std
        // With small sample sizes (50), statistical variation is expected
        // Use 3-sigma rule for statistical significance
        let relative_error = (std - expected_std).abs() / expected_std;
        assert!(
            relative_error < 0.5,
            "Standard deviation too far from expected: {} vs {}",
            std,
            expected_std
        );
    }

    #[test]
    fn test_kaiming_linear_initialization() {
        let init = Kaiming::new();
        let tensor: Tensor<f64, CpuBackend> = init.initialize_linear_relu(5, 10).unwrap();

        assert_eq!(tensor.shape(), &[10, 5]);
        assert_eq!(tensor.numel(), 50);
    }

    #[test]
    fn test_initialization_errors() {
        let xavier = Xavier::new();
        let kaiming = Kaiming::new();

        // Should fail for non-2D tensors
        assert!(xavier.initialize::<f64>(&[10]).is_err());
        assert!(xavier.initialize::<f64>(&[10, 5, 2]).is_err());
        assert!(kaiming.initialize_relu::<f64>(&[10]).is_err());
        assert!(kaiming.initialize_default::<f64>(&[10, 5, 2]).is_err());
    }
}


