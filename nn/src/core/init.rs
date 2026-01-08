//! Weight initialization methods for neural networks.
//!
//! This module provides various initialization strategies for neural network parameters,
//! following best practices from deep learning research.
//!
//! # Examples
//!
//! ```rust
//! use nn::init;
//! use nn::Linear;
//! use backend::{Backend, CpuBackend};
//! use storage::{Storage, StorageFromVec, DenseStorage};
//! use dtype::float::Float32;
//!
//! let mut layer = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(784, 128).unwrap();
//!
//! // Initialize with Kaiming/He initialization (recommended for ReLU)
//! // Note: Apply to weight tensor directly
//! let mut weight = layer.weight.data().clone();
//! init::kaiming_uniform_(&mut weight, 0.0, init::NonLinearity::ReLU);
//!
//! // Or use Xavier/Glorot initialization (recommended for sigmoid/tanh)
//! init::xavier_uniform_(&mut weight, 1.0);
//! ```
//!
//! # References
//! - Glorot & Bengio (2010): "Understanding the difficulty of training deep feedforward neural networks"
//! - He et al. (2015): "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
//! - Saxe et al. (2013): "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"

use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use rand_distr::StandardNormal;
use storage::{DenseStorage, Storage, StorageFromVec};
use tensor::Tensor;

use crate::core::error::Result;

/// Non-linearity type for Kaiming initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonLinearity {
    /// Linear activation (no non-linearity)
    Linear,
    /// ReLU activation
    ReLU,
    /// Leaky ReLU activation
    LeakyReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
}

impl NonLinearity {
    /// Get the gain factor for this non-linearity.
    ///
    /// # Arguments
    /// * `param` - Parameter for parameterized non-linearities (e.g., negative slope for LeakyReLU)
    pub fn gain(&self, param: f64) -> f64 {
        match self {
            NonLinearity::Linear => 1.0,
            NonLinearity::ReLU => (2.0_f64).sqrt(),
            NonLinearity::LeakyReLU => (2.0 / (1.0 + param * param)).sqrt(),
            NonLinearity::Sigmoid => 1.0,
            NonLinearity::Tanh => 5.0 / 3.0,
        }
    }
}

/// Calculate fan-in and fan-out for a tensor.
///
/// # Arguments
/// * `shape` - Shape of the tensor
///
/// # Returns
/// Tuple of (fan_in, fan_out)
fn calculate_fan_in_fan_out(shape: &[usize]) -> (usize, usize) {
    let dimensions = shape.len();

    if dimensions < 2 {
        // For 1D tensors, fan_in = fan_out = num_elements
        let num_elements = shape.iter().product();
        return (num_elements, num_elements);
    }

    if dimensions == 2 {
        // For 2D tensors (Linear layers): fan_in = input_features, fan_out = output_features
        return (shape[1], shape[0]);
    }

    // For Conv layers (3D+): fan_in/out = num_input/output_channels * receptive_field_size
    let num_input_fmaps = shape[1];
    let num_output_fmaps = shape[0];
    let receptive_field_size: usize = shape[2..].iter().product();

    let fan_in = num_input_fmaps * receptive_field_size;
    let fan_out = num_output_fmaps * receptive_field_size;

    (fan_in, fan_out)
}

/// Initialize tensor with uniform distribution.
///
/// Fills the tensor with values drawn from uniform distribution U(a, b).
///
/// # Arguments
/// * `tensor` - Tensor to initialize
/// * `a` - Lower bound
/// * `b` - Upper bound
///
/// # Examples
/// ```rust
/// use nn::init;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let mut tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[10, 10]).unwrap();
/// init::uniform_(&mut tensor, -0.1, 0.1).unwrap();
/// ```
pub fn uniform_<B, S, T>(tensor: &mut Tensor<B, S, T>, a: f64, b: f64) -> Result<()>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + Clone,
    T: DataType + FloatExt,
{
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(a, b);

    let data = tensor.as_mut_slice();
    for val in data.iter_mut() {
        *val = T::from(dist.sample(&mut rng)).unwrap();
    }

    Ok(())
}

/// Initialize tensor with normal distribution.
///
/// Fills the tensor with values drawn from normal distribution N(mean, std²).
///
/// # Arguments
/// * `tensor` - Tensor to initialize
/// * `mean` - Mean of the distribution
/// * `std` - Standard deviation of the distribution
///
/// # Examples
/// ```rust
/// use nn::init;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let mut tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[10, 10]).unwrap();
/// init::normal_(&mut tensor, 0.0, 0.01).unwrap();
/// ```
pub fn normal_<T: DataType + FloatExt>(
    tensor: &mut Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    mean: f64,
    std: f64,
) -> Result<()> {
    let mut rng = rand::thread_rng();

    let data = tensor.as_mut_slice();
    for val in data.iter_mut() {
        // Sample from standard normal N(0,1), then scale and shift
        let sample: f64 = rng.sample(StandardNormal);
        *val = T::from(mean + std * sample).unwrap();
    }

    Ok(())
}

/// Initialize tensor with constant value.
///
/// # Arguments
/// * `tensor` - Tensor to initialize
/// * `value` - Constant value to fill
///
/// # Examples
/// ```rust
/// use nn::init;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let mut tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[10, 10]).unwrap();
/// init::constant_(&mut tensor, 0.5).unwrap();
/// ```
pub fn constant_<T: DataType + FloatExt>(
    tensor: &mut Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    value: f64,
) -> Result<()> {
    let val = T::from(value).unwrap();
    let data = tensor.as_mut_slice();
    for elem in data.iter_mut() {
        *elem = val;
    }
    Ok(())
}

/// Initialize tensor with zeros.
///
/// # Examples
/// ```rust
/// use nn::init;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let mut tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[10, 10]).unwrap();
/// init::zeros_(&mut tensor).unwrap();
/// ```
pub fn zeros_<T: DataType + FloatExt>(
    tensor: &mut Tensor<CpuBackend<T>, DenseStorage<T>, T>,
) -> Result<()> {
    constant_(tensor, 0.0)
}

/// Initialize tensor with ones.
///
/// # Examples
/// ```rust
/// use nn::init;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let mut tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[10, 10]).unwrap();
/// init::ones_(&mut tensor).unwrap();
/// ```
pub fn ones_<T: DataType + FloatExt>(
    tensor: &mut Tensor<CpuBackend<T>, DenseStorage<T>, T>,
) -> Result<()> {
    constant_(tensor, 1.0)
}

/// Initialize tensor with Xavier/Glorot uniform distribution.
///
/// Fills the tensor with values drawn from uniform distribution:
/// U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
///
/// This initialization is recommended for layers with sigmoid or tanh activations.
///
/// # Arguments
/// * `tensor` - Tensor to initialize
/// * `gain` - Scaling factor (default: 1.0)
///
/// # Examples
/// ```rust
/// use nn::init;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let mut tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[128, 784]).unwrap();
/// init::xavier_uniform_(&mut tensor, 1.0).unwrap();
/// ```
///
/// # References
/// - Glorot & Bengio (2010): "Understanding the difficulty of training deep feedforward neural networks"
pub fn xavier_uniform_<B, S, T>(tensor: &mut Tensor<B, S, T>, gain: f64) -> Result<()>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + Clone,
    T: DataType + FloatExt,
{
    let (fan_in, fan_out) = calculate_fan_in_fan_out(tensor.shape().dims());
    let std = gain * (2.0 / (fan_in + fan_out) as f64).sqrt();
    let a = std * (3.0_f64).sqrt(); // sqrt(3) for uniform distribution
    uniform_(tensor, -a, a)
}

/// Initialize tensor with Xavier/Glorot normal distribution.
///
/// Fills the tensor with values drawn from normal distribution:
/// N(0, std²) where std = gain * sqrt(2 / (fan_in + fan_out))
///
/// This initialization is recommended for layers with sigmoid or tanh activations.
///
/// # Arguments
/// * `tensor` - Tensor to initialize
/// * `gain` - Scaling factor (default: 1.0)
///
/// # Examples
/// ```rust
/// use nn::init;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let mut tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[128, 784]).unwrap();
/// init::xavier_normal_(&mut tensor, 1.0).unwrap();
/// ```
///
/// # References
/// - Glorot & Bengio (2010): "Understanding the difficulty of training deep feedforward neural networks"
pub fn xavier_normal_<T: DataType + FloatExt>(
    tensor: &mut Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    gain: f64,
) -> Result<()> {
    let (fan_in, fan_out) = calculate_fan_in_fan_out(tensor.shape().dims());
    let std = gain * (2.0 / (fan_in + fan_out) as f64).sqrt();
    normal_(tensor, 0.0, std)
}

/// Initialize tensor with Kaiming/He uniform distribution.
///
/// Fills the tensor with values drawn from uniform distribution:
/// U(-bound, bound) where bound = gain * sqrt(3 / fan_in)
///
/// This initialization is recommended for layers with ReLU activations.
///
/// # Arguments
/// * `tensor` - Tensor to initialize
/// * `a` - Negative slope of the rectifier (0 for ReLU, >0 for LeakyReLU)
/// * `nonlinearity` - Type of non-linearity
///
/// # Examples
/// ```rust
/// use nn::init;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let mut tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[128, 784]).unwrap();
/// init::kaiming_uniform_(&mut tensor, 0.0, init::NonLinearity::ReLU).unwrap();
/// ```
///
/// # References
/// - He et al. (2015): "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
pub fn kaiming_uniform_<T: DataType + FloatExt>(
    tensor: &mut Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    a: f64,
    nonlinearity: NonLinearity,
) -> Result<()> {
    let (fan_in, _) = calculate_fan_in_fan_out(tensor.shape().dims());
    let gain = nonlinearity.gain(a);
    let std = gain / (fan_in as f64).sqrt();
    let bound = std * (3.0_f64).sqrt(); // sqrt(3) for uniform distribution
    uniform_(tensor, -bound, bound)
}

/// Initialize tensor with Kaiming/He normal distribution.
///
/// Fills the tensor with values drawn from normal distribution:
/// N(0, std²) where std = gain / sqrt(fan_in)
///
/// This initialization is recommended for layers with ReLU activations.
///
/// # Arguments
/// * `tensor` - Tensor to initialize
/// * `a` - Negative slope of the rectifier (0 for ReLU, >0 for LeakyReLU)
/// * `nonlinearity` - Type of non-linearity
///
/// # Examples
/// ```rust
/// use nn::init;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let mut tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[128, 784]).unwrap();
/// init::kaiming_normal_(&mut tensor, 0.0, init::NonLinearity::ReLU).unwrap();
/// ```
///
/// # References
/// - He et al. (2015): "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
pub fn kaiming_normal_<T: DataType + FloatExt>(
    tensor: &mut Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    a: f64,
    nonlinearity: NonLinearity,
) -> Result<()> {
    let (fan_in, _) = calculate_fan_in_fan_out(tensor.shape().dims());
    let gain = nonlinearity.gain(a);
    let std = gain / (fan_in as f64).sqrt();
    normal_(tensor, 0.0, std)
}

/// Initialize tensor with orthogonal matrix.
///
/// Fills the tensor with a (semi-)orthogonal matrix using QR decomposition.
/// This is particularly useful for RNN weight initialization.
///
/// For 2D tensors, the rows or columns (whichever is smaller) will be orthogonal.
/// For tensors with more than 2 dimensions, the last two dimensions are treated
/// as a matrix and orthogonalized.
///
/// # Arguments
/// * `tensor` - Tensor to initialize
/// * `gain` - Multiplicative factor to apply to the orthogonal matrix
///
/// # Examples
/// ```rust
/// use nn::init;
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// let mut tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[128, 128]).unwrap();
/// init::orthogonal_(&mut tensor, 1.0).unwrap();
/// ```
///
/// # References
/// - Saxe et al. (2013): "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"
pub fn orthogonal_<T: DataType + FloatExt + PartialOrd>(
    tensor: &mut Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    gain: f64,
) -> Result<()> {
    let shape = tensor.shape().dims();

    if shape.len() < 2 {
        return Err(crate::core::error::NNError::InvalidInput {
            message: "Orthogonal initialization requires at least 2D tensor".to_string(),
        });
    }

    // Get the last two dimensions
    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];

    // Number of matrices to orthogonalize (product of all dimensions except last 2)
    let num_matrices: usize = shape[..shape.len() - 2].iter().product::<usize>().max(1);

    let mut rng = rand::thread_rng();
    let data = tensor.as_mut_slice();

    for mat_idx in 0..num_matrices {
        let offset = mat_idx * rows * cols;

        // Fill with random normal values from N(0,1)
        for i in 0..rows * cols {
            let sample: f64 = rng.sample(StandardNormal);
            data[offset + i] = T::from(sample).unwrap();
        }

        // Perform QR decomposition using Gram-Schmidt process
        // This is a simplified implementation; for production, consider using a linear algebra library
        for col in 0..cols.min(rows) {
            // Normalize column
            let mut norm_sq = T::zero();
            for row in 0..rows {
                let idx = offset + row * cols + col;
                norm_sq = norm_sq + data[idx] * data[idx];
            }

            let norm = norm_sq.sqrt();
            if norm > T::from(1e-10).unwrap() {
                for row in 0..rows {
                    let idx = offset + row * cols + col;
                    data[idx] = data[idx] / norm;
                }
            }

            // Orthogonalize subsequent columns
            for next_col in (col + 1)..cols {
                let mut dot = T::zero();
                for row in 0..rows {
                    let idx1 = offset + row * cols + col;
                    let idx2 = offset + row * cols + next_col;
                    dot = dot + data[idx1] * data[idx2];
                }

                for row in 0..rows {
                    let idx1 = offset + row * cols + col;
                    let idx2 = offset + row * cols + next_col;
                    data[idx2] = data[idx2] - dot * data[idx1];
                }
            }
        }

        // Apply gain
        let gain_t = T::from(gain).unwrap();
        for i in 0..rows * cols {
            data[offset + i] = data[offset + i] * gain_t;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_uniform_initialization() {
        let mut tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[100, 100])
                .unwrap();
        uniform_(&mut tensor, -0.5, 0.5).unwrap();

        let data = tensor.as_slice();

        // Check all values are in range
        for &val in data {
            assert!(val.get() >= -0.5 && val.get() <= 0.5);
        }

        // Check not all zeros
        assert!(data.iter().any(|&v: &Float32| v.get().abs() > 1e-6));
    }

    #[test]
    fn test_normal_initialization() {
        let mut tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[100, 100])
                .unwrap();
        normal_(&mut tensor, 0.0, 0.1).unwrap();

        let data = tensor.as_slice();

        // Check not all zeros
        assert!(data.iter().any(|&v: &Float32| v.get().abs() > 1e-6));

        // Rough check: most values should be within 3 standard deviations
        let within_3std = data
            .iter()
            .filter(|&&v: &&Float32| v.get().abs() <= 0.3)
            .count();
        assert!(within_3std as f32 / data.len() as f32 > 0.95);

        // Calculate sample mean and std to verify distribution
        let mean: f32 = data.iter().map(|v: &Float32| v.get()).sum::<f32>() / data.len() as f32;
        let variance: f32 = data
            .iter()
            .map(|v: &Float32| (v.get() - mean).powi(2))
            .sum::<f32>()
            / data.len() as f32;
        let std = variance.sqrt();

        // Mean should be close to 0.0 (within 3 standard errors)
        let std_error = 0.1 / (data.len() as f32).sqrt();
        assert!(
            mean.abs() < 3.0 * std_error,
            "Mean {} not close to 0.0",
            mean
        );

        // Std should be close to 0.1 (within reasonable tolerance)
        assert!((std - 0.1).abs() < 0.02, "Std {} not close to 0.1", std);
    }

    #[test]
    fn test_constant_initialization() {
        let mut tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[10, 10])
                .unwrap();
        constant_(&mut tensor, 0.42).unwrap();

        let data = tensor.as_slice();
        for &val in data {
            let val_f32: f32 = val.get();
            assert!((val_f32 - 0.42).abs() < 1e-6);
        }
    }

    #[test]
    fn test_zeros_initialization() {
        let mut tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[10, 10]).unwrap();
        zeros_(&mut tensor).unwrap();

        let data = tensor.as_slice();
        for &val in data {
            let val_f32: f32 = val.get();
            assert_eq!(val_f32, 0.0);
        }
    }

    #[test]
    fn test_ones_initialization() {
        let mut tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[10, 10])
                .unwrap();
        ones_(&mut tensor).unwrap();

        let data = tensor.as_slice();
        for &val in data {
            let val_f32: f32 = val.get();
            assert_eq!(val_f32, 1.0);
        }
    }

    #[test]
    fn test_xavier_uniform_initialization() {
        let mut tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[100, 200])
                .unwrap();
        xavier_uniform_(&mut tensor, 1.0).unwrap();

        let data = tensor.as_slice();

        // Check not all zeros
        assert!(data.iter().any(|&v: &Float32| v.get().abs() > 1e-6));

        // Calculate expected bound: gain * sqrt(6 / (fan_in + fan_out)) * sqrt(3)
        let fan_in = 200;
        let fan_out = 100;
        let expected_bound = 1.0 * (6.0 / (fan_in + fan_out) as f32).sqrt();

        // All values should be within bounds
        for &val in data {
            let val_f32: f32 = val.get();
            assert!(val_f32.abs() <= expected_bound * 1.1); // Allow 10% tolerance
        }
    }

    #[test]
    fn test_xavier_normal_initialization() {
        let mut tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[100, 200])
                .unwrap();
        xavier_normal_(&mut tensor, 1.0).unwrap();

        let data = tensor.as_slice();

        // Check not all zeros
        assert!(data.iter().any(|&v: &Float32| v.get().abs() > 1e-6));
    }

    #[test]
    fn test_kaiming_uniform_initialization() {
        let mut tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[100, 200])
                .unwrap();
        kaiming_uniform_(&mut tensor, 0.0, NonLinearity::ReLU).unwrap();

        let data = tensor.as_slice();

        // Check not all zeros
        assert!(data.iter().any(|&v: &Float32| v.get().abs() > 1e-6));

        // Calculate expected bound: sqrt(2) / sqrt(fan_in) * sqrt(3)
        let fan_in = 200;
        let expected_bound = (2.0_f32).sqrt() / (fan_in as f32).sqrt() * (3.0_f32).sqrt();

        // All values should be within bounds
        for &val in data {
            let val_f32: f32 = val.get();
            assert!(val_f32.abs() <= expected_bound * 1.1); // Allow 10% tolerance
        }
    }

    #[test]
    fn test_kaiming_normal_initialization() {
        let mut tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[100, 200])
                .unwrap();
        kaiming_normal_(&mut tensor, 0.0, NonLinearity::ReLU).unwrap();

        let data = tensor.as_slice();

        // Check not all zeros
        assert!(data.iter().any(|&v: &Float32| v.get().abs() > 1e-6));
    }

    #[test]
    fn test_orthogonal_initialization() {
        let mut tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[50, 50])
                .unwrap();
        orthogonal_(&mut tensor, 1.0).unwrap();

        let data = tensor.as_slice();

        // Check not all zeros
        assert!(data.iter().any(|&v: &Float32| v.get().abs() > 1e-6));

        // Check orthogonality: columns should be orthonormal
        // For a 50x50 matrix, check a few column pairs
        for i in 0..5 {
            for j in (i + 1)..5 {
                let mut dot: f32 = 0.0;
                for row in 0..50 {
                    dot += data[row * 50 + i].get() * data[row * 50 + j].get();
                }
                // Dot product of orthogonal vectors should be ~0
                assert!(
                    dot.abs() < 0.1,
                    "Columns {} and {} not orthogonal: dot = {}",
                    i,
                    j,
                    dot
                );
            }
        }

        // Check normalization: column norms should be ~1 (scaled by gain)
        for i in 0..5 {
            let mut norm_sq: f32 = 0.0;
            for row in 0..50 {
                let val = data[row * 50 + i].get();
                norm_sq += val * val;
            }
            let norm = norm_sq.sqrt();
            assert!(
                (norm - 1.0).abs() < 0.1,
                "Column {} norm = {}, expected ~1.0",
                i,
                norm
            );
        }
    }

    #[test]
    fn test_calculate_fan_in_fan_out_2d() {
        let (fan_in, fan_out) = calculate_fan_in_fan_out(&[100, 200]);
        assert_eq!(fan_in, 200);
        assert_eq!(fan_out, 100);
    }

    #[test]
    fn test_calculate_fan_in_fan_out_conv() {
        // Conv2D: [out_channels, in_channels, kernel_h, kernel_w]
        let (fan_in, fan_out) = calculate_fan_in_fan_out(&[64, 32, 3, 3]);
        assert_eq!(fan_in, 32 * 3 * 3); // in_channels * receptive_field
        assert_eq!(fan_out, 64 * 3 * 3); // out_channels * receptive_field
    }

    #[test]
    fn test_nonlinearity_gain() {
        assert_eq!(NonLinearity::Linear.gain(0.0), 1.0);
        assert!((NonLinearity::ReLU.gain(0.0) - (2.0_f64).sqrt()).abs() < 1e-6);
        assert!((NonLinearity::Tanh.gain(0.0) - 5.0 / 3.0).abs() < 1e-6);
    }
}
