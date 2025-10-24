//! Group Normalization and Instance Normalization layers.
//!
//! This module provides normalization layers that are independent of batch size,
//! making them suitable for small batch training and inference.

use std::marker::PhantomData;

use coeus_backend::{Backend, CpuBackend};
use coeus_dtype::{traits::FloatExt, DataType};
use coeus_storage::{DenseStorage, Storage, StorageFromVec};
use coeus_tensor::Tensor;

use crate::error::{NNError, Result};
use crate::module::Module;
use crate::parameter::Parameter;

/// Group Normalization layer.
///
/// Divides channels into groups and normalizes within each group independently.
/// Unlike BatchNorm, GroupNorm is independent of batch size, making it suitable
/// for small batch training (e.g., object detection, video understanding).
///
/// # Formula
/// ```text
/// For each group g:
///   mean_g = Σ(x_g) / (C_g * H * W)
///   var_g = Σ((x_g - mean_g)²) / (C_g * H * W)
///   x_normalized = (x - mean_g) / √(var_g + ε)
///   output = γ * x_normalized + β
/// ```
///
/// where C_g is the number of channels per group.
///
/// # Shape
/// - Input: `(N, C, H, W)` where N is batch size, C is channels, H is height, W is width
/// - Output: `(N, C, H, W)` (same shape as input)
///
/// # Examples
/// ```rust
/// use coeus_nn::{GroupNorm, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// // 32 channels, 8 groups (4 channels per group)
/// let groupnorm = GroupNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(8, 32, 1e-5, true).unwrap();
///
/// // Input: [batch_size=2, channels=32, height=16, width=16]
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 32, 16, 16]).unwrap();
///
/// // Output: Same shape, normalized per group
/// let output = groupnorm.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 32, 16, 16]);
/// ```
///
/// # References
/// - Wu & He (2018): "Group Normalization" (ECCV 2018)
/// - Used in Mask R-CNN, video understanding, small batch training
#[derive(Debug, Clone)]
pub struct GroupNorm<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// Number of groups
    pub num_groups: usize,
    /// Number of channels
    pub num_channels: usize,
    /// Scale parameter γ [num_channels]
    pub weight: Parameter<B, S, T>,
    /// Shift parameter β [num_channels]
    pub bias: Parameter<B, S, T>,
    /// Numerical stability constant ε
    pub eps: f64,
    /// Whether to use affine transformation (learnable γ and β)
    pub affine: bool,
    _phantom: PhantomData<(B, S, T)>,
}

impl<B, S, T> GroupNorm<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + num_traits::One + num_traits::Zero,
{
    /// Create a new GroupNorm layer.
    ///
    /// # Arguments
    /// * `num_groups` - Number of groups to divide channels into
    /// * `num_channels` - Number of input channels (must be divisible by num_groups)
    /// * `eps` - Numerical stability constant (default: 1e-5)
    /// * `affine` - Whether to use learnable affine parameters (default: true)
    ///
    /// # Panics
    /// Panics if `num_channels` is not divisible by `num_groups`.
    ///
    /// # Examples
    /// ```rust
    /// use coeus_nn::GroupNorm;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let groupnorm = GroupNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(8, 32, 1e-5, true).unwrap();
    /// ```
    pub fn new(num_groups: usize, num_channels: usize, eps: f64, affine: bool) -> Result<Self> {
        if num_channels % num_groups != 0 {
            return Err(crate::error::NNError::InvalidConfiguration {
                message: format!(
                    "num_channels ({}) must be divisible by num_groups ({})",
                    num_channels, num_groups
                ),
            });
        }

        let weight_data = if affine {
            Tensor::<B, S, T>::from_vec(vec![T::one(); num_channels], &[num_channels])
                .unwrap()
                .requires_grad_(true)
        } else {
            Tensor::<B, S, T>::from_vec(vec![T::one(); num_channels], &[num_channels]).unwrap()
        };
        let weight = Parameter::<B, S, T>::new(weight_data, "weight".to_string());

        let bias_data = if affine {
            Tensor::<B, S, T>::from_vec(vec![T::zero(); num_channels], &[num_channels])
                .unwrap()
                .requires_grad_(true)
        } else {
            Tensor::<B, S, T>::from_vec(vec![T::zero(); num_channels], &[num_channels]).unwrap()
        };
        let bias = Parameter::<B, S, T>::new(bias_data, "bias".to_string());

        Ok(Self {
            num_groups,
            num_channels,
            weight,
            bias,
            eps,
            affine,
            _phantom: PhantomData,
        })
    }
}

impl<T> Module<CpuBackend<T>, DenseStorage<T>, T> for GroupNorm<CpuBackend<T>, DenseStorage<T>, T>
where
    T: DataType + FloatExt + PartialOrd,
{
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        let input_shape = input.shape().dims();

        if input_shape.len() != 4usize {
            return Err(NNError::InvalidInput {
                message: format!("Expected 4D input (N, C, H, W), got {}D", input_shape.len()),
            });
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];

        if channels != self.num_channels {
            return Err(NNError::InvalidInput {
                message: format!("Expected {} channels, got {}", self.num_channels, channels),
            });
        }

        let channels_per_group = channels / self.num_groups;
        let group_size = channels_per_group * height * width;

        let input_data = input.as_slice();
        let weight_data = self.weight.data().as_slice();
        let bias_data = self.bias.data().as_slice();

        let mut output_data = vec![T::zero(); batch_size * channels * height * width];

        // Normalize each group independently
        for b in 0..batch_size {
            for g in 0..self.num_groups {
                // Compute mean for this group
                let mut sum = T::zero();
                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    for h in 0..height {
                        for w in 0..width {
                            let idx = b * (channels * height * width)
                                + channel_idx * (height * width)
                                + h * width
                                + w;
                            sum = sum + input_data[idx];
                        }
                    }
                }
                let mean = sum / T::from(group_size as f64).unwrap();

                // Compute variance for this group
                let mut var_sum = T::zero();
                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    for h in 0..height {
                        for w in 0..width {
                            let idx = b * (channels * height * width)
                                + channel_idx * (height * width)
                                + h * width
                                + w;
                            let diff = input_data[idx] - mean;
                            var_sum = var_sum + diff * diff;
                        }
                    }
                }
                let variance = var_sum / T::from(group_size as f64).unwrap();

                // Normalize and apply affine transformation
                let std = (variance + T::from(self.eps).unwrap()).sqrt();
                for c in 0..channels_per_group {
                    let channel_idx = g * channels_per_group + c;
                    let gamma = if self.affine {
                        weight_data[channel_idx]
                    } else {
                        T::one()
                    };
                    let beta = if self.affine {
                        bias_data[channel_idx]
                    } else {
                        T::zero()
                    };

                    for h in 0..height {
                        for w in 0..width {
                            let idx = b * (channels * height * width)
                                + channel_idx * (height * width)
                                + h * width
                                + w;
                            let normalized = (input_data[idx] - mean) / std;
                            output_data[idx] = gamma * normalized + beta;
                        }
                    }
                }
            }
        }

        Tensor::from_vec(output_data, input_shape).map_err(Into::into)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        if self.affine {
            vec![self.weight.clone(), self.bias.clone()]
        } else {
            Vec::new()
        }
    }

    fn zero_grad(&mut self) {
        if self.affine {
            self.weight.zero_grad();
            self.bias.zero_grad();
        }
    }

    fn train(&mut self, _mode: bool) {
        // No-op: GroupNorm doesn't have training-specific behavior
    }

    fn name(&self) -> &str {
        "GroupNorm"
    }
}

/// Instance Normalization layer.
///
/// Normalizes each channel independently for each instance in the batch.
/// This is equivalent to GroupNorm with num_groups = num_channels.
/// Essential for style transfer and image-to-image translation.
///
/// # Formula
/// ```text
/// For each channel c:
///   mean_c = Σ(x_c) / (H * W)
///   var_c = Σ((x_c - mean_c)²) / (H * W)
///   x_normalized = (x - mean_c) / √(var_c + ε)
///   output = γ * x_normalized + β
/// ```
///
/// # Shape
/// - Input: `(N, C, H, W)`
/// - Output: `(N, C, H, W)` (same shape as input)
///
/// # Examples
/// ```rust
/// use coeus_nn::{InstanceNorm, Module};
/// use coeus_tensor::Tensor;
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// let instancenorm = InstanceNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(64, 1e-5, true).unwrap();
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 32, 32]).unwrap();
/// let output = instancenorm.forward(&input).unwrap();
/// assert_eq!(output.shape().dims(), &[1, 64, 32, 32]);
/// ```
///
/// # References
/// - Ulyanov et al. (2016): "Instance Normalization: The Missing Ingredient for Fast Stylization"
/// - Used in style transfer, GANs, image-to-image translation
#[derive(Debug, Clone)]
pub struct InstanceNorm<B, S, T>
where
    B: Backend + Clone,
    S: Storage<T> + Clone + 'static,
    T: DataType,
{
    /// Underlying GroupNorm with num_groups = num_channels
    group_norm: GroupNorm<B, S, T>,
}

impl<B, S, T> InstanceNorm<B, S, T>
where
    B: Backend + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T>,
    T: DataType + FloatExt,
{
    /// Create a new InstanceNorm layer.
    ///
    /// # Arguments
    /// * `num_channels` - Number of input channels
    /// * `eps` - Numerical stability constant (default: 1e-5)
    /// * `affine` - Whether to use learnable affine parameters (default: true)
    ///
    /// # Examples
    /// ```rust
    /// use coeus_nn::InstanceNorm;
    /// use coeus_backend::CpuBackend;
    /// use coeus_storage::DenseStorage;
    /// use coeus_dtype::float::Float32;
    ///
    /// let instancenorm = InstanceNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(64, 1e-5, true).unwrap();
    /// ```
    pub fn new(num_channels: usize, eps: f64, affine: bool) -> Result<Self> {
        // InstanceNorm is GroupNorm with num_groups = num_channels
        Ok(Self {
            group_norm: GroupNorm::new(num_channels, num_channels, eps, affine)?,
        })
    }

    /// Get the number of channels
    pub fn num_channels(&self) -> usize {
        self.group_norm.num_channels
    }

    /// Get the epsilon value
    pub fn eps(&self) -> f64 {
        self.group_norm.eps
    }

    /// Get the affine flag
    pub fn affine(&self) -> bool {
        self.group_norm.affine
    }

    /// Get the weight parameter
    pub fn weight(&self) -> &Parameter<B, S, T> {
        &self.group_norm.weight
    }

    /// Get the bias parameter
    pub fn bias(&self) -> &Parameter<B, S, T> {
        &self.group_norm.bias
    }
}

impl<T> Module<CpuBackend<T>, DenseStorage<T>, T>
    for InstanceNorm<CpuBackend<T>, DenseStorage<T>, T>
where
    T: DataType + FloatExt + PartialOrd,
{
    fn forward(
        &self,
        input: &Tensor<CpuBackend<T>, DenseStorage<T>, T>,
    ) -> Result<Tensor<CpuBackend<T>, DenseStorage<T>, T>> {
        self.group_norm.forward(input)
    }

    fn parameters(&self) -> Vec<Parameter<CpuBackend<T>, DenseStorage<T>, T>> {
        self.group_norm.parameters()
    }

    fn zero_grad(&mut self) {
        self.group_norm.zero_grad();
    }

    fn train(&mut self, mode: bool) {
        self.group_norm.train(mode);
    }

    fn name(&self) -> &str {
        "InstanceNorm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_dtype::float::Float32;

    #[test]
    fn test_groupnorm_creation() {
        let groupnorm = GroupNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            8, 32, 1e-5, true,
        )
        .unwrap();
        assert_eq!(groupnorm.num_groups, 8);
        assert_eq!(groupnorm.num_channels, 32);
    }

    #[test]
    fn test_groupnorm_invalid_channels() {
        let result = GroupNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            8, 30, 1e-5, true,
        ); // 30 not divisible by 8
        assert!(result.is_err());
    }

    #[test]
    fn test_groupnorm_forward_basic() {
        let groupnorm = GroupNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            8, 32, 1e-5, true,
        )
        .unwrap();
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 32, 16, 16])
                .unwrap();
        let output = groupnorm.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 32, 16, 16]);
    }

    #[test]
    fn test_groupnorm_forward_computation() {
        let groupnorm = GroupNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            1, 2, 1e-5, false,
        )
        .unwrap(); // 1 group, 2 channels, no affine

        // Input: [1, 2, 2, 2] with known values
        let input_data = vec![
            // Channel 0
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            // Channel 1
            Float32::new(5.0),
            Float32::new(6.0),
            Float32::new(7.0),
            Float32::new(8.0),
        ];
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 2, 2, 2],
        )
        .unwrap();

        let output = groupnorm.forward(&input).unwrap();

        // Mean = (1+2+3+4+5+6+7+8) / 8 = 4.5
        // Var = ((1-4.5)² + (2-4.5)² + ... + (8-4.5)²) / 8 = 5.25
        // Std = sqrt(5.25 + 1e-5) ≈ 2.291
        // Normalized values should be (x - 4.5) / 2.291

        assert_eq!(output.shape().dims(), &[1, 2, 2, 2]);

        // Check first value: (1 - 4.5) / 2.291 ≈ -1.528
        let first_val = output.as_slice()[0].get();
        assert!((first_val - (-1.528)).abs() < 0.01);
    }

    #[test]
    fn test_groupnorm_multiple_groups() {
        let groupnorm =
            GroupNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(4, 8, 1e-5, true)
                .unwrap();
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 8, 4, 4])
                .unwrap();
        let output = groupnorm.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 8, 4, 4]);
    }

    #[test]
    fn test_groupnorm_parameters_affine() {
        let groupnorm = GroupNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            8, 32, 1e-5, true,
        )
        .unwrap();
        let params = groupnorm.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }

    #[test]
    fn test_groupnorm_parameters_no_affine() {
        let groupnorm = GroupNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            8, 32, 1e-5, false,
        )
        .unwrap();
        let params = groupnorm.parameters();
        assert_eq!(params.len(), 0); // no parameters
    }

    #[test]
    fn test_instancenorm_creation() {
        let instancenorm =
            InstanceNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                64, 1e-5, true,
            )
            .unwrap();
        assert_eq!(instancenorm.group_norm.num_channels, 64);
        assert_eq!(instancenorm.group_norm.num_groups, 64); // num_groups = num_channels
    }

    #[test]
    fn test_instancenorm_forward_basic() {
        let instancenorm =
            InstanceNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                32, 1e-5, true,
            )
            .unwrap();
        let input =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 32, 16, 16])
                .unwrap();
        let output = instancenorm.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, 32, 16, 16]);
    }

    #[test]
    fn test_instancenorm_forward_computation() {
        let instancenorm =
            InstanceNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                1, 1e-5, false,
            )
            .unwrap(); // 1 channel, no affine

        // Input: [1, 1, 2, 2] with known values
        let input_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ];
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[1, 1, 2, 2],
        )
        .unwrap();

        let output = instancenorm.forward(&input).unwrap();

        // Mean = (1+2+3+4) / 4 = 2.5
        // Var = ((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²) / 4 = 1.25
        // Std = sqrt(1.25 + 1e-5) ≈ 1.118
        // Normalized: (1 - 2.5) / 1.118 ≈ -1.342

        assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);

        let first_val = output.as_slice()[0].get();
        assert!((first_val - (-1.342)).abs() < 0.01);
    }

    #[test]
    fn test_instancenorm_batch_independence() {
        let instancenorm =
            InstanceNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                2, 1e-5, false,
            )
            .unwrap();

        // Two different instances in batch
        let input_data = vec![
            // Instance 0, Channel 0
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            // Instance 0, Channel 1
            Float32::new(5.0),
            Float32::new(6.0),
            Float32::new(7.0),
            Float32::new(8.0),
            // Instance 1, Channel 0
            Float32::new(10.0),
            Float32::new(20.0),
            Float32::new(30.0),
            Float32::new(40.0),
            // Instance 1, Channel 1
            Float32::new(50.0),
            Float32::new(60.0),
            Float32::new(70.0),
            Float32::new(80.0),
        ];
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data,
            &[2, 2, 2, 2],
        )
        .unwrap();

        let output = instancenorm.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 2, 2, 2]);

        // Each instance and channel should be normalized independently
        // Instance 0, Channel 0: mean = 2.5, std ≈ 1.118
        // Instance 1, Channel 0: mean = 25.0, std ≈ 11.18
        // The normalized values should be different scales
    }

    #[test]
    fn test_instancenorm_parameters_affine() {
        let instancenorm = InstanceNormCpu::<Float32>::new(32, 1e-5, true).unwrap();
        let params = instancenorm.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }

    #[test]
    fn test_instancenorm_parameters_no_affine() {
        let instancenorm = InstanceNormCpu::<Float32>::new(32, 1e-5, false).unwrap();
        let params = instancenorm.parameters();
        assert_eq!(params.len(), 0); // no parameters
    }
}

// ============================================================================
// TYPE ALIASES FOR BACKWARD COMPATIBILITY
// ============================================================================

/// Type alias for GroupNorm layer with CPU backend.
/// This provides backward compatibility with existing code.
pub type GroupNormCpu<T> = GroupNorm<CpuBackend<T>, DenseStorage<T>, T>;

/// Type alias for InstanceNorm layer with CPU backend.
/// This provides backward compatibility with existing code.
pub type InstanceNormCpu<T> = InstanceNorm<CpuBackend<T>, DenseStorage<T>, T>;
