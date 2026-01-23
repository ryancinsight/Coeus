//! Group Normalization layer.

use std::marker::PhantomData;

use backend::{Backend, CpuBackend};
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// Group Normalization layer.
///
/// Divides channels into groups and normalizes within each group independently.
/// Unlike BatchNorm, GroupNorm is independent of batch size, making it suitable
/// for small batch training (e.g., object detection, video understanding).
#[derive(Debug, Clone)]
pub struct GroupNorm<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
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
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::One + num_traits::Zero,
{
    /// Create a new GroupNorm layer.
    pub fn new(num_groups: usize, num_channels: usize, eps: f64, affine: bool) -> Result<Self> {
        if num_channels % num_groups != 0 {
            return Err(NNError::InvalidConfiguration {
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
        let weight = Parameter::new(weight_data, "weight".to_string());

        let bias_data = if affine {
            Tensor::<B, S, T>::from_vec(vec![T::zero(); num_channels], &[num_channels])
                .unwrap()
                .requires_grad_(true)
        } else {
            Tensor::<B, S, T>::from_vec(vec![T::zero(); num_channels], &[num_channels]).unwrap()
        };
        let bias = Parameter::new(bias_data, "bias".to_string());

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

    fn train(&mut self, _mode: bool) {}

    fn name(&self) -> &str {
        "GroupNorm"
    }

    fn clone_box(&self) -> Box<dyn Module<CpuBackend<T>, DenseStorage<T>, T>> {
        Box::new(self.clone())
    }
}
