//! Batch Normalization for 2D inputs (NCHW format).
//!
//! This module implements BatchNorm2d, which normalizes activations across
//! the batch dimension for 4D tensors in NCHW format.

use std::fmt;

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{DenseStorage, Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::batchnorm_core::BatchNormBase;
use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// Batch Normalization layer for 2D inputs (NCHW format).
///
/// Normalizes activations across the batch dimension to stabilize training.
/// During training, uses batch statistics and automatically updates running statistics.
/// During evaluation, uses running statistics for deterministic inference.
///
/// Running statistics are updated automatically during training forward passes using
/// interior mutability (RefCell), requiring no manual intervention.
///
/// Formula:
/// ```text
/// Training mode:
///   batch_mean = Σ(x) / N
///   batch_var = Σ((x - batch_mean)²) / N
///   x_normalized = (x - batch_mean) / √(batch_var + ε)
///   output = γ * x_normalized + β
///
///   # Automatically update running statistics
///   running_mean = momentum * running_mean + (1 - momentum) * batch_mean
///   running_var = momentum * running_var + (1 - momentum) * batch_var
///
/// Evaluation mode:
///   x_normalized = (x - running_mean) / √(running_var + ε)
///   output = γ * x_normalized + β
/// ```
///
/// # Examples
/// ```rust
/// use nn::{BatchNorm2d, Module};
/// use tensor::Tensor;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
/// use dtype::float::Float32;
///
/// // Create BatchNorm2d for 64 channels
/// let mut batchnorm = BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(CpuBackend::<Float32>::default(), 64, 1e-5, 0.1).unwrap();
///
/// // Set to training mode
/// <BatchNorm2d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::train(&mut batchnorm, true);
///
/// // Input: [batch_size=2, channels=64, height=32, width=32]
/// let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 64, 32, 32]).unwrap();
///
/// // Output: Same shape, normalized
/// let output = <BatchNorm2d<CpuBackend<Float32>, DenseStorage<Float32>, Float32> as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(&batchnorm, &input).unwrap();
/// assert_eq!(output.shape().dims(), &[2, 64, 32, 32]);
/// ```
#[derive(Debug)]
pub struct BatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    /// Base batch normalization parameters
    base: BatchNormBase<B, S, T>,
}

impl<B, S, T> BatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    /// Create a new BatchNorm2d layer.
    ///
    /// # Arguments
    /// * `backend` - Backend instance for tensor operations
    /// * `num_features` - Number of channels/features to normalize (C dimension)
    /// * `eps` - Small constant added to variance for numerical stability (default: 1e-5)
    /// * `momentum` - Momentum for running statistics update (default: 0.1)
    ///
    /// # Errors
    /// Returns `NNError::InvalidInput` if `num_features` is 0.
    pub fn new_with_backend(
        backend: B,
        num_features: usize,
        eps: f64,
        momentum: f64,
    ) -> Result<Self> {
        if num_features == 0 {
            return Err(NNError::InvalidInput {
                message: "num_features must be > 0".to_string(),
            });
        }

        let base = BatchNormBase::new(backend, num_features, eps, momentum, true)?;

        Ok(Self { base })
    }

    /// Create a new BatchNorm2d layer with default parameters.
    ///
    /// Uses default epsilon (1e-5) and momentum (0.1).
    pub fn new(num_features: usize) -> Result<Self>
    where
        B: Default,
    {
        Self::new_with_backend(B::default(), num_features, 1e-5, 0.1)
    }

    /// Get the number of features being normalized
    #[must_use]
    pub const fn num_features(&self) -> usize {
        self.base.num_features
    }

    /// Get the epsilon value for numerical stability
    #[must_use]
    pub const fn eps(&self) -> f64 {
        self.base.eps
    }

    /// Get the momentum for running statistics
    #[must_use]
    pub const fn momentum(&self) -> f64 {
        self.base.momentum
    }

    /// Check if the layer is in training mode
    #[must_use]
    pub const fn training(&self) -> bool {
        self.base.training
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.base.training = training;
    }
}

impl<B, S, T> Clone for BatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
        }
    }
}

impl<B, S, T> Module<B, S, T> for BatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    fn forward(
        &self,
        input: &Tensor<B, S, T>,
    ) -> Result<Tensor<B, S, T>> {
        let requires_grad = input.requires_grad();
        // For now, BatchNorm2d only works with dense tensors
        // Convert to dense, compute, then convert back if needed
        let input_dense = input.to_dense_generic()?;

        let input_shape = input_dense.shape().dims();
        if input_shape.len() != 4usize {
            return Err(NNError::InvalidInput {
                message: "Input must be 4D [N, C, H, W]".to_string(),
            });
        }
        if input_shape[1] != self.base.num_features {
            return Err(NNError::InvalidInput {
                message: format!("Input channels ({}) must match num_features ({})", input_shape[1], self.base.num_features),
            });
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];

        let input_data = input_dense.as_slice();
        let weight_data = self.base.weight.data().as_slice();
        let bias_data = self.base.bias.data().as_slice();

        let mut output_data = Vec::with_capacity(input_data.len());

        // Process each channel independently
        for c in 0..channels {
            let channel_start = c * height * width;
            let channel_size = height * width;

            // Extract channel data across all batches
            let mut channel_data = Vec::with_capacity(batch_size * channel_size);
            for b in 0..batch_size {
                let batch_start = b * channels * height * width + channel_start;
                channel_data.extend_from_slice(&input_data[batch_start..batch_start + channel_size]);
            }

            // Compute batch statistics
            let (mean, var) = if self.base.training {
                // Training mode: use batch statistics
                self.compute_batch_stats(&channel_data, batch_size, channel_size)?
            } else {
                // Evaluation mode: use running statistics
                let running_mean = self.base.running_mean.borrow();
                let running_var = self.base.running_var.borrow();
                let mean_val = running_mean.as_slice()[c];
                let var_val = running_var.as_slice()[c];
                (mean_val, var_val)
            };

            // Normalize channel
            let normalized = self.normalize_channel(
                &channel_data,
                mean,
                var,
                weight_data[c],
                bias_data[c],
            )?;

            output_data.extend(normalized);

            // Update running statistics in training mode
            if self.base.training && self.base.track_running_stats {
                let batch_mean = Tensor::<B, S, T>::from_vec(vec![mean], &[1])?;
                let batch_var = Tensor::<B, S, T>::from_vec(vec![var], &[1])?;
                self.update_running_stats(&batch_mean, &batch_var)?;
            }
        }

        let output = Tensor::<B, DenseStorage<T>, T>::from_vec(
            output_data,
            &[batch_size, channels, height, width],
        )?;

        // Convert back to original storage type if needed
        let result = output.to_generic()?;
        if requires_grad {
            // Future enhancement: Implement gradient computation for BatchNorm
            // For now, just return the result without gradient tracking
        }
        Ok(result)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        vec![self.base.weight.clone(), self.base.bias.clone()]
    }

    fn zero_grad(&mut self) {
        self.base.weight.zero_grad();
        self.base.bias.zero_grad();
    }

    fn train(&mut self, mode: bool) {
        self.base.training = mode;
    }

    fn name(&self) -> &str {
        "BatchNorm2d"
    }

    fn named_buffers(&self) -> Vec<(String, Tensor<B, S, T>)> {
        println!("DEBUG: BatchNorm2d::named_buffers TRAIT OVERRIDE called for {}", self.name());
        vec![
            ("running_mean".to_string(), self.base.running_mean.borrow().clone()),
            ("running_var".to_string(), self.base.running_var.borrow().clone()),
        ]
    }

    fn load_buffer(&self, name: &str, value: &Tensor<B, S, T>) -> Result<()> {
        match name {
            "running_mean" => {
                self.base.running_mean.replace(value.clone());
                Ok(())
            },
            "running_var" => {
                self.base.running_var.replace(value.clone());
                Ok(())
            },
            _ => Ok(())
        }
    }
}



impl<B, S, T> BatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Compute batch mean and variance for a channel
    fn compute_batch_stats(
        &self,
        channel_data: &[T],
        batch_size: usize,
        channel_size: usize,
    ) -> Result<(T, T)> {
        let total_samples = T::from(batch_size * channel_size).unwrap();

        // Compute mean
        let mut sum = T::from(0.0).unwrap();
        for &val in channel_data {
            sum = sum + val;
        }
        let mean = sum / total_samples;

        // Compute variance
        let mut var_sum = T::from(0.0).unwrap();
        for &val in channel_data {
            let diff = val - mean;
            var_sum = var_sum + (diff * diff);
        }
        let var = var_sum / total_samples;

        Ok((mean, var))
    }

    /// Normalize a single channel with given statistics
    fn normalize_channel(
        &self,
        channel_data: &[T],
        mean: T,
        var: T,
        weight: T,
        bias: T,
    ) -> Result<Vec<T>> {
        let eps = T::from(self.base.eps).unwrap();
        let var_eps = var + eps;
        let std = var_eps.sqrt();

        let mut normalized = Vec::with_capacity(channel_data.len());
        for &val in channel_data {
            let normalized_val = (val - mean) / std;
            let scaled = normalized_val * weight;
            let shifted = scaled + bias;
            normalized.push(shifted);
        }

        Ok(normalized)
    }

    /// Update running statistics
    fn update_running_stats(
        &self,
        batch_mean: &Tensor<B, S, T>,
        batch_var: &Tensor<B, S, T>,
    ) -> Result<()> {
        let momentum = T::from(self.base.momentum).unwrap();
        let one_minus_momentum = T::from(1.0 - self.base.momentum).unwrap();

        // Update running mean: running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        let mut running_mean = self.base.running_mean.borrow_mut();
        let new_running_mean = &*running_mean * momentum + &(*batch_mean * one_minus_momentum);
        *running_mean = new_running_mean;

        // Update running var: running_var = momentum * running_var + (1 - momentum) * batch_var
        let mut running_var = self.base.running_var.borrow_mut();
        let new_running_var = &*running_var * momentum + &(*batch_var * one_minus_momentum);
        *running_var = new_running_var;

        Ok(())
    }
}

impl<B, S, T> fmt::Display for BatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BatchNorm2d(num_features={}, eps={:.2e}, momentum={:.3}, training={})",
            self.base.num_features,
            self.base.eps,
            self.base.momentum,
            self.base.training
        )
    }
}

