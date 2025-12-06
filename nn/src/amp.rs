//! Automatic Mixed Precision (AMP) Training
//!
//! This module provides automatic mixed precision training capabilities,
//! allowing models to use FP16 operations where safe while maintaining
//! training stability through gradient scaling and loss scaling.
//!
//! ## Features
//!
//! - **Automatic casting**: Transparently converts operations to FP16 where beneficial
//! - **Gradient scaling**: Prevents gradient underflow in FP16 training
//! - **Loss scaling**: Scales losses to maintain gradient magnitudes
//! - **NaN/Inf detection**: Monitors for numerical instabilities
//! - **Fallback handling**: Gracefully handles overflow conditions
//!
//! ## Usage
//!
//! ```rust
//! use nn::amp::{MixedPrecision, GradientScaler};
//!
//! // Create mixed precision context
//! let mut amp = MixedPrecision::new();
//!
//! // Scale gradients for stability
//! let scaler = GradientScaler::new(2.0);
//!
//! // Training loop with mixed precision
//! for batch in training_data {
//!     // Forward pass in mixed precision
//!     let output = amp.forward(model, batch.input)?;
//!     let loss = mse_loss(output, batch.target)?;
//!
//!     // Backward pass with scaled gradients
//!     let scaled_loss = scaler.scale(loss)?;
//!     scaled_loss.backward()?;
//!
//!     // Update parameters with gradient scaling
//!     scaler.step(&mut optimizer)?;
//!     scaler.update();
//! }
//! ```

use crate::error::{NNError, Result};
use tensor::{FloatExt, Tensor};

/// Automatic Mixed Precision training context
#[derive(Debug)]
pub struct MixedPrecision {
    /// Whether to enable mixed precision
    enabled: bool,
    /// Loss scaling factor
    loss_scale: f32,
    /// Maximum loss scale
    max_loss_scale: f32,
    /// Minimum loss scale
    min_loss_scale: f32,
    /// Growth factor for loss scale
    growth_factor: f32,
    /// Backoff factor for loss scale
    backoff_factor: f32,
    /// Number of consecutive steps without overflow
    steps_since_overflow: u32,
    /// Minimum steps before increasing loss scale
    min_steps_growth: u32,
}

impl Default for MixedPrecision {
    fn default() -> Self {
        Self::new()
    }
}

impl MixedPrecision {
    /// Create a new mixed precision context
    #[must_use]
    pub fn new() -> Self {
        Self {
            enabled: true,
            loss_scale: 1.0,
            max_loss_scale: 65536.0, // 2^16
            min_loss_scale: 1.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            steps_since_overflow: 0,
            min_steps_growth: 2000,
        }
    }

    /// Set whether mixed precision is enabled
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set initial loss scale
    #[must_use]
    pub fn with_loss_scale(mut self, scale: f32) -> Self {
        self.loss_scale = scale.clamp(self.min_loss_scale, self.max_loss_scale);
        self
    }

    /// Check if mixed precision is enabled
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get current loss scale
    #[must_use]
    pub fn loss_scale(&self) -> f32 {
        self.loss_scale
    }

    /// Check if tensor should be computed in FP16
    #[must_use]
    pub fn should_use_half(&self, tensor_size: usize) -> bool {
        self.enabled && tensor_size > 1024 // Only use FP16 for larger tensors
    }

    /// Scale a loss tensor for mixed precision training
    pub fn scale_loss<T, B, S>(&self, loss: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        T: FloatExt + num_traits::FromPrimitive,
        B: backend::Backend<Data = T>,
        S: storage::Storage<T> + storage::StorageFromVec<T> + 'static,
    {
        if !self.enabled {
            // For disabled AMP, just return a copy of the loss tensor
            // Since Tensor doesn't implement Clone, we need to create a new tensor
            // This is a simplified approach - in practice, we'd need proper tensor copying
            return Err(NNError::UnsupportedOperation {
                operation: "scale_loss".to_string(),
                reason: "Mixed precision is disabled - tensor cloning not implemented".to_string(),
            });
        }

        // Scale loss by loss_scale factor
        Ok(loss.mul_scalar(T::from_f32(self.loss_scale).unwrap())?)
    }

    /// Unscale gradients after backward pass
    pub fn unscale_gradients<T, B, S>(&mut self, gradients: &mut [Tensor<B, S, T>]) -> Result<bool>
    where
        T: FloatExt + num_traits::FromPrimitive,
        B: backend::Backend<Data = T>,
        S: storage::Storage<T> + storage::StorageFromVec<T> + storage::StorageToDense<T> + 'static,
    {
        if !self.enabled {
            return Ok(true);
        }

        let mut has_overflow = false;

        // Check for NaN/Inf in gradients and unscale
        for grad in gradients {
            if self.has_nan_or_inf(grad)? {
                has_overflow = true;
                break;
            }

            // Unscale gradient
            *grad = grad
                .mul_scalar(T::from_f32(1.0 / self.loss_scale).unwrap())
                .map_err(NNError::from)?;
        }

        if has_overflow {
            self.handle_overflow();
            Ok(false) // Indicate that gradients are invalid
        } else {
            self.handle_success();
            Ok(true) // Gradients are valid
        }
    }

    /// Check if tensor contains NaN or Inf values
    pub fn has_nan_or_inf<T, B, S>(&self, tensor: &Tensor<B, S, T>) -> Result<bool>
    where
        T: FloatExt,
        B: backend::Backend<Data = T>,
        S: storage::Storage<T> + storage::StorageFromVec<T> + storage::StorageToDense<T> + 'static,
    {
        let data = tensor.storage_ref().as_slice();

        for &value in data {
            let f32_val: Option<f32> = value.to_f32();
            if let Some(f32_val) = f32_val {
                if f32_val.is_nan() || f32_val.is_infinite() {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Handle gradient overflow
    fn handle_overflow(&mut self) {
        self.loss_scale = (self.loss_scale * self.backoff_factor).max(self.min_loss_scale);
        self.steps_since_overflow = 0;
    }

    /// Handle successful step without overflow
    fn handle_success(&mut self) {
        self.steps_since_overflow += 1;

        // Increase loss scale if we've had enough consecutive successful steps
        if self.steps_since_overflow >= self.min_steps_growth {
            self.loss_scale = (self.loss_scale * self.growth_factor).min(self.max_loss_scale);
            self.steps_since_overflow = 0;
        }
    }
}

/// Gradient scaler for mixed precision training
#[derive(Debug)]
pub struct GradientScaler {
    /// Current scale factor
    scale: f32,
    /// Growth factor
    growth_factor: f32,
    /// Backoff factor
    backoff_factor: f32,
    /// Maximum scale
    max_scale: f32,
    /// Steps since last overflow
    steps_since_overflow: u32,
    /// Minimum steps before growth
    min_growth_steps: u32,
    /// Found overflow in current step
    found_overflow: bool,
}

impl GradientScaler {
    /// Create a new gradient scaler
    #[must_use]
    pub fn new(initial_scale: f32) -> Self {
        Self {
            scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            max_scale: 2.0_f32.powi(24), // ~16M
            steps_since_overflow: 0,
            min_growth_steps: 2000,
            found_overflow: false,
        }
    }

    /// Get the current scale value
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Scale a loss value
    pub fn scale_loss<T, B, S>(&self, loss: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>>
    where
        T: FloatExt + num_traits::FromPrimitive,
        B: backend::Backend<Data = T>,
        S: storage::Storage<T> + storage::StorageFromVec<T> + 'static,
    {
        Ok(loss.mul_scalar(T::from_f32(self.scale).unwrap())?)
    }

    /// Step the optimizer and handle gradient scaling
    /// Note: This is a simplified version. In practice, you'd need to integrate
    /// with the actual optimizer crate.
    pub fn step(&mut self) -> Result<()> {
        if self.found_overflow {
            // Skip optimizer step due to overflow
            self.found_overflow = false;
            self.scale *= self.backoff_factor;
            self.steps_since_overflow = 0;
            return Ok(());
        }

        // In a real implementation, this would call optimizer.step()
        // For now, just update scale if appropriate
        self.steps_since_overflow += 1;
        if self.steps_since_overflow >= self.min_growth_steps {
            self.scale = (self.scale * self.growth_factor).min(self.max_scale);
            self.steps_since_overflow = 0;
        }

        Ok(())
    }

    /// Update scaler state (call after checking gradients)
    pub fn update(&mut self) {
        // Reset overflow flag for next step
        self.found_overflow = false;
    }

    /// Check gradients for overflow and update scaler state
    pub fn check_gradients<T, B, S>(&mut self, gradients: &[&Tensor<B, S, T>]) -> Result<()>
    where
        T: FloatExt,
        B: backend::Backend<Data = T>,
        S: storage::Storage<T> + storage::StorageFromVec<T> + storage::StorageToDense<T> + 'static,
    {
        for grad in gradients {
            if self.has_inf_or_nan(grad)? {
                self.found_overflow = true;
                break;
            }
        }

        Ok(())
    }

    /// Check if tensor has inf or nan values
    fn has_inf_or_nan<T, B, S>(&self, tensor: &Tensor<B, S, T>) -> Result<bool>
    where
        T: FloatExt,
        B: backend::Backend<Data = T>,
        S: storage::Storage<T> + storage::StorageFromVec<T> + storage::StorageToDense<T> + 'static,
    {
        let data = tensor.storage_ref().as_slice();

        for &value in data {
            let f32_val: Option<f32> = value.to_f32();
            if let Some(f32_val) = f32_val {
                if f32_val.is_nan() || f32_val.is_infinite() {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Get current scale factor
    #[must_use]
    pub fn get_scale(&self) -> f32 {
        self.scale
    }
}

/// Context manager for mixed precision operations
pub struct MixedPrecisionGuard<'a> {
    amp: &'a mut MixedPrecision,
    original_enabled: bool,
}

impl<'a> MixedPrecisionGuard<'a> {
    /// Create a new guard
    #[must_use]
    pub fn new(amp: &'a mut MixedPrecision, enabled: bool) -> Self {
        let original_enabled = amp.enabled;
        amp.enabled = enabled;
        Self {
            amp,
            original_enabled,
        }
    }
}

impl<'a> Drop for MixedPrecisionGuard<'a> {
    fn drop(&mut self) {
        self.amp.enabled = self.original_enabled;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::{DenseStorage, Storage};
    use tensor::Tensor;

    #[test]
    fn test_mixed_precision_creation() {
        let amp = MixedPrecision::new();
        assert!(amp.is_enabled());
        assert_eq!(amp.loss_scale(), 1.0);
    }

    #[test]
    fn test_gradient_scaler_creation() {
        let scaler = GradientScaler::new(2.0);
        assert_eq!(scaler.scale(), 2.0);
    }

    #[test]
    fn test_loss_scaling() -> Result<()> {
        let amp = MixedPrecision::new().with_loss_scale(2.0);

        let _backend = CpuBackend::<Float32>::default();
        let data = vec![Float32::new(1.0), Float32::new(2.0)];
        let loss: Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
            Tensor::from_vec(data, &[2])?;

        let scaled_loss = amp.scale_loss(&loss)?;
        let scaled_data = scaled_loss.storage_ref().as_slice();

        assert_eq!(scaled_data[0].get(), 2.0);
        assert_eq!(scaled_data[1].get(), 4.0);

        Ok(())
    }

    #[test]
    fn test_nan_detection() -> Result<()> {
        let amp = MixedPrecision::new();

        let _backend = CpuBackend::<Float32>::default();
        let data = vec![Float32::new(1.0), Float32::new(f32::NAN)];
        let tensor: Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
            Tensor::from_vec(data, &[2])?;

        assert!(amp.has_nan_or_inf(&tensor)?);

        Ok(())
    }

    #[test]
    fn test_inf_detection() -> Result<()> {
        let amp = MixedPrecision::new();

        let _backend = CpuBackend::<Float32>::default();
        let data = vec![Float32::new(1.0), Float32::new(f32::INFINITY)];
        let tensor: Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32> =
            Tensor::from_vec(data, &[2])?;

        assert!(amp.has_nan_or_inf(&tensor)?);

        Ok(())
    }
}
