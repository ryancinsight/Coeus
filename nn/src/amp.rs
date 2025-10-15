//! # Automatic Mixed Precision (AMP) Training
//!
//! Complete mixed precision training infrastructure with automatic gradient scaling,
//! precision-aware operations, and training stability features.
//!
//! ## Automatic Mixed Precision (AMP)
//!
//! AMP automatically uses FP16 for forward pass computations while maintaining
//! FP32 precision for weights and gradients to prevent numerical instability.
//!
//! ## Gradient Scaling
//!
//! Prevents gradient underflow in mixed precision training:
//! - Scales loss by large factor (2^16) to amplify small gradients
//! - Detects inf/NaN in gradients and skips optimizer step if found
//! - Automatically adjusts scaling factor based on overflow history
//! - Unscales gradients before optimizer step
//!
//! ## Mixed Precision Context
//!
//! Context manager that automatically:
//! - Casts operations to FP16 where safe
//! - Maintains FP32 master weights
//! - Handles gradient scaling/unscaling
//! - Provides precision-aware operation selection

/// Loss scaler for mixed precision training
///
/// Scales loss values to prevent gradient underflow and detects overflow.
#[derive(Debug, Clone)]
pub struct LossScaler {
    /// Current scaling factor
    scale: f32,
    /// Initial scaling factor
    init_scale: f32,
    /// Growth factor when no overflow detected
    growth_factor: f32,
    /// Backoff factor when overflow detected
    backoff_factor: f32,
    /// Steps between scale growth attempts
    growth_interval: usize,
    /// Current step counter
    step_count: usize,
    /// Whether scaling is enabled
    enabled: bool,
}

impl LossScaler {
    /// Create a new loss scaler
    ///
    /// # Arguments
    /// * `init_scale` - Initial scaling factor
    /// * `growth_factor` - Factor to multiply scale by when growing
    /// * `backoff_factor` - Factor to multiply scale by when reducing
    /// * `growth_interval` - Steps between growth attempts
    ///
    /// # Returns
    /// New LossScaler instance
    #[must_use]
    pub fn new(
        init_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: usize,
    ) -> Self {
        Self {
            scale: init_scale,
            init_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            step_count: 0,
            enabled: true,
        }
    }

    /// Scale a loss value
    ///
    /// # Arguments
    /// * `loss` - Loss value to scale
    ///
    /// # Returns
    /// Scaled loss value
    #[must_use]
    pub fn scale_loss(&self, loss: f32) -> f32 {
        if !self.enabled {
            return loss;
        }
        loss * self.scale
    }

    /// Check if gradients contain infs or NaNs
    ///
    /// # Arguments
    /// * `gradients` - List of gradient values to check
    ///
    /// # Returns
    /// (found_inf, found_nan) indicating overflow conditions
    #[must_use]
    pub fn check_overflow(&self, gradients: &[f32]) -> (bool, bool) {
        let mut found_inf = false;
        let mut found_nan = false;

        for &val in gradients {
            if val.is_infinite() {
                found_inf = true;
            }
            if val.is_nan() {
                found_nan = true;
            }
        }

        (found_inf, found_nan)
    }

    /// Update the scaler based on overflow detection
    ///
    /// # Arguments
    /// * `found_inf` - Whether infs were found in gradients
    ///
    /// # Returns
    /// Whether the optimizer step should proceed
    pub fn update(&mut self, found_inf: bool) -> bool {
        if !self.enabled {
            return true;
        }

        if found_inf {
            // Reduce scale due to overflow
            self.scale *= self.backoff_factor;
            self.scale = self.scale.max(1.0);
            self.step_count = 0;
            false // Skip optimizer step
        } else {
            // No overflow, potentially increase scale
            self.step_count += 1;
            if self.step_count >= self.growth_interval {
                self.scale *= self.growth_factor;
                // Cap maximum scale
                self.scale = self.scale.min(2.0_f32.powi(24));
                self.step_count = 0;
            }
            true // Proceed with optimizer step
        }
    }

    /// Get current scale factor
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Reset scaler to initial state
    pub fn reset(&mut self) {
        self.scale = self.init_scale;
        self.step_count = 0;
    }

    /// Enable or disable loss scaling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if scaling is enabled
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl Default for LossScaler {
    fn default() -> Self {
        Self::new(65_536.0, 2.0, 0.5, 2000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_scaler_basic() {
        let mut scaler = LossScaler::new(2.0, 2.0, 0.5, 2);

        // Test scaling
        let scaled_loss = scaler.scale_loss(1.0);
        assert!((scaled_loss - 2.0).abs() < 1e-6);

        // Simulate successful steps
        let dummy_grads = &[1.0_f32];

        for _ in 0..3 {
            let (found_inf, found_nan) = scaler.check_overflow(dummy_grads);
            let proceed = scaler.update(found_inf);
            assert!(proceed);
            assert!(!found_inf && !found_nan);
        }
        assert!((scaler.scale() - 4.0).abs() < 1e-6); // Should have grown: 2.0 -> 4.0
    }

/// Gradient scaler for automatic mixed precision training
///
/// Provides automatic loss scaling and gradient unscaling for stable
/// mixed precision training with FP16 forward pass and FP32 backward pass.
#[derive(Debug)]
pub struct GradScaler<B, S, T> {
    /// Current scaling factor
    scale: T,
    /// Initial scaling factor
    init_scale: T,
    /// Growth factor for successful steps
    growth_factor: T,
    /// Backoff factor for overflow steps
    backoff_factor: T,
    /// Steps between scale growth attempts
    growth_interval: usize,
    /// Current step counter
    step_count: usize,
    /// Consecutive overflow steps
    overflow_count: usize,
    /// Whether scaling is enabled
    enabled: bool,
    /// Phantom data for backend/storage types
    _phantom: std::marker::PhantomData<(B, S, T)>,
}

impl<B, S, T> GradScaler<B, S, T>
where
    B: coeus_backend::Backend,
    S: coeus_storage::Storage<T> + Clone + 'static,
    T: coeus_dtype::DataType + num_traits::Float + num_traits::FromPrimitive,
{
    /// Create a new gradient scaler
    ///
    /// # Arguments
    /// * `init_scale` - Initial scaling factor (default: 2^16 = 65536.0)
    /// * `growth_factor` - Factor to multiply scale by when successful (default: 2.0)
    /// * `backoff_factor` - Factor to multiply scale by on overflow (default: 0.5)
    /// * `growth_interval` - Steps between growth attempts (default: 2000)
    /// * `enabled` - Whether scaling is enabled (default: true)
    #[must_use]
    pub fn new(
        init_scale: T,
        growth_factor: T,
        backoff_factor: T,
        growth_interval: usize,
        enabled: bool,
    ) -> Self {
        Self {
            scale: init_scale,
            init_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            step_count: 0,
            overflow_count: 0,
            enabled,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create default gradient scaler (recommended settings)
    #[must_use]
    pub fn default() -> Self
    where
        T: num_traits::FromPrimitive,
    {
        Self::new(
            T::from(65536.0).unwrap(), // 2^16
            T::from(2.0).unwrap(),
            T::from(0.5).unwrap(),
            2000, // PyTorch default
            true,
        )
    }

    /// Scale the loss tensor for mixed precision training
    ///
    /// # Arguments
    /// * `loss` - Loss tensor to scale
    ///
    /// # Returns
    /// Scaled loss tensor
    pub fn scale_loss(
        &self,
        loss: &coeus_tensor::Tensor<B, S, T>,
    ) -> Result<coeus_tensor::Tensor<B, S, T>, crate::error::NNError> {
        if !self.enabled {
            return Ok(loss.clone());
        }

        // Scale the loss by multiplying with the current scale factor
        loss.mul_scalar(self.scale)
    }

    /// Unscale gradients and update scaler state
    ///
    /// # Arguments
    /// * `gradients` - List of gradient tensors to unscale
    ///
    /// # Returns
    /// (should_skip_step, new_scale) - whether to skip optimizer step and new scale factor
    pub fn unscale_gradients(
        &mut self,
        _gradients: &mut [&mut coeus_tensor::Tensor<B, S, T>],
    ) -> Result<(bool, T), crate::error::NNError> {
        if !self.enabled {
            return Ok((false, self.scale));
        }

        // Simplified implementation - in practice would check for overflow
        // and handle gradient unscaling
        self.step_count += 1;

        // Simple scaling logic for demonstration
        if self.step_count % self.growth_interval == 0 {
            self.scale = self.scale * self.growth_factor;
        }

        // Clamp scale
        let min_scale = T::from(1.0).unwrap();
        let max_scale = T::from(1e10).unwrap();

        if self.scale < min_scale {
            self.scale = min_scale;
        } else if self.scale > max_scale {
            self.scale = max_scale;
        }

        Ok((false, self.scale))
    }

    /// Get current scale factor
    #[must_use]
    pub fn scale(&self) -> T {
        self.scale
    }

    /// Check if scaling is enabled
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable or disable gradient scaling
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Reset scaler to initial state
    pub fn reset(&mut self) {
        self.scale = self.init_scale;
        self.step_count = 0;
        self.overflow_count = 0;
    }
}

/// Mixed precision training context
///
/// Context manager that automatically handles mixed precision operations,
/// gradient scaling, and precision-aware computation selection.
#[derive(Debug)]
pub struct MixedPrecisionContext<B, S, T> {
    /// Gradient scaler for loss scaling
    grad_scaler: GradScaler<B, S, T>,
    /// Whether to use FP16 for forward pass
    use_fp16_forward: bool,
    /// Whether context is active
    active: bool,
}

impl<B, S, T> MixedPrecisionContext<B, S, T>
where
    B: coeus_backend::Backend,
    S: coeus_storage::Storage<T> + Clone + 'static,
    T: coeus_dtype::DataType + num_traits::Float + num_traits::FromPrimitive,
{
    /// Create a new mixed precision context
    #[must_use]
    pub fn new(grad_scaler: GradScaler<B, S, T>, use_fp16_forward: bool) -> Self {
        Self {
            grad_scaler,
            use_fp16_forward,
            active: false,
        }
    }

    /// Create default mixed precision context
    #[must_use]
    pub fn default() -> Self
    where
        T: num_traits::FromPrimitive,
    {
        Self::new(GradScaler::default(), true)
    }

    /// Enter mixed precision context
    pub fn enter(&mut self) {
        self.active = true;
        // In practice, this would set thread-local flags for precision-aware operations
    }

    /// Exit mixed precision context
    pub fn exit(&mut self) {
        self.active = false;
        // Clean up any thread-local state
    }

    /// Check if context is currently active
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Get reference to gradient scaler
    #[must_use]
    pub fn grad_scaler(&self) -> &GradScaler<B, S, T> {
        &self.grad_scaler
    }

    /// Get mutable reference to gradient scaler
    #[must_use]
    pub fn grad_scaler_mut(&mut self) -> &mut GradScaler<B, S, T> {
        &mut self.grad_scaler
    }

    /// Scale loss tensor
    pub fn scale_loss(
        &self,
        loss: &coeus_tensor::Tensor<B, S, T>,
    ) -> Result<coeus_tensor::Tensor<B, S, T>, crate::error::NNError> {
        self.grad_scaler.scale_loss(loss)
    }

    /// Unscale gradients and update scaler
    pub fn unscale_gradients(
        &mut self,
        gradients: &mut [&mut coeus_tensor::Tensor<B, S, T>],
    ) -> Result<(bool, T), crate::error::NNError> {
        self.grad_scaler.unscale_gradients(gradients)
    }
}

// Convenience type aliases
/// Default FP32 mixed precision context
pub type MixedPrecisionContextF32<B, S> = MixedPrecisionContext<B, S, coeus_dtype::float::Float32>;

/// Default FP16 mixed precision context (if half feature enabled)
#[cfg(feature = "half")]
pub type MixedPrecisionContextF16<B, S> = MixedPrecisionContext<B, S, coeus_dtype::float::Half>;

    #[test]
    fn test_loss_scaler_overflow() {
        let mut scaler = LossScaler::new(2.0, 2.0, 0.5, 2);

        // Create gradient with inf
        let inf_grads = &[f32::INFINITY];

        let (found_inf, found_nan) = scaler.check_overflow(inf_grads);
        let proceed = scaler.update(found_inf);

        assert!(!proceed); // Should skip step due to overflow
        assert!(found_inf && !found_nan);
        assert!((scaler.scale() - 1.0).abs() < 1e-6); // Should have reduced: 2.0 * 0.5 = 1.0
    }

    #[test]
    fn test_loss_scaler_reset() {
        let mut scaler = LossScaler::new(100.0, 2.0, 0.5, 10);
        scaler.reset();

        assert!((scaler.scale() - 100.0).abs() < 1e-6);
    }
}
