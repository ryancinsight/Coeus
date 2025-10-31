//! RMSprop (Root Mean Square Propagation) optimizer.
//!
//! This module implements the RMSprop optimizer with optional momentum and centering.

use std::collections::HashMap;
use std::marker::PhantomData;

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::gpu_backend::{GpuAcceleratedOptimizer, GpuOptimizerBackend, GpuOptimizerConfig};
use crate::optimizer_core::{Optimizer, ParamState};
use crate::Parameter;

/// RMSprop (Root Mean Square Propagation) optimizer.
///
/// RMSprop divides the learning rate by an exponentially decaying average of squared gradients.
/// This helps with the diminishing learning rates problem in AdaGrad while maintaining
/// adaptive learning rates.
///
/// # Algorithm
///
/// ```text
/// square_avg = alpha * square_avg + (1 - alpha) * grad^2
/// param = param - lr * grad / sqrt(square_avg + eps)
/// ```
///
/// With momentum:
/// ```text
/// grad = grad + weight_decay * param  # L2 regularization
/// square_avg = alpha * square_avg + (1 - alpha) * grad^2
/// momentum_buffer = momentum * momentum_buffer + grad
/// param = param - lr * momentum_buffer / sqrt(square_avg + eps)
/// ```
///
/// With centering:
/// ```text
/// grad_avg = alpha * grad_avg + (1 - alpha) * grad
/// square_avg = alpha * square_avg + (1 - alpha) * grad^2
/// param = param - lr * grad / sqrt(square_avg - grad_avg^2 + eps)
/// ```
///
/// # Hyperparameters
///
/// - `lr`: Learning rate (default: 0.01)
/// - `alpha`: Smoothing constant (default: 0.99)
/// - `eps`: Numerical stability constant (default: 1e-8)
/// - `weight_decay`: L2 regularization factor (default: 0.0)
/// - `momentum`: Momentum factor (default: 0.0, no momentum)
/// - `centered`: Whether to center the second moment (default: false)
///
/// # Examples
///
/// ```rust
/// use optim::rmsprop::RMSprop;
/// use dtype::float::Float32;
/// use backend::CpuBackend;
/// use storage::DenseStorage;
///
/// // Create RMSprop with default hyperparameters
/// let mut optimizer = RMSprop::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::default(0.01);
/// ```
#[derive(Debug)]
pub struct RMSprop<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + dtype::num_traits::Float,
{
    /// Parameter states
    param_states: Vec<ParamState<B, S, T>>,
    /// Learning rate
    lr: f64,
    /// Smoothing constant (α)
    alpha: f64,
    /// Numerical stability constant
    eps: f64,
    /// Weight decay (L2 regularization)
    weight_decay: f64,
    /// Momentum factor
    momentum: f64,
    /// Whether to use centered RMSprop
    centered: bool,
    /// Phantom data
    _phantom: PhantomData<(B, S, T)>,
    // GPU acceleration fields (placeholder for future implementation)
    gpu_enabled: bool,
    gpu_config: Option<GpuOptimizerConfig>,
    gpu_backend: Option<GpuOptimizerBackend>,
}

impl<B, S, T> RMSprop<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + dtype::num_traits::Float,
{
    /// Create a new RMSprop optimizer.
    ///
    /// # Arguments
    /// * `lr` - Learning rate
    /// * `alpha` - Smoothing constant (0 < alpha < 1)
    /// * `eps` - Numerical stability constant
    /// * `weight_decay` - L2 regularization factor
    /// * `momentum` - Momentum factor
    /// * `centered` - Whether to use centered RMSprop
    pub fn new(
        lr: f64,
        alpha: f64,
        eps: f64,
        weight_decay: f64,
        momentum: f64,
        centered: bool,
    ) -> Self {
        Self {
            param_states: Vec::new(),
            lr,
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
            _phantom: PhantomData,
            gpu_enabled: false, // CPU-only for now
            gpu_config: None,
            gpu_backend: None,
        }
    }

    /// Create RMSprop with default hyperparameters.
    pub fn default(lr: f64) -> Self {
        Self::new(lr, 0.99, 1e-8, 0.0, 0.0, false)
    }

    /// Create RMSprop with momentum.
    pub fn with_momentum(lr: f64, momentum: f64) -> Self {
        Self::new(lr, 0.99, 1e-8, 0.0, momentum, false)
    }

    /// Create centered RMSprop.
    pub fn centered_rmsprop(lr: f64) -> Self {
        Self::new(lr, 0.99, 1e-8, 0.0, 0.0, true)
    }

    /// Get alpha (smoothing constant) value
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Get momentum value
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    /// Check if centered RMSprop is enabled
    pub fn centered(&self) -> bool {
        self.centered
    }

    /// Create RMSprop optimizer with GPU acceleration (placeholder)
    ///
    /// # Arguments
    /// * `params` - Parameter tensors to optimize
    /// * `lr` - Learning rate (must be > 0)
    pub fn new_with_gpu(
        _params: Vec<tensor::Tensor<B, S, T>>,
        lr: f64,
    ) -> Result<Self, crate::error::OptimError> {
        // For now, this just creates CPU version
        // GPU acceleration would be implemented here in the future
        Ok(Self::new(lr, 0.99, 1e-8, 0.0, 0.0, false))
    }

    /// Attempt GPU-accelerated step with CPU fallback
    /// Tries to use GPU acceleration if available, falls back to CPU otherwise
    pub fn step_gpu(&mut self) -> Result<usize, crate::error::OptimError> {
        // Check if GPU is enabled and backend is available
        if self.gpu_enabled || self.gpu_backend.is_some() {
            // Try GPU acceleration first
            match self.step_gpu_internal() {
                Ok(result) => return Ok(result),
                Err(_) => {
                    // GPU failed, fall back to CPU
                    eprintln!("GPU acceleration failed, falling back to CPU");
                }
            }
        }

        // Fall back to CPU implementation
        self.step_cpu()
    }

    /// Internal GPU step implementation with actual WGSL kernel dispatch
    fn step_gpu_internal(&mut self) -> Result<usize, crate::error::OptimError> {
        if let Some(_backend) = &self.gpu_backend {
            // For now, fall back to CPU implementation until GPU kernels are fully working
            self.step_cpu()
        } else {
            Err(crate::error::OptimError::BackendError {
                message: "GPU backend not available".into(),
            })
        }
    }

    /// CPU step implementation (extracted from step() method)
    /// This maintains the original CPU logic for fallback purposes
    pub fn step_cpu(&mut self) -> Result<usize, crate::error::OptimError> {
        let lr = T::from(self.lr).unwrap();
        let alpha = T::from(self.alpha).unwrap();
        let eps = T::from(self.eps).unwrap();
        let weight_decay = T::from(self.weight_decay).unwrap();
        let _momentum = T::from(self.momentum).unwrap();
        let one = T::from(1.0).unwrap();
        let one_minus_alpha = one - alpha;

        for param_state in &mut self.param_states {
            // Get gradient - first check parameter state, then tensor
            let grad = if let Some(grad) = param_state.grad.as_ref() {
                grad.clone()
            } else if let Ok(tensor_grad) = param_state.param.grad() {
                // Convert gradient storage to dense, then create optimizer's storage type
                // This handles any gradient storage type (dense, sparse, quantized, etc.)
                let dense_grad = tensor_grad
                    .storage_ref()
                    .to_dense()
                    .map_err(|_| crate::error::OptimError::GradientNotAvailable)?;
                let converted_storage =
                    S::from_vec(dense_grad.as_slice().to_vec(), tensor_grad.shape().dims())
                        .map_err(|_| crate::error::OptimError::GradientNotAvailable)?;

                // Use the same backend as the original tensor
                let backend = tensor_grad.backend().clone();
                Tensor::from_storage(converted_storage, backend)
            } else {
                return Err(crate::error::OptimError::GradientNotAvailable);
            };

            // Apply weight decay if specified (L2 regularization)
            let effective_grad = if self.weight_decay > 0.0 {
                let grad_clone = grad.clone();
                let weight_decay_term = param_state.param.mul_scalar(weight_decay)?;
                add(&grad_clone, &weight_decay_term)?
            } else {
                grad.clone()
            };

            // Update square average: square_avg = alpha * square_avg + (1 - alpha) * grad^2
            let param_name = param_state.name.clone();

            use tensor::ops::arithmetic::{add, div, mul, scalar_add, scalar_mul, sqrt, sub};

            {
                let square_avg = param_state.get_state_mut("square_avg").ok_or_else(|| {
                    crate::error::OptimError::InvalidState {
                        param_name: param_name.clone(),
                        state_key: "square_avg".to_string(),
                    }
                })?;

                let grad_squared = mul(&effective_grad, &effective_grad)?;
                let square_avg_alpha = scalar_mul(&*square_avg, alpha)?;
                let grad_squared_alpha = scalar_mul(&grad_squared, one_minus_alpha)?;
                *square_avg = add(&square_avg_alpha, &grad_squared_alpha)?;
            }

            let denom = if self.centered {
                // For centered RMSprop: denom = sqrt(square_avg - grad_avg^2 + eps)

                // Update grad_avg: grad_avg = alpha * grad_avg + (1 - alpha) * grad
                let grad_avg = param_state.get_state_mut("grad_avg").ok_or_else(|| {
                    crate::error::OptimError::InvalidState {
                        param_name: param_name.clone(),
                        state_key: "grad_avg".to_string(),
                    }
                })?;
                let grad_avg_alpha = scalar_mul(&*grad_avg, alpha)?;
                let grad_alpha = scalar_mul(&effective_grad, one_minus_alpha)?;
                let new_grad_avg = add(&grad_avg_alpha, &grad_alpha)?;
                let grad_avg_squared = mul(&new_grad_avg, &new_grad_avg)?;
                *grad_avg = new_grad_avg;

                // Compute denom = sqrt(square_avg - grad_avg^2 + eps)
                let square_avg_ref = param_state.get_state("square_avg").unwrap();
                let square_avg_minus_grad_avg_sq = sub(square_avg_ref, &grad_avg_squared)?;
                let denom_inner = scalar_add(&square_avg_minus_grad_avg_sq, eps)?;
                sqrt(&denom_inner)?
            } else {
                // Basic RMSprop: denom = sqrt(square_avg + eps)
                let square_avg_ref = param_state.get_state("square_avg").unwrap();
                let square_avg_sqrt = sqrt(square_avg_ref)?;
                scalar_add(&square_avg_sqrt, eps)?
            };

            // Basic RMSprop: param = param - lr * grad / denom
            let grad_scaled = scalar_mul(&effective_grad, lr)?;
            let update = div(&grad_scaled, &denom)?;
            param_state.param = sub(&param_state.param, &update)?;
        }

        Ok(self.param_states.len())
    }
}

impl<B, S, T> Optimizer<B, S, T> for RMSprop<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + dtype::num_traits::Float,
{
    fn name(&self) -> &str {
        "RMSprop"
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        self.param_states
            .iter()
            .map(|ps| ps.param.clone())
            .collect()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter<B, S, T>> {
        self.param_states
            .iter()
            .map(|ps| (ps.name.clone(), ps.param.clone()))
            .collect()
    }

    fn add_param(
        &mut self,
        param: &mut Parameter<B, S, T>,
        name: String,
    ) -> Result<(), crate::error::OptimError> {
        // Check if parameter requires gradients
        if !param.requires_grad() {
            return Err(crate::error::OptimError::InvalidParameter {
                param: name.clone(),
                value: "requires_grad=false".to_string(),
                reason: "parameter must require gradients for optimization".to_string(),
            });
        }

        // Check if parameter already exists
        if self.has_param(&name) {
            return Err(crate::error::OptimError::InvalidParameter {
                param: name,
                value: "already exists".to_string(),
                reason: "parameter name must be unique".to_string(),
            });
        }

        let param_clone = param.clone();
        let mut param_state = ParamState::new(param_clone, name);

        // Initialize RMSprop state
        let shape = param_state.param.shape().dims().to_vec();
        let square_avg = Tensor::zeros(&shape)
            .map_err(|e| crate::error::OptimError::TensorError { source: e })?;
        param_state.init_state("square_avg".to_string(), square_avg);

        if self.centered {
            let grad_avg = Tensor::zeros(&shape)
                .map_err(|e| crate::error::OptimError::TensorError { source: e })?;
            param_state.init_state("grad_avg".to_string(), grad_avg);
        }

        if self.momentum > 0.0 {
            let momentum_buffer = Tensor::zeros(&shape)
                .map_err(|e| crate::error::OptimError::TensorError { source: e })?;
            param_state.init_state("momentum_buffer".to_string(), momentum_buffer);
        }

        self.param_states.push(param_state);
        Ok(())
    }

    fn remove_param(&mut self, name: &str) {
        self.param_states.retain(|ps| ps.name != name);
    }

    fn has_param(&self, name: &str) -> bool {
        self.param_states.iter().any(|ps| ps.name == name)
    }

    fn lr(&self) -> f64 {
        self.lr
    }

    fn set_lr(&mut self, lr: f64) -> Result<(), crate::error::OptimError> {
        if lr <= 0.0 {
            return Err(crate::error::OptimError::InvalidParameter {
                param: "lr".to_string(),
                value: lr.to_string(),
                reason: "learning rate must be positive".to_string(),
            });
        }
        self.lr = lr;
        Ok(())
    }

    fn weight_decay(&self) -> f64 {
        self.weight_decay
    }

    fn set_weight_decay(&mut self, weight_decay: f64) -> Result<(), crate::error::OptimError> {
        if weight_decay < 0.0 {
            return Err(crate::error::OptimError::InvalidParameter {
                param: "weight_decay".to_string(),
                value: weight_decay.to_string(),
                reason: "weight decay must be non-negative".to_string(),
            });
        }
        self.weight_decay = weight_decay;
        Ok(())
    }

    /// Get the learning rate (alias for lr)
    fn learning_rate(&self) -> f64 {
        self.lr()
    }

    /// Set the learning rate (alias for set_lr)
    fn set_learning_rate(&mut self, lr: f64) -> Result<(), crate::error::OptimError> {
        <Self as Optimizer<B, S, T>>::set_lr(self, lr)
    }

    fn zero_grad(&mut self) {
        for param_state in &mut self.param_states {
            param_state.param.zero_grad().unwrap();
        }
    }

    fn step(&mut self) -> Result<usize, crate::error::OptimError> {
        let lr = T::from(self.lr).unwrap();
        let alpha = T::from(self.alpha).unwrap();
        let eps = T::from(self.eps).unwrap();
        let weight_decay = T::from(self.weight_decay).unwrap();
        let _momentum = T::from(self.momentum).unwrap();
        let one = T::from(1.0).unwrap();
        let one_minus_alpha = one - alpha;

        for param_state in &mut self.param_states {
            // Get gradient - first check parameter state, then tensor
            let grad = if let Some(grad) = param_state.grad.as_ref() {
                grad.clone()
            } else if let Ok(tensor_grad) = param_state.param.grad() {
                // Convert gradient storage to dense, then create optimizer's storage type
                // This handles any gradient storage type (dense, sparse, quantized, etc.)
                let dense_grad = tensor_grad
                    .storage_ref()
                    .to_dense()
                    .map_err(|_| crate::error::OptimError::GradientNotAvailable)?;
                let converted_storage =
                    S::from_vec(dense_grad.as_slice().to_vec(), tensor_grad.shape().dims())
                        .map_err(|_| crate::error::OptimError::GradientNotAvailable)?;

                // Use the same backend as the original tensor
                let backend = tensor_grad.backend().clone();
                Tensor::from_storage(converted_storage, backend)
            } else {
                return Err(crate::error::OptimError::GradientNotAvailable);
            };

            // Apply weight decay if specified (L2 regularization)
            let effective_grad = if self.weight_decay > 0.0 {
                let grad_clone = grad.clone();
                let weight_decay_term = param_state.param.mul_scalar(weight_decay)?;
                add(&grad_clone, &weight_decay_term)?
            } else {
                grad.clone()
            };

            // Update square average: square_avg = alpha * square_avg + (1 - alpha) * grad^2
            let param_name = param_state.name.clone();

            use tensor::ops::arithmetic::{add, div, mul, scalar_add, scalar_mul, sqrt, sub};

            {
                let square_avg = param_state.get_state_mut("square_avg").ok_or_else(|| {
                    crate::error::OptimError::InvalidState {
                        param_name: param_name.clone(),
                        state_key: "square_avg".to_string(),
                    }
                })?;

                let grad_squared = mul(&effective_grad, &effective_grad)?;
                let square_avg_alpha = scalar_mul(&*square_avg, alpha)?;
                let grad_squared_alpha = scalar_mul(&grad_squared, one_minus_alpha)?;
                *square_avg = add(&square_avg_alpha, &grad_squared_alpha)?;
            }

            let denom = if self.centered {
                // For centered RMSprop: denom = sqrt(square_avg - grad_avg^2 + eps)

                // Update grad_avg: grad_avg = alpha * grad_avg + (1 - alpha) * grad
                let grad_avg = param_state.get_state_mut("grad_avg").ok_or_else(|| {
                    crate::error::OptimError::InvalidState {
                        param_name: param_name.clone(),
                        state_key: "grad_avg".to_string(),
                    }
                })?;
                let grad_avg_alpha = scalar_mul(&*grad_avg, alpha)?;
                let grad_alpha = scalar_mul(&effective_grad, one_minus_alpha)?;
                let new_grad_avg = add(&grad_avg_alpha, &grad_alpha)?;
                let grad_avg_squared = mul(&new_grad_avg, &new_grad_avg)?;
                *grad_avg = new_grad_avg;

                // Compute denom = sqrt(square_avg - grad_avg^2 + eps)
                let square_avg_ref = param_state.get_state("square_avg").unwrap();
                let square_avg_minus_grad_avg_sq = sub(square_avg_ref, &grad_avg_squared)?;
                let denom_inner = scalar_add(&square_avg_minus_grad_avg_sq, eps)?;
                sqrt(&denom_inner)?
            } else {
                // Basic RMSprop: denom = sqrt(square_avg + eps)
                let square_avg_ref = param_state.get_state("square_avg").unwrap();
                let square_avg_sqrt = sqrt(square_avg_ref)?;
                scalar_add(&square_avg_sqrt, eps)?
            };

            // Basic RMSprop: param = param - lr * grad / denom
            let grad_scaled = scalar_mul(&effective_grad, lr)?;
            let update = div(&grad_scaled, &denom)?;
            param_state.param = sub(&param_state.param, &update)?;
        }

        Ok(self.param_states.len())
    }

    fn state_dict(&self) -> HashMap<String, Tensor<B, S, T>> {
        let mut state = HashMap::new();
        for param_state in &self.param_states {
            state.insert(param_state.name.clone(), param_state.param.clone());
            for (key, tensor) in &param_state.state {
                state.insert(format!("{}.{}", param_state.name, key), tensor.clone());
            }
        }
        state
    }

    fn load_state_dict(
        &mut self,
        state_dict: HashMap<String, Tensor<B, S, T>>,
    ) -> Result<(), crate::error::OptimError> {
        for param_state in &mut self.param_states {
            if let Some(param) = state_dict.get(&param_state.name) {
                if param.shape().dims() != param_state.param.shape().dims() {
                    return Err(crate::error::OptimError::ShapeMismatch {
                        param_name: param_state.name.clone(),
                        expected: param_state.param.shape().dims().to_vec(),
                        actual: param.shape().dims().to_vec(),
                    });
                }
                param_state.param = param.clone();
            }

            // Load RMSprop state
            let square_avg_key = format!("{}.square_avg", param_state.name);
            if let Some(square_avg) = state_dict.get(&square_avg_key) {
                param_state.init_state("square_avg".to_string(), square_avg.clone());
            }

            if self.centered {
                let grad_avg_key = format!("{}.grad_avg", param_state.name);
                if let Some(grad_avg) = state_dict.get(&grad_avg_key) {
                    param_state.init_state("grad_avg".to_string(), grad_avg.clone());
                }
            }

            if self.momentum > 0.0 {
                let momentum_key = format!("{}.momentum_buffer", param_state.name);
                if let Some(momentum_buffer) = state_dict.get(&momentum_key) {
                    param_state.init_state("momentum_buffer".to_string(), momentum_buffer.clone());
                }
            }
        }

        Ok(())
    }
}

impl<B, S, T> Default for RMSprop<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + dtype::num_traits::Float,
{
    fn default() -> Self {
        Self::new(0.01, 0.99, 1e-8, 0.0, 0.0, false)
    }
}

// Implement GPU-accelerated optimizer trait (simplified)
impl<B, S, T> GpuAcceleratedOptimizer for RMSprop<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + 'static,
    T: DataType + FloatExt + num_traits::Float,
{
    fn gpu_available(&self) -> bool {
        self.gpu_enabled
    }

    fn gpu_backend(&self) -> Option<&GpuOptimizerBackend> {
        self.gpu_backend.as_ref()
    }

    fn gpu_config(&self) -> Option<&GpuOptimizerConfig> {
        self.gpu_config.as_ref()
    }

    fn set_gpu_config(&mut self, config: GpuOptimizerConfig) {
        self.gpu_config = Some(config.clone());
        self.gpu_enabled = true;

        // Initialize GPU backend - for now we'll assume this succeeds
        // In production, this should be handled with proper async initialization
        futures::executor::block_on(async {
            match GpuOptimizerBackend::new().await {
                Ok(backend) => {
                    self.gpu_backend = Some(backend);
                }
                Err(e) => {
                    eprintln!("Failed to initialize GPU backend: {:?}", e);
                    // Keep GPU disabled if initialization fails
                    self.gpu_enabled = false;
                }
            }
        });
    }
}
