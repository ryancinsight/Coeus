//! Core batch normalization structures and traits.
//!
//! This module contains the fundamental BatchNorm structures and common functionality
//! shared across different dimensionalities (1D, 2D, 3D).

use std::cell::RefCell;
use std::marker::PhantomData;

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::core::parameter::Parameter;

/// Common trait for batch normalization operations
pub trait BatchNormOps<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    /// Update running statistics during training
    fn update_running_stats(&self, batch_mean: &[T], batch_var: &[T]) -> Result<()>;
}

/// Batch Normalization base structure
///
/// This contains the common fields shared by all BatchNorm variants.
#[derive(Debug)]
pub struct BatchNormBase<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    /// Number of features/channels to normalize
    pub num_features: usize,
    /// Scale parameter γ [num_features]
    pub weight: Parameter<B, S, T>,
    /// Shift parameter β [num_features]
    pub bias: Parameter<B, S, T>,
    /// Running mean [num_features] (interior mutability for automatic updates)
    pub running_mean: RefCell<Tensor<B, S, T>>,
    /// Running variance [num_features] (interior mutability for automatic updates)
    pub running_var: RefCell<Tensor<B, S, T>>,
    /// Numerical stability constant ε
    pub eps: f64,
    /// Running statistics momentum
    pub momentum: f64,
    /// Training mode flag
    pub training: bool,
    /// Whether to track running statistics
    pub track_running_stats: bool,
    /// Phantom data for unused generic parameters
    _phantom: PhantomData<B>,
}

impl<B, S, T> BatchNormBase<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    /// Create a new BatchNorm base with the given parameters
    pub fn new(
        backend: B,
        num_features: usize,
        eps: f64,
        momentum: f64,
        track_running_stats: bool,
    ) -> Result<Self> {
        // Initialize weight (γ) to ones
        let weight_data = vec![T::one(); num_features];
        let weight_storage = S::from_vec(weight_data, &[num_features])?;
        let weight_tensor = Tensor::from_storage(weight_storage, backend.clone());
        let weight = Parameter::new(weight_tensor.requires_grad_(true), "weight".to_string());

        // Initialize bias (β) to zeros
        let bias_data = vec![T::zero(); num_features];
        let bias_storage = S::from_vec(bias_data, &[num_features])?;
        let bias_tensor = Tensor::from_storage(bias_storage, backend.clone());
        let bias = Parameter::new(bias_tensor.requires_grad_(true), "bias".to_string());

        // Initialize running statistics
        let running_mean_data = vec![T::zero(); num_features];
        let running_mean_storage = S::from_vec(running_mean_data, &[num_features])?;
        let running_mean = Tensor::from_storage(running_mean_storage, backend.clone());

        let running_var_data = vec![T::one(); num_features];
        let running_var_storage = S::from_vec(running_var_data, &[num_features])?;
        let running_var = Tensor::from_storage(running_var_storage, backend);

        Ok(Self {
            num_features,
            weight,
            bias,
            running_mean: RefCell::new(running_mean),
            running_var: RefCell::new(running_var),
            eps,
            momentum,
            training: true,
            track_running_stats,
            _phantom: PhantomData,
        })
    }

    /// Update running statistics with exponential moving average.
    pub fn update_running_stats(&self, batch_mean: &[T], batch_var: &[T]) -> Result<()> {
        if !self.track_running_stats {
            return Ok(());
        }

        let momentum_t = T::from(self.momentum).ok_or_else(|| NNError::NumericalError {
            message: format!("momentum ({}) not representable", self.momentum),
        })?;
        let one_minus_momentum =
            T::from(1.0 - self.momentum).ok_or_else(|| NNError::NumericalError {
                message: format!("1.0 - momentum ({}) not representable", 1.0 - self.momentum),
            })?;

        // Update running mean
        {
            let running_mean_tensor = self.running_mean.borrow();
            let running_mean_slice = running_mean_tensor.as_slice();
            let mut updated_mean = Vec::with_capacity(self.num_features);
            for i in 0..self.num_features {
                updated_mean
                    .push(momentum_t * running_mean_slice[i] + one_minus_momentum * batch_mean[i]);
            }
            drop(running_mean_tensor);
            *self.running_mean.borrow_mut() = Tensor::from_vec(updated_mean, &[self.num_features])?;
        }

        // Update running variance
        {
            let running_var_tensor = self.running_var.borrow();
            let running_var_slice = running_var_tensor.as_slice();
            let mut updated_var = Vec::with_capacity(self.num_features);
            for i in 0..self.num_features {
                updated_var
                    .push(momentum_t * running_var_slice[i] + one_minus_momentum * batch_var[i]);
            }
            drop(running_var_tensor);
            *self.running_var.borrow_mut() = Tensor::from_vec(updated_var, &[self.num_features])?;
        }

        Ok(())
    }
}

impl<B, S, T> Clone for BatchNormBase<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    fn clone(&self) -> Self {
        Self {
            num_features: self.num_features,
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            running_mean: RefCell::new(self.running_mean.borrow().clone()),
            running_var: RefCell::new(self.running_var.borrow().clone()),
            eps: self.eps,
            momentum: self.momentum,
            training: self.training,
            track_running_stats: self.track_running_stats,
            _phantom: PhantomData,
        }
    }
}
