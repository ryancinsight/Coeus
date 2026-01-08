//! Core batch normalization structures and traits.
//!
//! This module contains the fundamental BatchNorm structures and common functionality
//! shared across different dimensionalities (1D, 2D, 3D).

use std::cell::RefCell;
use std::marker::PhantomData;

use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::parameter::Parameter;

/// Common trait for batch normalization operations
pub trait BatchNormOps<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    /// Update running statistics during training
    fn update_running_stats(&self, batch_mean: &Tensor<B, S, T>, batch_var: &Tensor<B, S, T>) -> crate::core::error::Result<()>;
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
    T: DataType,
{
    /// Create a new BatchNorm base with the given parameters
    pub fn new(
        backend: B,
        num_features: usize,
        eps: f64,
        momentum: f64,
        track_running_stats: bool,
    ) -> crate::core::error::Result<Self> {
        // Initialize weight (γ) to ones
        let weight_data = vec![T::from(1.0).unwrap(); num_features];
        let weight_tensor = Tensor::<B, S, T>::from_vec(weight_data, &[num_features])?;
        let weight = Parameter::new(weight_tensor, "weight".to_string());

        // Initialize bias (β) to zeros
        let bias_data = vec![T::from(0.0).unwrap(); num_features];
        let bias_tensor = Tensor::<B, S, T>::from_vec(bias_data, &[num_features])?;
        let bias = Parameter::new(bias_tensor, "bias".to_string());

        // Initialize running statistics
        let running_mean_data = vec![T::from(0.0).unwrap(); num_features];
        let running_mean = Tensor::<B, S, T>::from_vec(running_mean_data, &[num_features])?;

        let running_var_data = vec![T::from(1.0).unwrap(); num_features];
        let running_var = Tensor::<B, S, T>::from_vec(running_var_data, &[num_features])?;

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
