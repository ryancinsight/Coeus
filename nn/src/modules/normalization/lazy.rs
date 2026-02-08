//! Lazy Batch Normalization modules.

use std::sync::{Arc, Mutex};

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use num_traits;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::{Module, Parameter};
use crate::modules::normalization::batch::{BatchNorm1d, BatchNorm2d, BatchNorm3d};

/// A 1D batch normalization layer that initializes its parameters on the first forward pass.
#[derive(Debug, Clone)]
pub struct LazyBatchNorm1d<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    pub eps: f64,
    pub momentum: Option<f64>,
    pub affine: bool,
    pub track_running_stats: bool,
    pub inner: Arc<Mutex<Option<BatchNorm1d<B, S, T>>>>,
}

impl<B, S, T> LazyBatchNorm1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    pub fn new(eps: f64, momentum: Option<f64>, affine: bool, track_running_stats: bool) -> Self {
        Self {
            eps,
            momentum,
            affine,
            track_running_stats,
            inner: Arc::new(Mutex::new(None)),
        }
    }

    fn initialize_if_needed(&self, input: &Tensor<B, S, T>) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| NNError::ExecutionError {
            message: "Failed to acquire lock on LazyBatchNorm1d".to_string(),
        })?;

        if inner.is_none() {
            let shape = input.shape();
            let dims = shape.dims();
            if dims.len() < 2 {
                return Err(NNError::InvalidInput {
                    message: format!("Expected at least 2D input, got {:?}", dims),
                });
            }
            let num_features = dims[1];

            // TODO: Support affine parameter when underlying BatchNorm supports it.
            // Currently ignoring self.affine.

            let layer = BatchNorm1d::new_with_backend(
                B::default(),
                num_features,
                self.eps,
                self.momentum.unwrap_or(0.1),
                self.track_running_stats,
            )?;

            *inner = Some(layer);
        }
        Ok(())
    }
}

impl<B, S, T> Module<B, S, T> for LazyBatchNorm1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static + tensor::ops::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.initialize_if_needed(input)?;
        let mut guard = self.inner.lock().map_err(|_| NNError::ExecutionError {
            message: "Failed to acquire lock on LazyBatchNorm1d".to_string(),
        })?;
        guard.as_mut().unwrap().forward(input)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        if let Ok(guard) = self.inner.lock() {
            if let Some(layer) = guard.as_ref() {
                return layer.parameters();
            }
        }
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<B, S, T>> {
        vec![]
    }

    fn zero_grad(&mut self) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(layer) = guard.as_mut() {
                layer.zero_grad();
            }
        }
    }

    fn train(&mut self, mode: bool) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(layer) = guard.as_mut() {
                layer.train(mode);
            }
        }
    }

    fn name(&self) -> &str {
        "LazyBatchNorm1d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}

/// A 2D batch normalization layer that initializes its parameters on the first forward pass.
#[derive(Debug, Clone)]
pub struct LazyBatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    pub eps: f64,
    pub momentum: Option<f64>,
    pub affine: bool,
    pub track_running_stats: bool,
    pub inner: Arc<Mutex<Option<BatchNorm2d<B, S, T>>>>,
}

impl<B, S, T> LazyBatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    pub fn new(eps: f64, momentum: Option<f64>, affine: bool, track_running_stats: bool) -> Self {
        Self {
            eps,
            momentum,
            affine,
            track_running_stats,
            inner: Arc::new(Mutex::new(None)),
        }
    }

    fn initialize_if_needed(&self, input: &Tensor<B, S, T>) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| NNError::ExecutionError {
            message: "Failed to acquire lock on LazyBatchNorm2d".to_string(),
        })?;

        if inner.is_none() {
            let shape = input.shape();
            let dims = shape.dims();
            if dims.len() < 3 {
                // Expecting (N, C, H, W)
                return Err(NNError::InvalidInput {
                    message: format!("Expected 4D input, got {:?}", dims),
                });
            }
            let num_features = dims[1];

            let layer = BatchNorm2d::new_with_backend(
                B::default(),
                num_features,
                self.eps,
                self.momentum.unwrap_or(0.1),
                self.track_running_stats,
            )?;
            *inner = Some(layer);
        }
        Ok(())
    }
}

impl<B, S, T> Module<B, S, T> for LazyBatchNorm2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static + tensor::ops::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.initialize_if_needed(input)?;
        let mut guard = self.inner.lock().map_err(|_| NNError::ExecutionError {
            message: "Failed to acquire lock on LazyBatchNorm2d".to_string(),
        })?;
        guard.as_mut().unwrap().forward(input)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        if let Ok(guard) = self.inner.lock() {
            if let Some(layer) = guard.as_ref() {
                return layer.parameters();
            }
        }
        vec![]
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<B, S, T>> {
        vec![]
    }
    fn zero_grad(&mut self) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(layer) = guard.as_mut() {
                layer.zero_grad();
            }
        }
    }
    fn train(&mut self, mode: bool) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(layer) = guard.as_mut() {
                layer.train(mode);
            }
        }
    }
    fn name(&self) -> &str {
        "LazyBatchNorm2d"
    }
    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}

/// A 3D batch normalization layer that initializes its parameters on the first forward pass.
#[derive(Debug, Clone)]
pub struct LazyBatchNorm3d<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    pub eps: f64,
    pub momentum: Option<f64>,
    pub affine: bool,
    pub track_running_stats: bool,
    pub inner: Arc<Mutex<Option<BatchNorm3d<B, S, T>>>>,
}

impl<B, S, T> LazyBatchNorm3d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    pub fn new(eps: f64, momentum: Option<f64>, affine: bool, track_running_stats: bool) -> Self {
        Self {
            eps,
            momentum,
            affine,
            track_running_stats,
            inner: Arc::new(Mutex::new(None)),
        }
    }

    fn initialize_if_needed(&self, input: &Tensor<B, S, T>) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| NNError::ExecutionError {
            message: "Failed to acquire lock on LazyBatchNorm3d".to_string(),
        })?;

        if inner.is_none() {
            let shape = input.shape();
            let dims = shape.dims();
            if dims.len() < 4 {
                // Expecting (N, C, D, H, W)
                return Err(NNError::InvalidInput {
                    message: format!("Expected 5D input, got {:?}", dims),
                });
            }
            let num_features = dims[1];

            let layer = BatchNorm3d::new_with_backend(
                B::default(),
                num_features,
                self.eps,
                self.momentum.unwrap_or(0.1),
                self.track_running_stats,
            )?;
            *inner = Some(layer);
        }
        Ok(())
    }
}

impl<B, S, T> Module<B, S, T> for LazyBatchNorm3d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static + tensor::ops::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.initialize_if_needed(input)?;
        let mut guard = self.inner.lock().map_err(|_| NNError::ExecutionError {
            message: "Failed to acquire lock on LazyBatchNorm3d".to_string(),
        })?;
        guard.as_mut().unwrap().forward(input)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        if let Ok(guard) = self.inner.lock() {
            if let Some(layer) = guard.as_ref() {
                return layer.parameters();
            }
        }
        vec![]
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<B, S, T>> {
        vec![]
    }
    fn zero_grad(&mut self) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(layer) = guard.as_mut() {
                layer.zero_grad();
            }
        }
    }
    fn train(&mut self, mode: bool) {
        if let Ok(mut guard) = self.inner.lock() {
            if let Some(layer) = guard.as_mut() {
                layer.train(mode);
            }
        }
    }
    fn name(&self) -> &str {
        "LazyBatchNorm3d"
    }
    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}
