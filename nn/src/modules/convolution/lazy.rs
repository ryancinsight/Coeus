//! Lazy Convolution modules.
//!
//! These modules initialize their inner convolution layers on the first forward pass,
//! inferring the number of input channels from the input tensor.

use std::sync::{Arc, Mutex};

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use num_traits;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;
use crate::modules::convolution::{Conv1D, Conv2D, Conv3D};

/// A 1D convolution layer that initializes its parameters on the first forward pass.
#[derive(Debug, Clone)]
pub struct LazyConv1d<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub use_bias: bool,
    pub inner: Arc<Mutex<Option<Conv1D<B, S, T>>>>,
}

impl<B, S, T> LazyConv1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    pub fn new(
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Self {
        Self {
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias: bias,
            inner: Arc::new(Mutex::new(None)),
        }
    }

    fn initialize_if_needed(&self, input: &Tensor<B, S, T>) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| NNError::ExecutionError {
            message: "Failed to acquire lock on LazyConv1d".to_string(),
        })?;

        if inner.is_none() {
            // Input shape for Conv1d: (N, C_in, L)
            let shape = input.shape();
            let dims = shape.dims();
            if dims.len() != 3 {
                return Err(NNError::InvalidInput {
                    message: format!("Expected 3D input (N, C, L), got {:?}", dims),
                });
            }
            let in_channels = dims[1];

            // Note: Conv1D in coeus currently does not support dilation or groups.
            // Arguments are stored but ignored here.

            let layer = Conv1D::new(
                in_channels,
                self.out_channels,
                self.kernel_size,
                Some(self.stride),
                Some(self.padding),
                Some(self.use_bias),
            )?;

            *inner = Some(layer);
        }
        Ok(())
    }
}

impl<B, S, T> Module<B, S, T> for LazyConv1d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.initialize_if_needed(input)?;
        let guard = self.inner.lock().unwrap();
        guard.as_ref().unwrap().forward(input)
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
        vec![] // Same limitation as LazyLinear
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
        "LazyConv1d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}

/// A 2D convolution layer that initializes its parameters on the first forward pass.
#[derive(Debug, Clone)]
pub struct LazyConv2d<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    pub out_channels: usize,
    pub kernel_size: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
    pub use_bias: bool,
    pub inner: Arc<Mutex<Option<Conv2D<B, S, T>>>>,
}

impl<B, S, T> LazyConv2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    pub fn new(
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        bias: bool,
    ) -> Self {
        Self {
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias: bias,
            inner: Arc::new(Mutex::new(None)),
        }
    }

    fn initialize_if_needed(&self, input: &Tensor<B, S, T>) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| NNError::ExecutionError {
            message: "Failed to acquire lock on LazyConv2d".to_string(),
        })?;

        if inner.is_none() {
            // Input shape for Conv2d: (N, C_in, H, W)
            let shape = input.shape();
            let dims = shape.dims();
            if dims.len() != 4 {
                return Err(NNError::InvalidInput {
                    message: format!("Expected 4D input (N, C, H, W), got {:?}", dims),
                });
            }
            let in_channels = dims[1];

            // Note: Conv2D currently does not support dilation or groups.
            let layer = Conv2D::new(
                in_channels,
                self.out_channels,
                self.kernel_size,
                Some(self.stride),
                Some(self.padding),
                Some(self.use_bias),
            )?;

            *inner = Some(layer);
        }
        Ok(())
    }
}

impl<B, S, T> Module<B, S, T> for LazyConv2d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.initialize_if_needed(input)?;
        let guard = self.inner.lock().unwrap();
        guard.as_ref().unwrap().forward(input)
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
        "LazyConv2d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}

/// A 3D convolution layer that initializes its parameters on the first forward pass.
#[derive(Debug, Clone)]
pub struct LazyConv3d<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt,
{
    pub out_channels: usize,
    pub kernel_size: (usize, usize, usize),
    pub stride: (usize, usize, usize),
    pub padding: (usize, usize, usize),
    pub dilation: (usize, usize, usize),
    pub groups: usize,
    pub use_bias: bool,
    pub inner: Arc<Mutex<Option<Conv3D<B, S, T>>>>,
}

impl<B, S, T> LazyConv3d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    pub fn new(
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
        bias: bool,
    ) -> Self {
        Self {
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            use_bias: bias,
            inner: Arc::new(Mutex::new(None)),
        }
    }

    fn initialize_if_needed(&self, input: &Tensor<B, S, T>) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| NNError::ExecutionError {
            message: "Failed to acquire lock on LazyConv3d".to_string(),
        })?;

        if inner.is_none() {
            // Input shape for Conv3d: (N, C_in, D, H, W)
            let shape = input.shape();
            let dims = shape.dims();
            if dims.len() != 5 {
                return Err(NNError::InvalidInput {
                    message: format!("Expected 5D input (N, C, D, H, W), got {:?}", dims),
                });
            }
            let in_channels = dims[1];

            // Note: Conv3D currently does not support dilation or groups.
            let layer = Conv3D::new(
                in_channels,
                self.out_channels,
                self.kernel_size,
                Some(self.stride),
                Some(self.padding),
                Some(self.use_bias),
            )?;

            *inner = Some(layer);
        }
        Ok(())
    }
}

impl<B, S, T> Module<B, S, T> for LazyConv3d<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.initialize_if_needed(input)?;
        let guard = self.inner.lock().unwrap();
        guard.as_ref().unwrap().forward(input)
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
        "LazyConv3d"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
        Box::new(self.clone())
    }
}
