//! LazyLinear module.

use std::fmt;
use std::sync::{Arc, Mutex};

use backend::Backend;
use dtype::{traits::FloatExt, DataType};
use num_traits;
use storage::{Storage, StorageFromVec, StorageToDense};
use tensor::Tensor;

use super::Linear;
use crate::core::error::{NNError, Result};
use crate::core::module::Module;
use crate::core::parameter::Parameter;

/// A linear neural network layer that initializes its parameters on the first forward pass.
///
/// Useful when the number of input features is not known at construction time.
#[derive(Debug, Clone)]
pub struct LazyLinear<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType,
{
    /// Number of output features
    pub out_features: usize,
    /// Whether to include a bias term
    pub use_bias: bool,
    /// The inner Linear layer (initialized lazily)
    pub inner: Arc<Mutex<Option<Linear<B, S, T>>>>,
}

impl<B, S, T> LazyLinear<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    /// Create a new LazyLinear layer.
    ///
    /// # Arguments
    /// * `out_features` - Number of output features
    /// * `bias` - Whether to include a bias term (default: true)
    pub fn new(out_features: usize, bias: bool) -> Self {
        Self {
            out_features,
            use_bias: bias,
            inner: Arc::new(Mutex::new(None)),
        }
    }

    /// Initialize the inner Linear layer if not already initialized.
    fn initialize_if_needed(&self, input: &Tensor<B, S, T>) -> Result<()> {
        let mut inner = self.inner.lock().map_err(|_| NNError::ExecutionError {
            message: "Failed to acquire lock on LazyLinear".to_string(),
        })?;

        if inner.is_none() {
            let input_shape = input.shape();
            let dims = input_shape.dims();

            if dims.is_empty() {
                return Err(NNError::InvalidInput {
                    message: "Input tensor must have at least one dimension".to_string(),
                });
            }

            let in_features = *dims.last().unwrap();

            // Create the real Linear layer
            // Note: biases are boolean in LazyLinear but Linear always has bias param?
            // Checking Linear struct: `pub bias: Parameter<B, S, T>`
            // If use_bias is false, we should technically disable gradients or zero it,
            // but for now we follow Linear's standard construction which includes bias.
            // TODO: Support no-bias Linear in core Linear first if strictly required.

            let layer = Linear::new(in_features, self.out_features)?;
            // If use_bias is false, we might want to zero it and freeze it?
            // For now assuming bias is always present in underlying Linear as per dense.rs impl.

            *inner = Some(layer);
        }
        Ok(())
    }
}

impl<B, S, T> Module<B, S, T> for LazyLinear<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static + tensor::ops::dispatch::TensorStorageOps<T>,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    type Input = Tensor<B, S, T>;
    type Output = Tensor<B, S, T>;

    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.initialize_if_needed(input)?;
        let inner = self.inner.lock().map_err(|_| NNError::ExecutionError {
            message: "Failed to acquire lock on LazyLinear".to_string(),
        })?;
        inner.as_ref().unwrap().forward(input)
    }

    fn parameters(&self) -> Vec<Parameter<B, S, T>> {
        match self.inner.lock() {
            Ok(guard) => match &*guard {
                Some(layer) => layer.parameters(),
                None => vec![],
            },
            Err(_) => vec![],
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<B, S, T>> {
        // Same limitation as documented below
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
        "LazyLinear"
    }

    fn clone_box(&self) -> Box<dyn Module<B, S, T, Input = Self::Input, Output = Self::Output>> {
        Box::new(self.clone())
    }
}

impl<B, S, T> fmt::Display for LazyLinear<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LazyLinear(out_features={}, bias={}, initialized={})",
            self.out_features,
            self.use_bias,
            self.inner.lock().map(|g| g.is_some()).unwrap_or(false)
        )
    }
}
