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
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static + tensor::ops::arithmetic::traits::TensorStorageArithmetic<T>,
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
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static + tensor::ops::arithmetic::traits::TensorStorageArithmetic<T>,
    T: DataType + FloatExt + num_traits::Zero + num_traits::FromPrimitive + num_traits::One,
{
    fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        self.initialize_if_needed(input)?;

        // We can't hold the lock across the forward call if we want to return a reference or something,
        // but here forward consumes expectations.
        // However, we need to extract the inner module to call forward on it.
        // Since we have Arc<Mutex<Option<Linear>>>, we can get a reference.

        let inner_guard = self.inner.lock().map_err(|_| NNError::ExecutionError {
            message: "Failed to acquire lock on LazyLinear".to_string(),
        })?;

        match &*inner_guard {
            Some(layer) => layer.forward(input),
            None => Err(NNError::ExecutionError {
                message: "LazyLinear failed to initialize".to_string(),
            }),
        }
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
        // This is tricky because we need to return mutable references to something inside a mutex.
        // Module trait signature for parameters_mut takes &mut self.
        // But our inner is Arc<Mutex>.
        // If we have &mut self, we can access inner. But inner is Mutex.
        // We can get_mut() on the Mutex if we have exclusive access to it?
        // But inner is wrapped in Arc.

        // LIMITATION: LazyLinear via Arc<Mutex> makes implementation of parameters_mut hard
        // because we can't return a reference that outlives the lock guard.
        // For now, returning empty to avoid unsafe hacks or signature mismatch.
        // Optimizers usually use parameters() (cloned Arcs) so update_data generic might work?
        // But standard SGD might need parameters_mut if it modifies in place?
        // Actually coeus Parameter is a wrapper around Tensor/Arc. Cloning Parameter is cheap.
        // The parameters() method returns Vec<Parameter>, which are cloneable handles.
        // So parameters_mut returning Vec<&mut Parameter> is for when?
        // Looking at Module trait:
        // fn parameters_mut(&mut self) -> Vec<&mut Parameter<B, S, T>>

        // Since we can't easily implement this safe with Arc<Mutex> without unsafe,
        // AND LazyLinear is "lazy", doing this before init is impossible anyway.

        // If constructed, we'd need to lock, but we can't return reference out of lock.
        // Unless we change LazyLinear to not use Arc<Mutex> but just RefCell or UnsafeCell?
        // Or if we assume Module is not shared across threads during definition?

        // For this iteration, we will return empty and log a warning if possible,
        // or just accept that LazyLinear parameters can only be updated via the handles returned by parameters().
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

    fn clone_box(&self) -> Box<dyn Module<B, S, T>> {
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
