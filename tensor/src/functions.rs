//! Function objects for automatic differentiation
//!
//! This module defines the backward functions that implement gradient computation
//! for various tensor operations in the computation graph.

extern crate alloc;

use alloc::vec::Vec;
use alloc::sync::Arc;
use crate::{Tensor, DifferentiableFunction, Function};
use backend::Backend;
use dtype::DataType;
use storage::{Storage, StorageFromVec, DenseStorage};
use anyhow;

/// Add function for element-wise addition
#[derive(Debug)]
pub struct AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Input tensors: [lhs, rhs]
    pub inputs: Vec<Arc<Tensor<B, S, T>>>,
}

impl<B, S, T> crate::AsAny for AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}


impl<B, S, T> AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    /// Create a new AddFunction with the given inputs
    #[must_use]
    pub fn new(inputs: Vec<Arc<Tensor<B, S, T>>>) -> Self {
        Self { inputs }
    }
}

impl<B, S, T> DifferentiableFunction<B, S, T> for AddFunction<B, S, T>
where
    B: Backend<Data = T>,
    S: Storage<T> + StorageFromVec<T>,
    T: DataType,
{
    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

impl<B, S, T> Function<B, S, T> for AddFunction<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + StorageFromVec<T> + Clone + 'static,
    T: DataType + Clone + Copy,
{
    fn inputs(&self) -> &[Arc<Tensor<B, S, T>>] {
        &self.inputs
    }

    fn backward(&self, grad_output: &Tensor<B, DenseStorage<T>, T>) -> std::result::Result<Vec<Tensor<B, S, T>>, anyhow::Error> {
        // For addition, gradient w.r.t. both inputs is the same as grad_output
        // Need to broadcast grad_output to match input shapes if necessary
        let mut result = Vec::with_capacity(self.inputs.len());

        for input in &self.inputs {
            // Convert grad_output to the same storage type as the input
            // Get the data as a vec and create a tensor with the appropriate storage
            let data = grad_output.as_slice().to_vec();
            let grad_tensor = Tensor::from_vec(data, grad_output.shape().dims())?;
            result.push(grad_tensor);
        }

        Ok(result)
    }
}
