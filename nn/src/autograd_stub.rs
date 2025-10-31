//! Autograd stub module for decoupling NN operations from full autograd crate.
//!
//! This module provides stub implementations of autograd functions that the NN crate
//! uses, allowing the NN crate to be tested independently of the full autograd system.
//! These stubs work with the MinimalTensor implementation instead of requiring the
//! full Tensor API with automatic differentiation traits.

use crate::error::{NNError, Result};
use tensor::Tensor;

/// Mock autograd error that doesn't depend on the full tensor API
#[derive(Debug, thiserror::Error)]
pub enum AutogradError {
    #[error("Autograd operation not supported in testing mode: {operation}")]
    NotSupportedInTesting {
        operation: String,
    },
    #[error("Gradient computation failed: {message}")]
    GradientComputationFailed {
        message: String,
    },
}

impl From<AutogradError> for NNError {
    fn from(err: AutogradError) -> Self {
        NNError::TrainingError {
            message: format!("Autograd error: {}", err),
        }
    }
}

/// Stub implementation of backward function for testing.
/// In a real implementation, this would perform gradient computation.
/// For testing purposes, this is a no-op that succeeds.
pub fn backward<B, S, T>(_tensor: &Tensor<B, S, T>) -> Result<()>
where
    B: backend::Backend,
    S: storage::Storage<T>,
    T: dtype::DataType,
{
    // This is a stub implementation for testing.
    // In the real autograd system, this would traverse the computation graph
    // and compute gradients for all tensors that require them.
    //
    // For NN testing purposes, we just return success since we're not
    // actually performing gradient computation in isolated tests.
    Ok(())
}

/// Additional autograd stub functions can be added here as needed
/// for other NN operations that might depend on autograd in the future.

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;
    use tensor::Tensor;

    #[test]
    fn test_backward_stub() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            CpuBackend::new(),
            vec![Float32::new(1.0), Float32::new(2.0)],
            vec![2],
        )
        .unwrap();

        // Should succeed (no-op in testing mode)
        let result = backward(&tensor);
        assert!(result.is_ok());
    }
}

