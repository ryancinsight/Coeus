//! TPU tanh activation operations
//!
//! This module provides TPU-optimized tanh activation primitives.

use dtype::DataType;

/// Tanh activation primitive for TPU
///
/// Performs element-wise tanh: result[i] = tanh(input[i])
///
/// # Arguments
/// * `input` - Input data slice
/// * `result` - Output slice to write results
///
/// # Returns
/// Result indicating success or failure
pub fn tanh_primitive<T: DataType>(_input: &[T], _result: &mut [T]) -> crate::Result<()>
where
    T: Copy,
{
    // TODO: Implement tanh for TPU
    // Requires floating-point exponential operations
    Err(crate::BackendError::UnsupportedOperation {
        operation: "tanh".to_string(),
        backend: "TPU".to_string(),
    })
}

/// Tanh gradient primitive for TPU
///
/// Computes gradient of tanh: result[i] = grad_output[i] * (1 - tanh(input[i])^2)
///
/// # Arguments
/// * `input` - Original input data slice
/// * `grad_output` - Gradient from next layer
/// * `result` - Output gradient slice
///
/// # Returns
/// Result indicating success or failure
pub fn tanh_grad_primitive<T: DataType>(
    _input: &[T],
    _grad_output: &[T],
    _result: &mut [T],
) -> crate::Result<()>
where
    T: Copy,
{
    // TODO: Implement tanh gradient for TPU
    Err(crate::BackendError::UnsupportedOperation {
        operation: "tanh_grad".to_string(),
        backend: "TPU".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_tanh_not_implemented() {
        let input = [Float32::new(0.0); 4];
        let mut result = [Float32::new(0.0); 4];

        let res = tanh_primitive(&input, &mut result);
        assert!(res.is_err());
    }
}
