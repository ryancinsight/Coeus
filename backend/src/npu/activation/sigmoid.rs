//! NPU sigmoid activation operations
//!
//! This module provides NPU-optimized sigmoid activation primitives.

use dtype::DataType;

/// Sigmoid activation primitive for NPU
///
/// Performs element-wise sigmoid: result[i] = 1 / (1 + exp(-input[i]))
///
/// # Arguments
/// * `input` - Input data slice
/// * `result` - Output slice to write results
///
/// # Returns
/// Result indicating success or failure
pub fn sigmoid_primitive<T: DataType>(_input: &[T], _result: &mut [T]) -> crate::Result<()>
where
    T: Copy,
{
    // TODO: Implement sigmoid for NPU
    // Requires floating-point exponential operations
    Err(crate::BackendError::UnsupportedOperation {
        operation: "sigmoid".to_string(),
        backend: "NPU".to_string(),
    })
}

/// Sigmoid gradient primitive for NPU
///
/// Computes gradient of sigmoid: result[i] = grad_output[i] * sigmoid(input[i]) * (1 - sigmoid(input[i]))
///
/// # Arguments
/// * `input` - Original input data slice
/// * `grad_output` - Gradient from next layer
/// * `result` - Output gradient slice
///
/// # Returns
/// Result indicating success or failure
pub fn sigmoid_grad_primitive<T: DataType>(
    _input: &[T],
    _grad_output: &[T],
    _result: &mut [T],
) -> crate::Result<()>
where
    T: Copy,
{
    // TODO: Implement sigmoid gradient for NPU
    Err(crate::BackendError::UnsupportedOperation {
        operation: "sigmoid_grad".to_string(),
        backend: "NPU".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_sigmoid_not_implemented() {
        let input = [Float32::new(0.0); 4];
        let mut result = [Float32::new(0.0); 4];

        let res = sigmoid_primitive(&input, &mut result);
        assert!(res.is_err());
    }
}
