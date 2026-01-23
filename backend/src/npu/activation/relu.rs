//! NPU ReLU activation operations
//!
//! This module provides NPU-optimized ReLU activation primitives.

use dtype::DataType;

/// ReLU activation primitive for NPU
///
/// Performs element-wise ReLU: result[i] = max(0, input[i])
///
/// # Arguments
/// * `input` - Input data slice
/// * `result` - Output slice to write results
///
/// # Returns
/// Result indicating success or failure
pub fn relu_primitive<T: DataType>(input: &[T], result: &mut [T]) -> crate::Result<()>
where
    T: PartialOrd + Default + Copy,
{
    if input.len() != result.len() {
        return Err(crate::BackendError::InvalidInput(
            "Input and result slice lengths must match".to_string(),
        ));
    }

    let zero = T::default();
    // TODO: Replace with actual NPU activation unit
    for (inp, res) in input.iter().zip(result.iter_mut()) {
        *res = if *inp > zero { *inp } else { zero };
    }

    Ok(())
}

/// ReLU gradient primitive for NPU
///
/// Computes gradient of ReLU: result[i] = grad_output[i] if input[i] > 0 else 0
///
/// # Arguments
/// * `input` - Original input data slice
/// * `grad_output` - Gradient from next layer
/// * `result` - Output gradient slice
///
/// # Returns
/// Result indicating success or failure
pub fn relu_grad_primitive<T: DataType>(
    input: &[T],
    grad_output: &[T],
    result: &mut [T],
) -> crate::Result<()>
where
    T: PartialOrd + Default + Copy,
{
    if input.len() != grad_output.len() || input.len() != result.len() {
        return Err(crate::BackendError::InvalidInput(
            "All slice lengths must match".to_string(),
        ));
    }

    let zero = T::default();
    // TODO: Replace with actual NPU gradient computation
    for ((inp, grad), res) in input.iter().zip(grad_output.iter()).zip(result.iter_mut()) {
        *res = if *inp > zero { *grad } else { zero };
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_relu_primitive() {
        let input = [
            Float32::new(-2.0),
            Float32::new(-1.0),
            Float32::new(0.0),
            Float32::new(1.0),
            Float32::new(2.0),
        ];
        let mut result = [Float32::new(0.0); 5];

        relu_primitive(&input, &mut result).unwrap();

        let expected = [
            Float32::new(0.0),
            Float32::new(0.0),
            Float32::new(0.0),
            Float32::new(1.0),
            Float32::new(2.0),
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_relu_grad_primitive() {
        let input = [
            Float32::new(-1.0),
            Float32::new(0.0),
            Float32::new(1.0),
            Float32::new(2.0),
        ];
        let grad_output = [
            Float32::new(1.0),
            Float32::new(1.0),
            Float32::new(1.0),
            Float32::new(1.0),
        ];
        let mut result = [Float32::new(0.0); 4];

        relu_grad_primitive(&input, &grad_output, &mut result).unwrap();

        let expected = [
            Float32::new(0.0),
            Float32::new(0.0),
            Float32::new(1.0),
            Float32::new(1.0),
        ];
        assert_eq!(result, expected);
    }
}
