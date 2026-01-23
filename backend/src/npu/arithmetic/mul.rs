//! NPU multiplication operations
//!
//! This module provides NPU-optimized element-wise multiplication primitives.

use dtype::DataType;

/// Element-wise multiplication primitive for NPU
///
/// Performs element-wise multiplication: result[i] = lhs[i] * rhs[i]
///
/// # Arguments
/// * `lhs` - Left-hand side data slice
/// * `rhs` - Right-hand side data slice
/// * `result` - Output slice to write results
///
/// # Returns
/// Result indicating success or failure
pub fn mul_primitive<T: DataType>(
    lhs: &[T],
    rhs: &[T],
    result: &mut [T],
) -> crate::Result<()>
where
    T: core::ops::Mul<Output = T> + Copy,
{
    if lhs.len() != rhs.len() || lhs.len() != result.len() {
        return Err(crate::BackendError::InvalidInput(
            "Slice lengths must match for element-wise multiplication".to_string(),
        ));
    }

    // TODO: Replace with actual NPU kernel implementation
    for ((l, r), res) in lhs.iter().zip(rhs.iter()).zip(result.iter_mut()) {
        *res = *l * *r;
    }

    Ok(())
}

/// Element-wise multiplication with scalar primitive for NPU
///
/// Performs element-wise multiplication with scalar: result[i] = input[i] * scalar
///
/// # Arguments
/// * `input` - Input data slice
/// * `scalar` - Scalar value to multiply
/// * `result` - Output slice to write results
///
/// # Returns
/// Result indicating success or failure
pub fn mul_scalar_primitive<T: DataType>(
    input: &[T],
    scalar: T,
    result: &mut [T],
) -> crate::Result<()>
where
    T: core::ops::Mul<Output = T> + Copy,
{
    if input.len() != result.len() {
        return Err(crate::BackendError::InvalidInput(
            "Input and result slice lengths must match".to_string(),
        ));
    }

    // TODO: Replace with actual NPU kernel implementation
    for (inp, res) in input.iter().zip(result.iter_mut()) {
        *res = *inp * scalar;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_mul_primitive() {
        let lhs = [Float32::new(2.0), Float32::new(3.0)];
        let rhs = [Float32::new(4.0), Float32::new(5.0)];
        let mut result = [Float32::new(0.0); 2];

        mul_primitive(&lhs, &rhs, &mut result).unwrap();

        let expected = [Float32::new(8.0), Float32::new(15.0)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mul_scalar_primitive() {
        let input = [Float32::new(2.0), Float32::new(3.0)];
        let scalar = Float32::new(10.0);
        let mut result = [Float32::new(0.0); 2];

        mul_scalar_primitive(&input, scalar, &mut result).unwrap();

        let expected = [Float32::new(20.0), Float32::new(30.0)];
        assert_eq!(result, expected);
    }
}
