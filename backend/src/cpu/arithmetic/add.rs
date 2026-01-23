//! CPU element-wise addition primitive
//!
//! Provides SIMD-ready element-wise addition for CPU execution.
//! This is the foundation primitive that storage operations delegate to.

use crate::DataType;

/// Element-wise addition primitive for CPU backend
///
/// Performs element-wise addition: result[i] = lhs[i] + rhs[i]
/// 
/// This is the foundation primitive that higher-level operations delegate to.
/// Future SIMD optimizations will be implemented here.
///
/// # Arguments
/// * `lhs` - Left-hand side data slice
/// * `rhs` - Right-hand side data slice
/// * `result` - Output slice to write results
///
/// # Returns
/// Result indicating success or failure
///
/// # Errors
/// Returns error if slice lengths don't match
///
/// # Examples
/// ```
/// use backend::cpu::arithmetic::add_primitive;
/// use dtype::float::Float32;
///
/// let lhs = [Float32::new(1.0), Float32::new(2.0)];
/// let rhs = [Float32::new(3.0), Float32::new(4.0)];
/// let mut result = [Float32::new(0.0); 2];
/// 
/// add_primitive(&lhs, &rhs, &mut result).unwrap();
/// // result now contains [4.0, 6.0]
/// ```
pub fn add_primitive<T: DataType>(
    lhs: &[T],
    rhs: &[T],
    result: &mut [T],
) -> crate::Result<()>
where
    T: core::ops::Add<Output = T> + Copy,
{
    if lhs.len() != rhs.len() || lhs.len() != result.len() {
        return Err(crate::BackendError::InvalidInput(
            "Slice lengths must match for element-wise addition".to_string(),
        ));
    }

    // TODO: Future SIMD optimization point
    // For now, use simple loop - will be replaced with SIMD intrinsics
    for ((l, r), res) in lhs.iter().zip(rhs.iter()).zip(result.iter_mut()) {
        *res = *l + *r;
    }

    Ok(())
}

/// Element-wise addition with scalar primitive
///
/// Performs element-wise addition with scalar: result[i] = input[i] + scalar
///
/// # Arguments
/// * `input` - Input data slice
/// * `scalar` - Scalar value to add
/// * `result` - Output slice to write results
///
/// # Returns
/// Result indicating success or failure
pub fn add_scalar_primitive<T: DataType>(
    input: &[T],
    scalar: T,
    result: &mut [T],
) -> crate::Result<()>
where
    T: core::ops::Add<Output = T> + Copy,
{
    if input.len() != result.len() {
        return Err(crate::BackendError::InvalidInput(
            "Input and result slice lengths must match".to_string(),
        ));
    }

    for (inp, res) in input.iter().zip(result.iter_mut()) {
        *res = *inp + scalar;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_add_primitive() {
        let lhs = [Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let rhs = [Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)];
        let mut result = [Float32::new(0.0); 3];

        add_primitive(&lhs, &rhs, &mut result).unwrap();

        let expected = [Float32::new(5.0), Float32::new(7.0), Float32::new(9.0)];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_add_primitive_length_mismatch() {
        let lhs = [Float32::new(1.0)];
        let rhs = [Float32::new(1.0), Float32::new(2.0)];
        let mut result = [Float32::new(0.0); 2];

        let result_op = add_primitive(&lhs, &rhs, &mut result);
        assert!(result_op.is_err());
    }

    #[test]
    fn test_add_scalar_primitive() {
        let input = [Float32::new(1.0), Float32::new(2.0)];
        let scalar = Float32::new(10.0);
        let mut result = [Float32::new(0.0); 2];

        add_scalar_primitive(&input, scalar, &mut result).unwrap();

        let expected = [Float32::new(11.0), Float32::new(12.0)];
        assert_eq!(result, expected);
    }
}