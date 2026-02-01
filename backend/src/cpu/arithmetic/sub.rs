//! CPU element-wise subtraction primitive
//!
//! Provides SIMD-ready element-wise subtraction for CPU execution.

use crate::DataType;
use crate::binary_strided_primitive;

/// Element-wise subtraction primitive for CPU backend
///
/// Performs element-wise subtraction: result[i] = lhs[i] - rhs[i]
///
/// # Arguments
/// * `lhs` - Left-hand side data slice
/// * `rhs` - Right-hand side data slice  
/// * `result` - Output slice to write results
///
/// # Returns
/// Result indicating success or failure
pub fn sub_primitive<T: DataType>(
    lhs: &[T],
    rhs: &[T],
    result: &mut [T],
) -> crate::Result<()>
where
    T: core::ops::Sub<Output = T> + Copy,
{
    if lhs.len() != rhs.len() || lhs.len() != result.len() {
        return Err(crate::BackendError::InvalidInput(
            "Slice lengths must match for element-wise subtraction".to_string(),
        ));
    }

    // TODO: Future SIMD optimization point
    for ((l, r), res) in lhs.iter().zip(rhs.iter()).zip(result.iter_mut()) {
        *res = *l - *r;
    }

    Ok(())
}

// Re-implement sub_strided_primitive using macro
binary_strided_primitive!(sub_strided_primitive, Sub, sub);

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_sub_primitive() {
        let lhs = [Float32::new(5.0), Float32::new(7.0), Float32::new(9.0)];
        let rhs = [Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let mut result = [Float32::new(0.0); 3];

        sub_primitive(&lhs, &rhs, &mut result).unwrap();

        let expected = [Float32::new(4.0), Float32::new(5.0), Float32::new(6.0)];
        assert_eq!(result, expected);
    }
}