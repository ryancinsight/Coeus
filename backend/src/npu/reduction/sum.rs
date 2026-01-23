//! NPU sum reduction operations
//!
//! This module provides NPU-optimized sum reduction primitives.

use dtype::DataType;

/// Sum reduction primitive for NPU
///
/// Computes the sum of all elements in the input
///
/// # Arguments
/// * `input` - Input data slice
///
/// # Returns
/// Sum of all elements
pub fn sum_primitive<T: DataType>(input: &[T]) -> T
where
    T: core::ops::Add<Output = T> + Copy + Default,
{
    // TODO: Replace with actual NPU reduction unit
    input.iter().fold(T::default(), |acc, &x| acc + x)
}

/// Sum reduction along axis primitive for NPU
///
/// Computes the sum along a specific axis
///
/// # Arguments
/// * `input` - Input data slice
/// * `result` - Output slice to write results
/// * `shape` - Input tensor shape
/// * `axis` - Axis to reduce along
///
/// # Returns
/// Result indicating success or failure
pub fn sum_axis_primitive<T: DataType>(
    _input: &[T],
    _result: &mut [T],
    _shape: &[usize],
    _axis: usize,
) -> crate::Result<()>
where
    T: Copy,
{
    // TODO: Implement axis-specific sum reduction for NPU
    Err(crate::BackendError::UnsupportedOperation {
        operation: "sum_axis".to_string(),
        backend: "NPU".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_sum_primitive() {
        let input = [
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ];

        let result = sum_primitive(&input);

        assert_eq!(result, Float32::new(10.0));
    }

    #[test]
    fn test_sum_axis_not_implemented() {
        let input = [Float32::new(1.0); 4];
        let mut result = [Float32::new(0.0); 2];
        let shape = [2, 2];

        let res = sum_axis_primitive(&input, &mut result, &shape, 0);
        assert!(res.is_err());
    }
}
