//! TPU max reduction operations
//!
//! This module provides TPU-optimized max reduction primitives.

use dtype::DataType;

/// Max reduction primitive for TPU
///
/// Computes the maximum of all elements in the input
///
/// # Arguments
/// * `input` - Input data slice
///
/// # Returns
/// Maximum element
pub fn max_primitive<T: DataType>(input: &[T]) -> crate::Result<T>
where
    T: PartialOrd + Copy,
{
    if input.is_empty() {
        return Err(crate::BackendError::InvalidInput(
            "Cannot compute max of empty slice".to_string(),
        ));
    }

    // TODO: Replace with actual TPU reduction unit
    let mut max_val = input[0];
    for &val in &input[1..] {
        if val > max_val {
            max_val = val;
        }
    }

    Ok(max_val)
}

/// Max reduction along axis primitive for TPU
///
/// Computes the maximum along a specific axis
///
/// # Arguments
/// * `input` - Input data slice
/// * `result` - Output slice to write results
/// * `shape` - Input tensor shape
/// * `axis` - Axis to reduce along
///
/// # Returns
/// Result indicating success or failure
pub fn max_axis_primitive<T: DataType>(
    _input: &[T],
    _result: &mut [T],
    _shape: &[usize],
    _axis: usize,
) -> crate::Result<()>
where
    T: Copy,
{
    // TODO: Implement axis-specific max reduction for TPU
    Err(crate::BackendError::UnsupportedOperation {
        operation: "max_axis".to_string(),
        backend: "TPU".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_max_primitive() {
        let input = [
            Float32::new(1.0),
            Float32::new(4.0),
            Float32::new(2.0),
            Float32::new(3.0),
        ];

        let result = max_primitive(&input).unwrap();

        assert_eq!(result, Float32::new(4.0));
    }

    #[test]
    fn test_max_empty_input() {
        let input: [Float32; 0] = [];

        let result = max_primitive(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_max_axis_not_implemented() {
        let input = [Float32::new(1.0); 4];
        let mut result = [Float32::new(0.0); 2];
        let shape = [2, 2];

        let res = max_axis_primitive(&input, &mut result, &shape, 0);
        assert!(res.is_err());
    }
}
