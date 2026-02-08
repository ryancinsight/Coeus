//! CPU sum reduction primitive
//!
//! Provides optimized sum reduction for CPU execution.

use crate::DataType;

/// Sum reduction primitive for CPU backend
///
/// Computes the sum of all elements in the input slice
///
/// # Arguments
/// * `input` - Input data slice
///
/// # Returns
/// Sum of all elements
pub fn sum_primitive<T: DataType>(input: &[T]) -> T
where
    T: core::ops::Add<Output = T> + Default + Copy,
{
    let mut sum = T::default();
    
    // TODO: Future SIMD optimization point
    for &val in input.iter() {
        sum = sum + val;
    }
    
    sum
}

/// Sum reduction for strided storage
pub fn sum_strided_primitive<T: DataType>(
    input_data: &[T],
    input_shape: &[usize],
    input_strides: &[usize],
    input_offset: usize,
) -> T
where
    T: core::ops::Add<Output = T> + Default + Copy,
{
    let mut sum = T::default();
    let size = input_shape.iter().product();

    for i in 0..size {
        let idx = input_offset + storage::iter::compute_strided_index_fast(i, input_shape, input_strides);
        sum = sum + input_data[idx];
    }
    
    sum
}

/// Sum reduction along axis primitive (placeholder)
///
/// Future implementation will provide sum reduction along specified axes
///
/// # Arguments
/// * `input` - Input data slice
/// * `input_shape` - Shape of input tensor
/// * `axis` - Axis to reduce along
/// * `result` - Output slice for reduced tensor
///
/// # Returns
/// Result indicating success or failure
pub fn sum_axis_primitive<T: DataType>(
    _input: &[T],
    _input_shape: &[usize],
    _axis: usize,
    _result: &mut [T],
) -> crate::Result<()>
where
    T: Copy + Default,
{
    // TODO: Implement axis-specific sum reduction
    Err(crate::BackendError::UnsupportedOperation {
        operation: "sum_axis".to_string(),
        backend: "cpu".to_string(),
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
    fn test_sum_primitive_empty() {
        let input: [Float32; 0] = [];
        let result = sum_primitive(&input);
        assert_eq!(result, Float32::new(0.0));
    }
}