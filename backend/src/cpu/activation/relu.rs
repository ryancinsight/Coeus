//! CPU ReLU activation primitive
//!
//! Provides optimized ReLU activation for CPU execution.

use crate::DataType;

/// ReLU activation primitive for CPU backend
///
/// Performs element-wise ReLU: result[i] = max(0, input[i])
///
/// # Arguments
/// * `input` - Input data slice
/// * `result` - Output slice to write results
///
/// # Returns
/// Result indicating success or failure
pub fn relu_primitive<T: DataType>(
    input: &[T],
    result: &mut [T],
) -> crate::Result<()>
where
    T: PartialOrd + Default + Copy,
{
    if input.len() != result.len() {
        return Err(crate::BackendError::InvalidInput(
            "Input and result slice lengths must match".to_string(),
        ));
    }

    let zero = T::default();
    
    // TODO: Future SIMD optimization point
    for (inp, res) in input.iter().zip(result.iter_mut()) {
        *res = if *inp > zero { *inp } else { zero };
    }

    Ok(())
}

// Implement relu_strided_primitive using macro
crate::unary_strided_primitive!(relu_strided_primitive, |x_val| {
    let zero = T::default();
    if x_val > zero { x_val } else { zero }
}, std::cmp::PartialOrd);

// Implement relu_csr_primitive using macro
crate::unary_csr_primitive!(relu_csr_primitive, |x_val| {
    let zero = T::default();
    if x_val > zero { x_val } else { zero }
}, std::cmp::PartialOrd);

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
}