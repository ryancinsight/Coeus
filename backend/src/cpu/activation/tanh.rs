//! CPU tanh activation primitive
//!
//! Provides optimized tanh activation for CPU execution.

use crate::DataType;

/// Tanh activation primitive for CPU backend
///
/// Performs element-wise tanh: result[i] = tanh(input[i])
///
/// # Arguments
/// * `input` - Input data slice
/// * `result` - Output slice to write results
///
/// # Returns
/// Result indicating success or failure
pub fn tanh_primitive<T: DataType>(
    input: &[T],
    result: &mut [T],
) -> crate::Result<()>
where
    T: Copy,
{
    if input.len() != result.len() {
        return Err(crate::BackendError::InvalidInput(
            "Input and result slice lengths must match".to_string(),
        ));
    }

    // TODO: Future SIMD optimization point
    for (inp, res) in input.iter().zip(result.iter_mut()) {
        // Convert to f64 for computation, then back
        let x_f64 = inp.to_f64().unwrap_or(0.0);
        let tanh_result = x_f64.tanh();
        *res = T::from(tanh_result).unwrap_or(*inp);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_tanh_primitive() {
        let input = [
            Float32::new(-2.0),
            Float32::new(0.0),
            Float32::new(2.0),
        ];
        let mut result = [Float32::new(0.0); 3];

        tanh_primitive(&input, &mut result).unwrap();

        // Check that results are in valid tanh range [-1, 1]
        for &val in &result {
            assert!(val.get() >= -1.0 && val.get() <= 1.0);
        }

        // Check specific values (approximately)
        assert!((result[0].get() - (-0.964)).abs() < 0.01); // tanh(-2) ≈ -0.964
        assert!((result[1].get() - 0.0).abs() < 0.01);      // tanh(0) = 0.0
        assert!((result[2].get() - 0.964).abs() < 0.01);    // tanh(2) ≈ 0.964
    }
}