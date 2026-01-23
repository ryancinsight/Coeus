//! NPU transpose operations
//!
//! This module provides NPU-optimized matrix transpose primitives.

use dtype::DataType;

/// Matrix transpose primitive for NPU
///
/// Performs matrix transpose: B = A^T
/// where A is m×n and B is n×m
///
/// # Arguments
/// * `input` - Input matrix data (row-major, m×n)
/// * `result` - Output matrix data (row-major, n×m)
/// * `m` - Number of rows in input
/// * `n` - Number of columns in input
///
/// # Returns
/// Result indicating success or failure
pub fn transpose_primitive<T: DataType>(
    input: &[T],
    result: &mut [T],
    m: usize,
    n: usize,
) -> crate::Result<()>
where
    T: Copy,
{
    if input.len() != m * n {
        return Err(crate::BackendError::InvalidInput(format!(
            "Input matrix size mismatch: expected {}, got {}",
            m * n,
            input.len()
        )));
    }
    if result.len() != m * n {
        return Err(crate::BackendError::InvalidInput(format!(
            "Result matrix size mismatch: expected {}, got {}",
            m * n,
            result.len()
        )));
    }

    // TODO: Replace with actual NPU transpose implementation
    for i in 0..m {
        for j in 0..n {
            result[j * m + i] = input[i * n + j];
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_transpose_primitive() {
        // 2×3 matrix
        let input = [
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ];
        let mut result = [Float32::new(0.0); 6];

        transpose_primitive(&input, &mut result, 2, 3).unwrap();

        // Expected: 3×2 matrix [[1, 4], [2, 5], [3, 6]]
        let expected = [
            Float32::new(1.0),
            Float32::new(4.0),
            Float32::new(2.0),
            Float32::new(5.0),
            Float32::new(3.0),
            Float32::new(6.0),
        ];
        assert_eq!(result, expected);
    }
}
