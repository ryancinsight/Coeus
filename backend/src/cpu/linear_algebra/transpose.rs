//! CPU matrix transpose primitive
//!
//! Provides optimized matrix transpose for CPU execution.

use crate::DataType;

/// Matrix transpose primitive for CPU backend
///
/// Performs matrix transpose: B = A^T where A is m×n, B is n×m
///
/// # Arguments
/// * `input` - Input matrix data (row-major, m×n)
/// * `result` - Output matrix data (row-major, n×m)
/// * `m` - Number of rows in input matrix
/// * `n` - Number of columns in input matrix
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
    // Validate input dimensions
    if input.len() != m * n {
        return Err(crate::BackendError::InvalidInput(format!(
            "Input matrix size {} does not match expected {}×{} = {}",
            input.len(),
            m,
            n,
            m * n
        )));
    }
    if result.len() != n * m {
        return Err(crate::BackendError::InvalidInput(format!(
            "Result matrix size {} does not match expected {}×{} = {}",
            result.len(),
            n,
            m,
            n * m
        )));
    }

    // Perform transpose: result[j][i] = input[i][j]
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
        // Test 2×3 matrix transpose
        let input = [
            Float32::new(1.0), Float32::new(2.0), Float32::new(3.0),  // [1, 2, 3]
            Float32::new(4.0), Float32::new(5.0), Float32::new(6.0),  // [4, 5, 6]
        ];
        let mut result = [Float32::new(0.0); 6];

        transpose_primitive(&input, &mut result, 2, 3).unwrap();

        // Expected: [1, 4]
        //           [2, 5]
        //           [3, 6]
        let expected = [
            Float32::new(1.0), Float32::new(4.0),
            Float32::new(2.0), Float32::new(5.0),
            Float32::new(3.0), Float32::new(6.0),
        ];
        assert_eq!(result, expected);
    }
}