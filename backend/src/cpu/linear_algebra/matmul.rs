//! CPU matrix multiplication primitive
//!
//! Provides optimized matrix multiplication for CPU execution.

use crate::DataType;

/// Matrix multiplication primitive for CPU backend
///
/// Performs matrix multiplication: C = A × B where A is m×k, B is k×n, C is m×n
///
/// # Mathematical Theorem
///
/// For matrices A ∈ ℝ^(m×k) and B ∈ ℝ^(k×n), the matrix product C = AB is defined as:
/// C[i][j] = Σ(p=0 to k-1) A[i][p] × B[p][j] for all i ∈ [0,m-1], j ∈ [0,n-1]
///
/// # Arguments
/// * `lhs` - Left-hand side matrix data (row-major, m×k)
/// * `rhs` - Right-hand side matrix data (row-major, k×n)
/// * `result` - Output matrix data (row-major, m×n)
/// * `m` - Number of rows in A and C
/// * `k` - Number of columns in A and rows in B
/// * `n` - Number of columns in B and C
///
/// # Returns
/// Result indicating success or failure
///
/// # Algorithm Complexity
/// - Time Complexity: O(m × k × n)
/// - Space Complexity: O(1) additional space
pub fn matmul_primitive<T: DataType>(
    lhs: &[T],
    rhs: &[T],
    result: &mut [T],
    m: usize,
    k: usize,
    n: usize,
) -> crate::Result<()>
where
    T: core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Copy + Default,
{
    // Validate input dimensions
    if lhs.len() != m * k {
        return Err(crate::BackendError::InvalidInput(format!(
            "LHS matrix size {} does not match expected {}×{} = {}",
            lhs.len(),
            m,
            k,
            m * k
        )));
    }
    if rhs.len() != k * n {
        return Err(crate::BackendError::InvalidInput(format!(
            "RHS matrix size {} does not match expected {}×{} = {}",
            rhs.len(),
            k,
            n,
            k * n
        )));
    }
    if result.len() != m * n {
        return Err(crate::BackendError::InvalidInput(format!(
            "Result matrix size {} does not match expected {}×{} = {}",
            result.len(),
            m,
            n,
            m * n
        )));
    }

    // Initialize result to zero
    for res in result.iter_mut() {
        *res = T::default();
    }

    // Perform matrix multiplication
    // TODO: Future optimization with blocking/tiling for cache efficiency
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                let a_val = lhs[i * k + l];
                let b_val = rhs[l * n + j];
                result[i * n + j] = result[i * n + j] + a_val * b_val;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    #[test]
    fn test_matmul_primitive_2x2() {
        // Test 2×2 matrix multiplication
        let lhs = [
            Float32::new(1.0), Float32::new(2.0),  // [1, 2]
            Float32::new(3.0), Float32::new(4.0),  // [3, 4]
        ];
        let rhs = [
            Float32::new(5.0), Float32::new(6.0),  // [5, 6]
            Float32::new(7.0), Float32::new(8.0),  // [7, 8]
        ];
        let mut result = [Float32::new(0.0); 4];

        matmul_primitive(&lhs, &rhs, &mut result, 2, 2, 2).unwrap();

        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //           [3*5+4*7, 3*6+4*8] = [43, 50]
        let expected = [
            Float32::new(19.0), Float32::new(22.0),
            Float32::new(43.0), Float32::new(50.0),
        ];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_primitive_dimension_mismatch() {
        let lhs = [Float32::new(1.0)];
        let rhs = [Float32::new(1.0), Float32::new(2.0)];
        let mut result = [Float32::new(0.0); 2];

        let result_op = matmul_primitive(&lhs, &rhs, &mut result, 1, 1, 2);
        assert!(result_op.is_err());
    }
}