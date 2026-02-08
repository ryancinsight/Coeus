//! CPU matrix multiplication primitive
//!
//! Provides optimized matrix multiplication for CPU execution.

use crate::DataType;
use dtype::num_traits;

/// General Matrix Multiplication (GEMM) primitive for CPU backend
///
/// Performs: C = alpha * (A @ B) + beta * C
///
/// # Arguments
/// * `alpha` - Scalar for A @ B
/// * `lhs` - Matrix A data (row-major, m×k)
/// * `rhs` - Matrix B data (row-major, k×n)
/// * `beta` - Scalar for C
/// * `result` - Matrix C data (row-major, m×n) - Input/Output
/// * `m` - Number of rows in A and C
/// * `k` - Number of columns in A and rows in B
/// * `n` - Number of columns in B and C
pub fn gemm_primitive<T>(
    alpha: T,
    lhs: &[T],
    rhs: &[T],
    beta: T,
    result: &mut [T],
    m: usize,
    k: usize,
    n: usize,
) -> crate::Result<()>
where
    T: DataType + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Copy + Default + PartialEq,
{
    // Validate input dimensions
    if lhs.len() != m * k {
        return Err(crate::BackendError::InvalidInput(format!(
            "LHS matrix size {} does not match expected {}×{} = {}",
            lhs.len(), m, k, m * k
        )));
    }
    if rhs.len() != k * n {
        return Err(crate::BackendError::InvalidInput(format!(
            "RHS matrix size {} does not match expected {}×{} = {}",
            rhs.len(), k, n, k * n
        )));
    }
    if result.len() != m * n {
        return Err(crate::BackendError::InvalidInput(format!(
            "Result matrix size {} does not match expected {}×{} = {}",
            result.len(), m, n, m * n
        )));
    }

    // If beta is effectively zero, we can just overwrite result (standard matmul)
    // If beta is non-zero, we scale existing C
    let zero = T::default();
    
    // Naive implementation for now (O(m*k*n))
    // TODO: Implement tiling/blocking for cache efficiency
    for i in 0..m {
        for j in 0..n {
            let mut acc = zero;
            for l in 0..k {
                let a_val = lhs[i * k + l];
                let b_val = rhs[l * n + j];
                acc = acc + a_val * b_val;
            }
            
            let idx = i * n + j;
            if beta != zero {
                 result[idx] = alpha * acc + beta * result[idx];
            } else {
                 result[idx] = alpha * acc;
            }
        }
    }

    Ok(())
}

/// Matrix multiplication primitive (wrapper around GEMM)
///
/// Performs matrix multiplication: C = A × B where A is m×k, B is k×n, C is m×n
///
/// # Arguments
/// * `lhs` - Left-hand side matrix data (row-major, m×k)
/// * `rhs` - Right-hand side matrix data (row-major, k×n)
/// * `result` - Output matrix data (row-major, m×n)
/// * `m` - Number of rows in A and C
/// * `k` - Number of columns in A and rows in B
/// * `n` - Number of columns in B and C
pub fn matmul_primitive<T>(
    lhs: &[T],
    rhs: &[T],
    result: &mut [T],
    m: usize,
    k: usize,
    n: usize,
) -> crate::Result<()>
where
    T: DataType + core::ops::Add<Output = T> + core::ops::Mul<Output = T> + Copy + Default + PartialEq + num_traits::One,
{
    gemm_primitive(T::one(), lhs, rhs, T::default(), result, m, k, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;
    use dtype::num_traits::One;

    #[test]
    fn test_gemm_primitive_basic() {
        // [1 2]   [5 6]   [19 22]
        // [3 4] @ [7 8] = [43 50]
        let lhs = [Float32::new(1.0), Float32::new(2.0),
                   Float32::new(3.0), Float32::new(4.0)];
        let rhs = [Float32::new(5.0), Float32::new(6.0),
                   Float32::new(7.0), Float32::new(8.0)];
        let mut result = [Float32::new(0.0); 4];

        gemm_primitive(
            Float32::one(), &lhs, &rhs, Float32::new(0.0), &mut result,
            2, 2, 2
        ).unwrap();

        let expected = [Float32::new(19.0), Float32::new(22.0),
                        Float32::new(43.0), Float32::new(50.0)];
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_gemm_primitive_addmm() {
        // alpha=2, beta=0.5
        // C = 0.5 * C_in + 2 * (A @ B)
        // C_in = [1, 1, 1, 1]
        // A @ B = [19, 22, 43, 50]
        // Expected = [0.5 + 38, 0.5 + 44, 0.5 + 86, 0.5 + 100] = [38.5, 44.5, 86.5, 100.5]
        
        let lhs = [Float32::new(1.0), Float32::new(2.0),
                   Float32::new(3.0), Float32::new(4.0)];
        let rhs = [Float32::new(5.0), Float32::new(6.0),
                   Float32::new(7.0), Float32::new(8.0)];
        let mut result = [Float32::new(1.0); 4];

        gemm_primitive(
            Float32::new(2.0), &lhs, &rhs, Float32::new(0.5), &mut result,
            2, 2, 2
        ).unwrap();

        let expected = [Float32::new(38.5), Float32::new(44.5),
                        Float32::new(86.5), Float32::new(100.5)];
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_matmul_primitive_wrapper() {
         let lhs = [Float32::new(1.0)];
         let rhs = [Float32::new(2.0)];
         let mut result = [Float32::new(0.0)];
         
         matmul_primitive(&lhs, &rhs, &mut result, 1, 1, 1).unwrap();
         assert_eq!(result[0], Float32::new(2.0));
    }
}
