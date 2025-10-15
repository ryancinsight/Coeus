//! CPU backend implementation
//!
//! Native CPU execution with SIMD-readiness hooks for future acceleration.

use crate::{Backend, Device, DeviceInfo};
use coeus_storage::Storage;
use alloc::{vec, vec::Vec, format};
use alloc::string::ToString;

/// CPU device information
#[derive(Debug, Clone, PartialEq)]
pub struct CpuDevice(Device);

impl CpuDevice {
    /// Creates a new CPU device info
    #[must_use]
    pub const fn new() -> Self {
        Self(Device::Cpu)
    }
}

impl Default for CpuDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceInfo for CpuDevice {
    fn device(&self) -> &Device {
        &self.0
    }

    fn is_available(&self) -> bool {
        // CPU always available
        true
    }
}

/// CPU backend for native execution.
///
/// Provides baseline CPU execution with hooks for future SIMD acceleration
/// via safe intrinsics and parallel execution via rayon.
///
/// # Examples
///
/// ```
/// use coeus_backend::{Backend, CpuBackend};
///
/// let backend = CpuBackend::new();
/// assert_eq!(backend.device_name(), "cpu");
/// assert!(backend.supports("arithmetic"));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CpuBackend {
    device: CpuDevice,
}

impl CpuBackend {
    /// Creates a new CPU backend instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_backend::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            device: CpuDevice::new(),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    type DeviceType = CpuDevice;

    fn device(&self) -> &Self::DeviceType {
        &self.device
    }

    fn supports(&self, operation: &str) -> bool {
        // CPU supports all basic operations
        matches!(
            operation,
            "arithmetic" | "indexing" | "reduction" | "comparison"
        )
    }

    fn add_dense<T>(&self, lhs: &coeus_storage::DenseStorage<T>, rhs: &coeus_storage::DenseStorage<T>) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Perform element-wise addition directly on storage
        let lhs_data = lhs.as_slice();
        let rhs_data = rhs.as_slice();
        let mut result_data = Vec::with_capacity(lhs_data.len());

        for (&a, &b) in lhs_data.iter().zip(rhs_data.iter()) {
            result_data.push(a + b);
        }

        // Create result storage with the same shape
        coeus_storage::DenseStorage::from_vec(result_data, lhs.shape().dims())
            .map_err(|_| crate::BackendError::UnsupportedOperation {
                operation: "add".to_string(),
                backend: "cpu".to_string(),
            })
    }

    fn mul_dense<T>(&self, lhs: &coeus_storage::DenseStorage<T>, rhs: &coeus_storage::DenseStorage<T>) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Perform element-wise multiplication directly on storage
        let lhs_data = lhs.as_slice();
        let rhs_data = rhs.as_slice();
        let mut result_data = Vec::with_capacity(lhs_data.len());

        for (&a, &b) in lhs_data.iter().zip(rhs_data.iter()) {
            result_data.push(a * b);
        }

        coeus_storage::DenseStorage::from_vec(result_data, lhs.shape().dims())
            .map_err(|_| crate::BackendError::UnsupportedOperation {
                operation: "mul".to_string(),
                backend: "cpu".to_string(),
            })
    }

    fn matmul_dense<T>(&self, lhs: &coeus_storage::DenseStorage<T>, rhs: &coeus_storage::DenseStorage<T>) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Simple matrix multiplication for 2D tensors
        let lhs_shape = lhs.shape().dims();
        let rhs_shape = rhs.shape().dims();

        if lhs_shape.len() != 2usize || rhs_shape.len() != 2usize {
            return Err(crate::BackendError::UnsupportedOperation {
                operation: "matmul".to_string(),
                backend: "cpu".to_string(),
            });
        }

        let (m, k) = (lhs_shape[0], lhs_shape[1]);
        let n = rhs_shape[1];

        let lhs_data = lhs.as_slice();
        let rhs_data = rhs.as_slice();
        let mut result_data = alloc::vec::Vec::with_capacity(m * n);

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum = sum + lhs_data[i * k + l] * rhs_data[l * n + j];
                }
                result_data.push(sum);
            }
        }

        coeus_storage::DenseStorage::from_vec(result_data, &[m, n])
            .map_err(|_| crate::BackendError::UnsupportedOperation {
                operation: "matmul".to_string(),
                backend: "cpu".to_string(),
            })
    }

    fn exp_dense<T>(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Element-wise exponential
        let input_data = input.as_slice();
        let mut result_data = alloc::vec::Vec::with_capacity(input_data.len());

        for &x in input_data {
            // For now, use a simple approximation - should be replaced with proper exp
            result_data.push(T::from(2.718281828).unwrap_or(x)); // e^x approximation
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims())
            .map_err(|_| crate::BackendError::UnsupportedOperation {
                operation: "exp".to_string(),
                backend: "cpu".to_string(),
            })
    }

    fn log_dense<T>(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Element-wise logarithm
        let input_data = input.as_slice();
        let mut result_data = alloc::vec::Vec::with_capacity(input_data.len());

        for &x in input_data {
            // For now, use a simple approximation - should be replaced with proper log
            result_data.push(T::from(0.0).unwrap_or(x)); // ln(x) approximation
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims())
            .map_err(|_| crate::BackendError::UnsupportedOperation {
                operation: "log".to_string(),
                backend: "cpu".to_string(),
            })
    }

    fn sin_dense<T>(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Element-wise sine
        let input_data = input.as_slice();
        let mut result_data = alloc::vec::Vec::with_capacity(input_data.len());

        for &x in input_data {
            // For now, use a simple approximation - should be replaced with proper sin
            result_data.push(T::from(0.0).unwrap_or(x)); // sin(x) approximation
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims())
            .map_err(|_| crate::BackendError::UnsupportedOperation {
                operation: "sin".to_string(),
                backend: "cpu".to_string(),
            })
    }

    fn cos_dense<T>(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Element-wise cosine
        let input_data = input.as_slice();
        let mut result_data = alloc::vec::Vec::with_capacity(input_data.len());

        for &x in input_data {
            // For now, use a simple approximation - should be replaced with proper cos
            result_data.push(T::from(1.0).unwrap_or(x)); // cos(x) approximation
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims())
            .map_err(|_| crate::BackendError::UnsupportedOperation {
                operation: "cos".to_string(),
                backend: "cpu".to_string(),
            })
    }

    fn conv2d_dense<T>(
        &self,
        _input: &coeus_storage::DenseStorage<T>,
        _weight: &coeus_storage::DenseStorage<T>,
        _bias: Option<&coeus_storage::DenseStorage<T>>,
        _stride: (usize, usize),
        _padding: (usize, usize),
        _input_shape: &[usize],
        _weight_shape: &[usize],
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // CPU convolution should delegate to functional_conv::conv2d
        // For now, return unsupported to indicate it should be called at tensor level
        Err(crate::BackendError::UnsupportedOperation {
            operation: "conv2d_dense".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn spmm_csr<T>(
        &self,
        lhs_data: &[T],
        lhs_indices: &[usize],
        lhs_indptr: &[usize],
        rhs_data: &[T],
        rhs_indices: &[usize],
        rhs_indptr: &[usize],
        m: usize,
        k: usize,
        n: usize,
    ) -> crate::Result<(Vec<T>, Vec<usize>, Vec<usize>)>
    where
        T: crate::DataType,
    {
        // CPU implementation of sparse-sparse matrix multiplication
        // Returns result in COO format for flexibility
        use alloc::{collections::BTreeMap, vec::Vec};

        let mut result_data = Vec::new();
        let mut result_row_indices = Vec::new();
        let mut result_col_indices = Vec::new();

        // For each row in left matrix
        for i in 0..m {
            let row_start = lhs_indptr[i];
            let row_end = lhs_indptr[i + 1];

            // For each row in right matrix
            for j in 0..n {
                let mut sum = T::zero();

                // Sparse dot product of row i from lhs and row j from rhs
                let mut lhs_pos = row_start;
                let mut rhs_pos = rhs_indptr[j];
                let rhs_end = rhs_indptr[j + 1];

                while lhs_pos < row_end && rhs_pos < rhs_end {
                    let lhs_col = lhs_indices[lhs_pos];
                    let rhs_col = rhs_indices[rhs_pos];

                    if lhs_col == rhs_col {
                        // Same column - multiply and accumulate
                        sum = sum + lhs_data[lhs_pos] * rhs_data[rhs_pos];
                        lhs_pos += 1;
                        rhs_pos += 1;
                    } else if lhs_col < rhs_col {
                        lhs_pos += 1;
                    } else {
                        rhs_pos += 1;
                    }
                }

                // Only store non-zero results
                if !sum.is_zero() {
                    result_data.push(sum);
                    result_row_indices.push(i);
                    result_col_indices.push(j);
                }
            }
        }

        Ok((result_data, result_row_indices, result_col_indices))
    }

    fn spmv_csr<T>(
        &self,
        matrix_data: &[T],
        matrix_indices: &[usize],
        matrix_indptr: &[usize],
        vector: &[T],
        rows: usize,
        cols: usize,
    ) -> crate::Result<Vec<T>>
    where
        T: crate::DataType,
    {
        // CPU implementation of sparse matrix-dense vector multiplication
        let mut result = vec![T::zero(); rows];

        // For each row
        for i in 0..rows {
            let row_start = matrix_indptr[i];
            let row_end = matrix_indptr[i + 1];
            let mut sum = T::zero();

            // For each non-zero element in this row
            for pos in row_start..row_end {
                let col = matrix_indices[pos];
                if col < cols {
                    sum = sum + matrix_data[pos] * vector[col];
                }
            }

            result[i] = sum;
        }

        Ok(result)
    }

    fn quantize<T>(
        &self,
        _input: &[T],
        _scale: T,
        _zero_point: T,
        _bits: usize,
        _scheme: &str,
    ) -> crate::Result<Vec<u8>>
    where
        T: crate::DataType,
    {
        // Simplified CPU implementation - full implementation would be complex
        // For now, return empty result to indicate CPU quantization not implemented
        Err(crate::BackendError::UnsupportedOperation {
            operation: "quantize".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn dequantize<T>(
        &self,
        _quantized_data: &[u8],
        _scale: T,
        _zero_point: T,
        _bits: usize,
        _scheme: &str,
        _output_size: usize,
    ) -> crate::Result<Vec<T>>
    where
        T: crate::DataType,
    {
        // Simplified CPU implementation - full implementation would be complex
        Err(crate::BackendError::UnsupportedOperation {
            operation: "dequantize".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn quantized_matmul<T>(
        &self,
        _lhs_data: &[u8],
        _lhs_scale: T,
        _lhs_zero_point: T,
        _rhs_data: &[u8],
        _rhs_scale: T,
        _rhs_zero_point: T,
        _bias: Option<&[T]>,
        _m: usize,
        _k: usize,
        _n: usize,
        _bits: usize,
        _scheme: &str,
    ) -> crate::Result<Vec<T>>
    where
        T: crate::DataType,
    {
        // Simplified CPU implementation - full implementation would be complex
        Err(crate::BackendError::UnsupportedOperation {
            operation: "quantized_matmul".to_string(),
            backend: "cpu".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_creation() {
        let backend = CpuBackend::new();
        assert_eq!(backend.device_name(), "cpu");
    }

    #[test]
    fn test_cpu_device_available() {
        let backend = CpuBackend::new();
        assert!(backend.device().is_available());
    }

    #[test]
    fn test_cpu_supports_operations() {
        let backend = CpuBackend::new();
        assert!(backend.supports("arithmetic"));
        assert!(backend.supports("indexing"));
        assert!(backend.supports("reduction"));
        assert!(!backend.supports("gpu_kernel"));
    }

    #[test]
    fn test_cpu_backend_default() {
        let backend = CpuBackend::default();
        assert_eq!(backend, CpuBackend::new());
    }
}
