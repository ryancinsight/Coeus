//! CPU backend implementation
//!
//! Native CPU execution with SIMD-readiness hooks for future acceleration.

use crate::{Backend, DataType, Device, DeviceInfo};
use coeus_storage::{AsAny, Storage, StorageFromVec};
#[cfg(not(feature = "std"))]
use libm;
#[cfg(feature = "std")]
use std;
use std::vec::Vec;

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

    fn memory_gb(&self) -> usize {
        16 // Assume 16GB system RAM for CPU
    }

    fn compute_units(&self) -> usize {
        num_cpus::get()
    }
}

/// CPU backend for native execution with associated types.
///
/// Provides baseline CPU execution with hooks for future SIMD acceleration
/// via safe intrinsics and parallel execution via rayon.
///
/// # Examples
///
/// ```
/// use coeus_backend::{Backend, CpuBackend};
/// use coeus_dtype::float::Float32;
///
/// let backend = CpuBackend::<Float32>::new();
/// assert_eq!(backend.device_name(), "cpu");
/// assert!(backend.supports("arithmetic"));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CpuBackend<T: crate::DataType> {
    device: CpuDevice,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: crate::DataType> Default for CpuBackend<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: crate::DataType> CpuBackend<T> {
    /// Creates a new CPU backend instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_backend::CpuBackend;
    /// use coeus_dtype::float::Float32;
    ///
    /// let backend = CpuBackend::<Float32>::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            device: CpuDevice::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: DataType> Backend for CpuBackend<T>
{
    /// Data type supported by this backend
    type Data = T;

    /// Device type for this backend
    type Device = CpuDevice;

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn device_name(&self) -> &str {
        "cpu"
    }

    fn device_info(&self) -> Box<dyn crate::DeviceInfo> {
        Box::new(self.device.clone())
    }


    fn supports(&self, operation: &str) -> bool {
        // CPU supports all basic operations
        matches!(
            operation,
            "arithmetic" | "indexing" | "reduction" | "comparison"
        )
    }
    fn add_dense(
        &self,
        lhs: &coeus_storage::DenseStorage<T>,
        rhs: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
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
        coeus_storage::DenseStorage::from_vec(result_data, lhs.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "add".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn mul_dense(
        &self,
        lhs: &coeus_storage::DenseStorage<T>,
        rhs: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
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

        coeus_storage::DenseStorage::from_vec(result_data, lhs.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "mul".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn sub_dense(
        &self,
        lhs: &coeus_storage::DenseStorage<T>,
        rhs: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Perform element-wise subtraction directly on storage
        let lhs_data = lhs.as_slice();
        let rhs_data = rhs.as_slice();
        let mut result_data = Vec::with_capacity(lhs_data.len());

        for (&a, &b) in lhs_data.iter().zip(rhs_data.iter()) {
            result_data.push(a - b);
        }

        coeus_storage::DenseStorage::from_vec(result_data, lhs.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "sub".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn matmul_dense(
        &self,
        lhs: &coeus_storage::DenseStorage<T>,
        rhs: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Basic matrix multiplication for 2D tensors
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();

        if lhs_shape.dims().len() != 2 || rhs_shape.dims().len() != 2 {
            return Err(crate::BackendError::UnsupportedOperation {
                operation: "matmul_dense".to_string(),
                backend: "cpu".to_string(),
            });
        }

        let (m, k) = (lhs_shape.dims()[0], lhs_shape.dims()[1]);
        let (k2, n) = (rhs_shape.dims()[0], rhs_shape.dims()[1]);

        if k != k2 {
            return Err(crate::BackendError::UnsupportedOperation {
                operation: "matmul_dense".to_string(),
                backend: "cpu".to_string(),
            });
        }

        let lhs_data = lhs.as_slice();
        let rhs_data = rhs.as_slice();
        let mut result_data = vec![T::zero(); m * n];

        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    result_data[i * n + j] = result_data[i * n + j] + lhs_data[i * k + l] * rhs_data[l * n + j];
                }
            }
        }

        coeus_storage::DenseStorage::from_vec(result_data, &[m, n]).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "matmul_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn exp_dense(
        &self,
        input: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Element-wise exponential
        let input_data = input.as_slice();
        let mut result_data = Vec::with_capacity(input_data.len());

        for &x in input_data.iter() {
            // Convert to f64 for computation, then back
            let x_f64 = x.to_f64().unwrap_or(0.0);
            let exp_result = x_f64.exp();
            result_data.push(T::from(exp_result).unwrap_or(x));
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "exp_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn relu_dense(
        &self,
        input: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: PartialOrd + Default,
    {
        // Element-wise ReLU: max(0, x)
        let input_data = input.as_slice();
        let mut result_data = Vec::with_capacity(input_data.len());

        let zero = T::zero();
        for &x in input_data.iter() {
            result_data.push(if x > zero { x } else { zero });
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "relu_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn sum_dense(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<T>
    where
        T: crate::DataType,
    {
        let input_data = input.as_slice();
        let mut sum = T::zero();

        for &x in input_data.iter() {
            sum = sum + x;
        }

        Ok(sum)
    }

    fn max_dense(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<T>
    where
        T: PartialOrd,
    {
        let input_data = input.as_slice();
        if input_data.is_empty() {
            return Err(crate::BackendError::UnsupportedOperation {
                operation: "max_dense".to_string(),
                backend: "cpu".to_string(),
            });
        }

        let mut max_val = input_data[0];
        for &x in input_data.iter().skip(1) {
            if x > max_val {
                max_val = x;
            }
        }

        Ok(max_val)
    }

    fn min_dense(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<T>
    where
        T: PartialOrd,
    {
        let input_data = input.as_slice();
        if input_data.is_empty() {
            return Err(crate::BackendError::UnsupportedOperation {
                operation: "min_dense".to_string(),
                backend: "cpu".to_string(),
            });
        }

        let mut min_val = input_data[0];
        for &x in input_data.iter().skip(1) {
            if x < min_val {
                min_val = x;
            }
        }

        Ok(min_val)
    }

    fn argmax_dense(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<usize>
    where
        T: PartialOrd,
    {
        let input_data = input.as_slice();
        if input_data.is_empty() {
            return Err(crate::BackendError::UnsupportedOperation {
                operation: "argmax_dense".to_string(),
                backend: "cpu".to_string(),
            });
        }

        let mut max_idx = 0;
        let mut max_val = input_data[0];

        for (i, &x) in input_data.iter().enumerate().skip(1) {
            if x > max_val {
                max_val = x;
                max_idx = i;
            }
        }

        Ok(max_idx)
    }

    fn argmin_dense(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<usize>
    where
        T: PartialOrd,
    {
        let input_data = input.as_slice();
        if input_data.is_empty() {
            return Err(crate::BackendError::UnsupportedOperation {
                operation: "argmin_dense".to_string(),
                backend: "cpu".to_string(),
            });
        }

        let mut min_idx = 0;
        let mut min_val = input_data[0];

        for (i, &x) in input_data.iter().enumerate().skip(1) {
            if x < min_val {
                min_val = x;
                min_idx = i;
            }
        }

        Ok(min_idx)
    }

    fn log_dense(
        &self,
        input: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Element-wise natural logarithm
        let input_data = input.as_slice();
        let mut result_data = Vec::with_capacity(input_data.len());

        for &x in input_data.iter() {
            let x_f64: f64 = x.to_f64().unwrap_or(1.0_f64);
            let log_result = x_f64.ln();
            result_data.push(T::from(log_result).unwrap_or(x));
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "log_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn sin_dense(
        &self,
        input: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Element-wise sine
        let input_data = input.as_slice();
        let mut result_data = Vec::with_capacity(input_data.len());

        for &x in input_data.iter() {
            let x_f64: f64 = x.to_f64().unwrap_or(0.0_f64);
            let sin_result = x_f64.sin();
            result_data.push(T::from(sin_result).unwrap_or(x));
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "sin_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn cos_dense(
        &self,
        input: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Element-wise cosine
        let input_data = input.as_slice();
        let mut result_data = Vec::with_capacity(input_data.len());

        for &x in input_data.iter() {
            let x_f64: f64 = x.to_f64().unwrap_or(0.0_f64);
            let cos_result = x_f64.cos();
            result_data.push(T::from(cos_result).unwrap_or(x));
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "cos_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn mean_dense(
        &self,
        input: &coeus_storage::DenseStorage<T>,
        _axes: Option<&[usize]>,
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Simple mean of all elements (ignoring axes for now)
        let input_data = input.as_slice();
        if input_data.is_empty() {
            return Err(crate::BackendError::UnsupportedOperation {
                operation: "mean_dense".to_string(),
                backend: "cpu".to_string(),
            });
        }

        let mut sum = T::zero();
        for &x in input_data.iter() {
            sum = sum + x;
        }

        let mean = sum / T::from(input_data.len() as f64).unwrap_or(T::one());
        let result_data = vec![mean];

        coeus_storage::DenseStorage::from_vec(result_data, &[1]).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "mean_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn spmm_csr(
        &self,
        data: &[Self::Data],
        indices: &[usize],
        indptr: &[usize],
        other: &coeus_storage::DenseStorage<Self::Data>,
        num_rows: usize,
        num_cols: usize,
    ) -> crate::Result<Vec<Self::Data>> {
        // Sparse matrix-matrix multiplication (CSR format)
        // This is a basic CPU implementation
        let mut result = vec![Self::Data::zero(); num_rows * num_cols];

        // For each row in the sparse matrix
        for i in 0..num_rows {
            let row_start = indptr[i];
            let row_end = indptr[i + 1];

            // For each non-zero element in this row
            for pos in row_start..row_end {
                let col = indices[pos];
                let val = data[pos];

                // Multiply with corresponding row from dense matrix
                if let Some(dense_row) = other.as_slice().get(col * num_cols..(col + 1) * num_cols) {
                    for j in 0..num_cols {
                        result[i * num_cols + j] = result[i * num_cols + j] + val * dense_row[j];
                    }
                }
            }
        }

        Ok(result)
    }

    fn spmv_csr(
        &self,
        data: &[T],
        indices: &[usize],
        indptr: &[usize],
        vector: &[T],
        num_rows: usize,
        _num_cols: usize,
    ) -> crate::Result<Vec<T>> {
        // Validate input dimensions
        if indptr.len() != num_rows + 1 {
            return Err(crate::BackendError::InvalidInput(
                format!("indptr length {} does not match num_rows + 1 = {}", indptr.len(), num_rows + 1),
            ));
        }

        let mut result = vec![T::default(); num_rows];

        // Perform CSR sparse matrix-vector multiplication
        for row in 0..num_rows {
            let row_start = indptr[row];
            let row_end = indptr[row + 1];

            for idx in row_start..row_end {
                if idx >= data.len() || indices[idx] >= vector.len() {
                    return Err(crate::BackendError::InvalidInput(
                        "index out of bounds in CSR data".to_string(),
                    ));
                }
                let col = indices[idx];
                let val = data[idx];
                result[row] = result[row] + val * vector[col];
            }
        }

        Ok(result)
    }

    fn coo_matmul_sparse(
        &self,
        _lhs_data: &[T],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs_data: &[T],
        _rhs_row: &[usize],
        _rhs_col: &[usize],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> crate::Result<coeus_storage::CooStorage<T>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "coo_matmul_sparse".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn coo_matmul_dense(
        &self,
        lhs_data: &[Self::Data],
        lhs_row: &[usize],
        lhs_col: &[usize],
        rhs: &coeus_storage::DenseStorage<Self::Data>,
        m: usize,
        k: usize,
        n: usize,
    ) -> crate::Result<coeus_storage::DenseStorage<Self::Data>> {
        // Create dense result matrix initialized to zero
        let total_elements = rhs.len();
        let mut result_data = vec![Self::Data::zero(); total_elements];

        // For each non-zero element in sparse matrix
        for (&val, (&row, &col)) in lhs_data.iter().zip(lhs_row.iter().zip(lhs_col.iter())) {
            // Add contribution to result[row * n + j] for each j where dense matrix has data
            for j in 0..n {
                let result_idx = row * n + j;
                let dense_idx = col * n + j;
                if result_idx < result_data.len() && dense_idx < rhs.as_slice().len() {
                    result_data[result_idx] = result_data[result_idx] + val * rhs.as_slice()[dense_idx];
                }
            }
        }

        // Return the dense result
        coeus_storage::DenseStorage::from_vec(result_data, &[m, n]).map_err(|_| crate::BackendError::UnsupportedOperation {
            operation: "coo_matmul_dense".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn coo_add_sparse(
        &self,
        _lhs_data: &[T],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs_data: &[T],
        _rhs_row: &[usize],
        _rhs_col: &[usize],
        _m: usize,
        _n: usize,
    ) -> crate::Result<coeus_storage::CooStorage<T>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "coo_add_sparse".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn coo_mul_sparse(
        &self,
        _lhs_data: &[T],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs_data: &[T],
        _rhs_row: &[usize],
        _rhs_col: &[usize],
        _m: usize,
        _n: usize,
    ) -> crate::Result<coeus_storage::CooStorage<T>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "coo_mul_sparse".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn quantize(&self, input: &coeus_storage::DenseStorage<Self::Data>, levels: usize) -> crate::Result<coeus_storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        // Quantize the dense storage
        self.quantize_dense_impl(input, levels)
    }

    fn conv2d_dense(
        &self,
        _input: &coeus_storage::DenseStorage<Self::Data>,
        _weight: &coeus_storage::DenseStorage<Self::Data>,
    ) -> crate::Result<coeus_storage::DenseStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "conv2d_dense".to_string(),
            backend: "cpu".to_string(),
        })
    }
}

// Internal dense quantization implementation
#[allow(dead_code)]
impl<T> CpuBackend<T>
where
    T: crate::DataType + PartialOrd,
{
    fn quantize_dense_impl(
        &self,
        input: &coeus_storage::DenseStorage<T>,
        levels: usize,
    ) -> crate::Result<coeus_storage::DenseStorage<T>> {
        if levels == 0 {
            return Err(crate::BackendError::InvalidInput(
                "Cannot quantize with 0 levels".to_string(),
            ));
        }

        let input_data = input.as_slice();
        let mut result_data = Vec::with_capacity(input_data.len());

        // Find min and max values for scaling
        let mut min_val = *input_data.first().ok_or_else(|| {
            crate::BackendError::InvalidInput("Cannot quantize empty tensor".to_string())
        })?;
        let mut max_val = min_val;

        for &val in input_data {
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }

        let range = if max_val == min_val {
            T::one() // Avoid division by zero
        } else {
            max_val - min_val
        };

        // Quantize each value
        for &val in input_data {
            // Normalize to [0, 1]
            let normalized = if range == T::zero() {
                T::zero()
            } else {
                (val - min_val) / range
            };

            // Convert to f64 for quantization
            let normalized_f64 = normalized.to_f64().unwrap_or(0.0);

            // Quantize to discrete levels
            let quantized_f64 = ((normalized_f64 * (levels - 1) as f64).round()) / (levels - 1) as f64;

            // Scale back to original range
            let quantized = if range == T::zero() {
                min_val
            } else {
                min_val + range * T::from(quantized_f64).unwrap_or(normalized)
            };

            result_data.push(quantized);
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "quantize".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn relu_dense(
        &self,
        input: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: PartialOrd + Default,
    {
        // Element-wise ReLU: max(0, x)
        let input_data = input.as_slice();
        let mut result_data = Vec::with_capacity(input_data.len());

        let zero = T::zero();
        for &value in input_data {
            if value > zero {
                result_data.push(value);
            } else {
                result_data.push(zero);
            }
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "relu_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn sum_dense(
        &self,
        input: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<T>
    where
        T: crate::DataType,
    {
        let data = input.as_slice();
        let mut sum = T::zero();
        for &value in data {
            sum = sum + value;
        }
        Ok(sum)
    }

    fn max_dense(
        &self,
        input: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<T>
    where
        T: PartialOrd,
    {
        let data = input.as_slice();
        if data.is_empty() {
            return Err(crate::BackendError::InvalidInput(
                "Cannot find maximum of empty tensor".to_string(),
            ));
        }

        let mut max_val = data[0];
        for &val in data.iter().skip(1) {
            if val > max_val {
                max_val = val;
            }
        }
        Ok(max_val)
    }

    fn min_dense(
        &self,
        input: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<T>
    where
        T: PartialOrd,
    {
        let data = input.as_slice();
        if data.is_empty() {
            return Err(crate::BackendError::InvalidInput(
                "Cannot find minimum of empty tensor".to_string(),
            ));
        }

        let mut min_val = data[0];
        for &val in data.iter().skip(1) {
            if val < min_val {
                min_val = val;
            }
        }
        Ok(min_val)
    }

    fn argmax_dense(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<usize>
    where
        T: PartialOrd,
    {
        let data = input.as_slice();
        if data.is_empty() {
            return Err(crate::BackendError::InvalidInput(
                "Cannot find argmax of empty tensor".to_string(),
            ));
        }

        let mut max_idx = 0;
        let mut max_val = &data[0];

        for (i, val) in data.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        Ok(max_idx)
    }

    fn argmin_dense(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<usize>
    where
        T: PartialOrd,
    {
        let data = input.as_slice();
        if data.is_empty() {
            return Err(crate::BackendError::InvalidInput(
                "Cannot find argmin of empty tensor".to_string(),
            ));
        }

        let mut min_idx = 0;
        let mut min_val = &data[0];

        for (i, val) in data.iter().enumerate() {
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
        }

        Ok(min_idx)
    }

    fn log_dense(
        &self,
        input: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Element-wise natural logarithm
        let input_data = input.as_slice();
        let mut result_data = Vec::with_capacity(input_data.len());

        for &x in input_data {
            // We need to convert to f64, compute log, then convert back
            let x_f64: f64 = x.to_f64().unwrap_or(1.0_f64);
            #[cfg(feature = "std")]
            let log_result = x_f64.ln();
            #[cfg(not(feature = "std"))]
            let log_result = libm::log(x_f64);
            result_data.push(T::from(log_result).unwrap_or(x));
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "log_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn sin_dense(
        &self,
        input: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Element-wise sine function
        let input_data = input.as_slice();
        let mut result_data = Vec::with_capacity(input_data.len());

        for &x in input_data {
            // Convert to f64, compute sin, then convert back
            let x_f64: f64 = x.to_f64().unwrap_or(0.0_f64);
            #[cfg(feature = "std")]
            let sin_result = x_f64.sin();
            #[cfg(not(feature = "std"))]
            let sin_result = libm::sin(x_f64);
            result_data.push(T::from(sin_result).unwrap_or(x));
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "sin_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn cos_dense(
        &self,
        input: &coeus_storage::DenseStorage<T>,
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType,
    {
        // Element-wise cosine function
        let input_data = input.as_slice();
        let mut result_data = Vec::with_capacity(input_data.len());

        for &x in input_data {
            // Convert to f64, compute cos, then convert back
            let x_f64: f64 = x.to_f64().unwrap_or(0.0_f64);
            #[cfg(feature = "std")]
            let cos_result = x_f64.cos();
            #[cfg(not(feature = "std"))]
            let cos_result = libm::cos(x_f64);
            result_data.push(T::from(cos_result).unwrap_or(x));
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "cos_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

}
