//! CPU backend implementation
//!
//! Native CPU execution with SIMD-readiness hooks for future acceleration.

use crate::{Backend, DataType, Device, DeviceInfo};
use storage::{AsAny, Storage, StorageFromVec};
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
/// use backend::{Backend, CpuBackend};
/// use dtype::float::Float32;
///
/// let backend = CpuBackend::<Float32>::new();
/// assert_eq!(backend.device_name(), "cpu");
/// assert!(backend.supports("arithmetic"));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CpuBackend<T: crate::DataType> where T: PartialOrd {
    device: CpuDevice,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: crate::DataType + std::cmp::PartialOrd> Default for CpuBackend<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: crate::DataType + std::cmp::PartialOrd> CpuBackend<T> {
    /// Creates a new CPU backend instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use backend::CpuBackend;
    /// use dtype::float::Float32;
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

impl<T: DataType + std::cmp::PartialOrd> Backend for CpuBackend<T>
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
        lhs: &storage::DenseStorage<T>,
        rhs: &storage::DenseStorage<T>,
    ) -> crate::Result<storage::DenseStorage<T>>
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
        storage::DenseStorage::from_vec(result_data, lhs.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "add".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn mul_dense(
        &self,
        lhs: &storage::DenseStorage<T>,
        rhs: &storage::DenseStorage<T>,
    ) -> crate::Result<storage::DenseStorage<T>>
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

        storage::DenseStorage::from_vec(result_data, lhs.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "mul".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn sub_dense(
        &self,
        lhs: &storage::DenseStorage<T>,
        rhs: &storage::DenseStorage<T>,
    ) -> crate::Result<storage::DenseStorage<T>>
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

        storage::DenseStorage::from_vec(result_data, lhs.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "sub".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn matmul_dense(
        &self,
        lhs: &storage::DenseStorage<T>,
        rhs: &storage::DenseStorage<T>,
    ) -> crate::Result<storage::DenseStorage<T>>
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

        storage::DenseStorage::from_vec(result_data, &[m, n]).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "matmul_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn exp_dense(
        &self,
        input: &storage::DenseStorage<T>,
    ) -> crate::Result<storage::DenseStorage<T>>
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

        storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "exp_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn relu_dense(
        &self,
        input: &storage::DenseStorage<T>,
    ) -> crate::Result<storage::DenseStorage<T>>
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

        storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "relu_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn sum_dense(&self, input: &storage::DenseStorage<T>) -> crate::Result<T>
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

    fn max_dense(&self, input: &storage::DenseStorage<T>) -> crate::Result<T>
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

    fn min_dense(&self, input: &storage::DenseStorage<T>) -> crate::Result<T>
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

    fn argmax_dense(&self, input: &storage::DenseStorage<T>) -> crate::Result<usize>
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

    fn argmin_dense(&self, input: &storage::DenseStorage<T>) -> crate::Result<usize>
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
        input: &storage::DenseStorage<T>,
    ) -> crate::Result<storage::DenseStorage<T>>
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

        storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "log_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn sin_dense(
        &self,
        input: &storage::DenseStorage<T>,
    ) -> crate::Result<storage::DenseStorage<T>>
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

        storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "sin_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn cos_dense(
        &self,
        input: &storage::DenseStorage<T>,
    ) -> crate::Result<storage::DenseStorage<T>>
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

        storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "cos_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn mean_dense(
        &self,
        input: &storage::DenseStorage<T>,
        _axes: Option<&[usize]>,
    ) -> crate::Result<storage::DenseStorage<T>>
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

        storage::DenseStorage::from_vec(result_data, &[1]).map_err(|_| {
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
        other: &storage::DenseStorage<Self::Data>,
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
    ) -> crate::Result<storage::CooStorage<T>> {
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
        rhs: &storage::DenseStorage<Self::Data>,
        m: usize,
        k: usize,
        n: usize,
    ) -> crate::Result<storage::DenseStorage<Self::Data>> {
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
        storage::DenseStorage::from_vec(result_data, &[m, n]).map_err(|_| crate::BackendError::UnsupportedOperation {
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
    ) -> crate::Result<storage::CooStorage<T>> {
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
    ) -> crate::Result<storage::CooStorage<T>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "coo_mul_sparse".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn quantize(&self, input: &storage::DenseStorage<Self::Data>, levels: usize) -> crate::Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        // Quantize the dense storage
        self.quantize_dense_impl(input, levels)
    }

    fn conv2d_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
        _weight: &storage::DenseStorage<Self::Data>,
    ) -> crate::Result<storage::DenseStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "conv2d_dense".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn clip_info_nce_loss(&self, image_embeddings: &storage::DenseStorage<Self::Data>, text_embeddings: &storage::DenseStorage<Self::Data>, temperature: f32) -> crate::Result<Self::Data> {
        CpuBackend::clip_info_nce_loss(self, image_embeddings, text_embeddings, temperature)
    }

    fn clip_attention(&self, queries: &storage::DenseStorage<<CpuBackend<T> as Backend>::Data>, keys: &storage::DenseStorage<<CpuBackend<T> as Backend>::Data>, values: &storage::DenseStorage<<CpuBackend<T> as Backend>::Data>, num_heads: usize) -> crate::Result<storage::DenseStorage<<CpuBackend<T> as Backend>::Data>> {
        CpuBackend::clip_attention(self, queries, keys, values, num_heads)
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
        input: &storage::DenseStorage<T>,
        levels: usize,
    ) -> crate::Result<storage::DenseStorage<T>> {
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

        storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "quantize".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn relu_dense(
        &self,
        input: &storage::DenseStorage<T>,
    ) -> crate::Result<storage::DenseStorage<T>>
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

        storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "relu_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn sum_dense(
        &self,
        input: &storage::DenseStorage<T>,
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
        input: &storage::DenseStorage<T>,
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
        input: &storage::DenseStorage<T>,
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

    fn argmax_dense(&self, input: &storage::DenseStorage<T>) -> crate::Result<usize>
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

    fn argmin_dense(&self, input: &storage::DenseStorage<T>) -> crate::Result<usize>
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
        input: &storage::DenseStorage<T>,
    ) -> crate::Result<storage::DenseStorage<T>>
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

        storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "log_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn sin_dense(
        &self,
        input: &storage::DenseStorage<T>,
    ) -> crate::Result<storage::DenseStorage<T>>
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

        storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "sin_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn cos_dense(
        &self,
        input: &storage::DenseStorage<T>,
    ) -> crate::Result<storage::DenseStorage<T>>
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

        storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "cos_dense".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    /// Compute CLIP InfoNCE loss using CPU implementation
    fn clip_info_nce_loss(
        &self,
        image_embeddings: &storage::DenseStorage<<CpuBackend<T> as Backend>::Data>,
        text_embeddings: &storage::DenseStorage<<CpuBackend<T> as Backend>::Data>,
        temperature: f32,
    ) -> crate::Result<<CpuBackend<T> as Backend>::Data> {
        use std::collections::HashMap;

        // Get embedding dimensions
        let image_shape = image_embeddings.shape().dims();
        let text_shape = text_embeddings.shape().dims();

        // Validate shapes: both should be [batch_size, embed_dim]
        if image_shape.len() != 2 || text_shape.len() != 2 {
            return Err(crate::BackendError::InvalidInput(
                "Embeddings must be 2D tensors [batch_size, embed_dim]".to_string(),
            ));
        }

        let batch_size = image_shape[0];
        let embed_dim = image_shape[1];

        if text_shape[0] != batch_size || text_shape[1] != embed_dim {
            return Err(crate::BackendError::InvalidInput(
                "Image and text embeddings must have same shape [batch_size, embed_dim]".to_string(),
            ));
        }

        let image_data = image_embeddings.as_slice();
        let text_data = text_embeddings.as_slice();

        // Compute L2 normalization for all embeddings
        let mut image_norms = vec![T::zero(); batch_size];
        let mut text_norms = vec![T::zero(); batch_size];

        for i in 0..batch_size {
            for j in 0..embed_dim {
                let img_val = image_data[i * embed_dim + j];
                let txt_val = text_data[i * embed_dim + j];

                // Convert to f64 for computation
                let img_f64 = img_val.to_f64().unwrap_or(0.0);
                let txt_f64 = txt_val.to_f64().unwrap_or(0.0);

                image_norms[i] = T::from(image_norms[i].to_f64().unwrap_or(0.0) + img_f64 * img_f64).unwrap_or(T::zero());
                text_norms[i] = T::from(text_norms[i].to_f64().unwrap_or(0.0) + txt_f64 * txt_f64).unwrap_or(T::zero());
            }
        }

        // Take square roots and handle zeros
        for i in 0..batch_size {
            let img_norm_f64 = image_norms[i].to_f64().unwrap_or(0.0).sqrt().max(1e-10);
            let txt_norm_f64 = text_norms[i].to_f64().unwrap_or(0.0).sqrt().max(1e-10);
            image_norms[i] = T::from(img_norm_f64).unwrap_or(T::one());
            text_norms[i] = T::from(txt_norm_f64).unwrap_or(T::one());
        }

        // Compute similarity matrix [batch_size, batch_size]
        let mut logits = vec![vec![T::zero(); batch_size]; batch_size];

        for i in 0..batch_size {
            for j in 0..batch_size {
                let mut similarity = 0.0;

                for k in 0..embed_dim {
                    let img_i = image_data[i * embed_dim + k].to_f64().unwrap_or(0.0) / image_norms[i].to_f64().unwrap_or(1.0);
                    let txt_j = text_data[j * embed_dim + k].to_f64().unwrap_or(0.0) / text_norms[j].to_f64().unwrap_or(1.0);
                    similarity += img_i * txt_j;
                }

                // Apply temperature scaling
                similarity /= temperature as f64;
                logits[i][j] = T::from(similarity).unwrap_or(T::zero());
            }
        }

        // Compute InfoNCE loss
        let mut total_loss = 0.0;

        for i in 0..batch_size {
            // Positive pair similarity (diagonal)
            let pos_sim = logits[i][i].to_f64().unwrap_or(0.0);

            // Negative pair similarities (off-diagonal)
            let mut neg_sum = 0.0;
            for j in 0..batch_size {
                if i != j {
                    neg_sum += (logits[i][j].to_f64().unwrap_or(0.0)).exp();
                }
            }

            // Cross-entropy loss for this positive pair
            let pos_prob = pos_sim.exp() / (pos_sim.exp() + neg_sum);
            let loss = -(pos_prob.ln().max(-1e10)); // Clamp for numerical stability
            total_loss += loss;
        }

        // Average loss across batch
        let avg_loss = total_loss / batch_size as f64;
        Ok(T::from(avg_loss).unwrap_or(T::zero()))
    }

    /// Compute CLIP attention mechanism using CPU implementation
    fn clip_attention(
        &self,
        queries: &storage::DenseStorage<<CpuBackend<T> as Backend>::Data>,
        keys: &storage::DenseStorage<<CpuBackend<T> as Backend>::Data>,
        values: &storage::DenseStorage<<CpuBackend<T> as Backend>::Data>,
        num_heads: usize,
    ) -> crate::Result<storage::DenseStorage<<CpuBackend<T> as Backend>::Data>> {
        let query_shape = queries.shape().dims();
        let key_shape = keys.shape().dims();
        let value_shape = values.shape().dims();

        // Validate shapes: [batch_size, seq_len, embed_dim]
        if query_shape.len() != 3 || key_shape.len() != 3 || value_shape.len() != 3 {
            return Err(crate::BackendError::InvalidInput(
                "All inputs must be 3D tensors [batch_size, seq_len, embed_dim]".to_string(),
            ));
        }

        let batch_size = query_shape[0];
        let seq_len_q = query_shape[1];
        let seq_len_kv = key_shape[1];
        let embed_dim = query_shape[2];

        if key_shape[0] != batch_size || value_shape[0] != batch_size ||
           key_shape[2] != embed_dim || value_shape[2] != embed_dim {
            return Err(crate::BackendError::InvalidInput(
                "Incompatible tensor shapes for attention".to_string(),
            ));
        }

        if embed_dim % num_heads != 0 {
            return Err(crate::BackendError::InvalidInput(
                format!("embed_dim ({}) must be divisible by num_heads ({})", embed_dim, num_heads),
            ));
        }

        let head_dim = embed_dim / num_heads;

        let query_data = queries.as_slice();
        let key_data = keys.as_slice();
        let value_data = values.as_slice();

        // Allocate output tensor
        let output_size = batch_size * seq_len_q * embed_dim;
        let mut output_data = vec![T::zero(); output_size];

        // Process each batch, head, and query position
        for batch_idx in 0..batch_size {
            for head_idx in 0..num_heads {
                for query_pos in 0..seq_len_q {
                    // Compute attention scores for this query against all keys
                    let mut attention_scores = vec![0.0; seq_len_kv];
                    let mut max_score = f64::NEG_INFINITY;

                    // First pass: compute raw attention scores and find max for numerical stability
                    for kv_pos in 0..seq_len_kv {
                        let mut dot_product = 0.0;

                        for d in 0..head_dim {
                            let head_offset = head_idx * head_dim;
                            let q_idx = ((batch_idx * seq_len_q + query_pos) * embed_dim) + head_offset + d;
                            let k_idx = ((batch_idx * seq_len_kv + kv_pos) * embed_dim) + head_offset + d;

                            let q_val = query_data[q_idx].to_f64().unwrap_or(0.0);
                            let k_val = key_data[k_idx].to_f64().unwrap_or(0.0);
                            dot_product += q_val * k_val;
                        }

                        // Scale by sqrt(head_dim)
                        let scaled_score = dot_product / (head_dim as f32).sqrt() as f64;
                        attention_scores[kv_pos] = scaled_score;
                        max_score = max_score.max(scaled_score);
                    }

                    // Second pass: compute softmax weights
                    let mut weights = vec![0.0; seq_len_kv];
                    let mut weight_sum = 0.0;

                    for kv_pos in 0..seq_len_kv {
                        let exp_score = (attention_scores[kv_pos] - max_score).exp();
                        weights[kv_pos] = exp_score;
                        weight_sum += exp_score;
                    }

                    // Third pass: apply attention weights to values
                    for d in 0..head_dim {
                        let head_offset = head_idx * head_dim;
                        let out_idx = ((batch_idx * seq_len_q + query_pos) * embed_dim) + head_offset + d;
                        let mut result = 0.0;

                        for kv_pos in 0..seq_len_kv {
                            let v_idx = ((batch_idx * seq_len_kv + kv_pos) * embed_dim) + head_offset + d;
                            let v_val = value_data[v_idx].to_f64().unwrap_or(0.0);
                            let weight = weights[kv_pos] / weight_sum;
                            result += weight * v_val;
                        }

                        output_data[out_idx] = T::from(result).unwrap_or(T::zero());
                    }
                }
            }
        }

        storage::DenseStorage::from_vec(output_data, query_shape).map_err(|_| {
            crate::BackendError::InvalidInput("Failed to create output tensor".to_string())
        })
    }
}
