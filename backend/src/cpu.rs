//! CPU backend implementation
//!
//! Native CPU execution with SIMD-readiness hooks for future acceleration.

use crate::{Backend, Device, DeviceInfo};
use coeus_storage::Storage;
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

/// CPU backend for native execution.
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

impl<T> Backend for CpuBackend<T>
where
    T: crate::DataType,
{
    type Data = T;
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
        lhs: &coeus_storage::DenseStorage<Self::Data>,
        rhs: &coeus_storage::DenseStorage<Self::Data>,
    ) -> crate::Result<coeus_storage::DenseStorage<Self::Data>>
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
        lhs: &coeus_storage::DenseStorage<Self::Data>,
        rhs: &coeus_storage::DenseStorage<Self::Data>,
    ) -> crate::Result<coeus_storage::DenseStorage<Self::Data>>
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
        lhs: &coeus_storage::DenseStorage<Self::Data>,
        rhs: &coeus_storage::DenseStorage<Self::Data>,
    ) -> crate::Result<coeus_storage::DenseStorage<Self::Data>>
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
        lhs: &coeus_storage::DenseStorage<Self::Data>,
        rhs: &coeus_storage::DenseStorage<Self::Data>,
    ) -> crate::Result<coeus_storage::DenseStorage<Self::Data>>
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
        let mut result_data = Vec::with_capacity(m * n);

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    sum = sum + lhs_data[i * k + l] * rhs_data[l * n + j];
                }
                result_data.push(sum);
            }
        }

        coeus_storage::DenseStorage::from_vec(result_data, &[m, n]).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "matmul".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn exp_dense(
        &self,
        input: &coeus_storage::DenseStorage<Self::Data>,
    ) -> crate::Result<coeus_storage::DenseStorage<Self::Data>>
    where
        T: crate::DataType,
    {
        // Element-wise exponential
        let input_data = input.as_slice();
        let mut result_data = std::vec::Vec::with_capacity(input_data.len());

        for &x in input_data {
            // Proper exponential function
            let x_f64 = x.to_f64().unwrap_or(0.0);
            #[cfg(feature = "std")]
            let exp_result = std::f64::consts::E.powf(x_f64);
            #[cfg(not(feature = "std"))]
            let exp_result = libm::exp(x_f64);
            result_data.push(T::from(exp_result).unwrap_or(x));
        }

        coeus_storage::DenseStorage::from_vec(result_data, input.shape().dims()).map_err(|_| {
            crate::BackendError::UnsupportedOperation {
                operation: "exp".to_string(),
                backend: "cpu".to_string(),
            }
        })
    }

    fn mean_dense(
        &self,
        input: &coeus_storage::DenseStorage<Self::Data>,
        axes: Option<&[usize]>,
    ) -> crate::Result<coeus_storage::DenseStorage<Self::Data>>
    where
        T: crate::DataType,
    {
        let input_shape = input.shape().dims();
        let input_data = input.as_slice();

        if let Some(axes) = axes {
            // Reduce along specified axes
            if axes.is_empty() {
                // Return copy of input if no axes specified
                return Ok(input.clone());
            }

            // Sort axes in descending order
            let mut sorted_axes = axes.to_vec();
            sorted_axes.sort_unstable_by(|a, b| b.cmp(a));
            sorted_axes.dedup();

            // Calculate output shape
            let mut output_shape = input_shape.to_vec();
            for &axis in &sorted_axes {
                if axis >= input_shape.len() {
                    return Err(crate::BackendError::InvalidInput(
                        format!("Axis {} is out of bounds for tensor with {} dimensions", axis, input_shape.len())
                    ));
                }
                output_shape[axis] = 1;
            }

            // Calculate mean along specified axes
            let mut result_data = vec![T::zero(); output_shape.iter().product()];

            // Simple implementation - sum along axes and divide by count
            let mut divisor = 1.0;

            for &axis in &sorted_axes {
                divisor *= input_shape[axis] as f64;
            }

            // For now, compute global mean if axes specified
            let sum = input_data.iter().fold(T::zero(), |acc, &x| acc + x);
            let mean = if divisor > 0.0 {
                let sum_f64 = sum.to_f64().unwrap_or(0.0);
                T::from(sum_f64 / divisor).unwrap_or(sum)
            } else {
                sum
            };

            // Fill result with mean value (broadcast to output shape)
            result_data.fill(mean);

            coeus_storage::DenseStorage::from_vec(result_data, &output_shape).map_err(|_| {
                crate::BackendError::UnsupportedOperation {
                    operation: "mean_dense".to_string(),
                    backend: "cpu".to_string(),
                }
            })
        } else {
            // Global mean - reduce all axes
            let total_elements = input_data.len() as f64;
            if total_elements == 0.0 {
                return Err(crate::BackendError::InvalidInput(
                    "Cannot compute mean of empty tensor".to_string(),
                ));
            }

            let sum = input_data.iter().fold(T::zero(), |acc, &x| acc + x);
            let sum_f64 = sum.to_f64().unwrap_or(0.0);
            let mean = T::from(sum_f64 / total_elements).unwrap_or(sum);

            // Return scalar result in 0-d tensor
            let result_data = vec![mean];
            coeus_storage::DenseStorage::from_vec(result_data, &[]).map_err(|_| {
                crate::BackendError::UnsupportedOperation {
                    operation: "mean_dense".to_string(),
                    backend: "cpu".to_string(),
                }
            })
        }
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
        // SPMM-CSR: Sparse Matrix (CSR) × Dense Matrix multiplication
        // Result will be dense matrix of size (num_rows × other_cols)

        let other_data = other.as_slice();
        let other_shape = other.shape().dims();

        if other_shape.len() != 2 {
            return Err(crate::BackendError::InvalidInput(
                "Dense matrix must be 2D for SPMM-CSR".to_string(),
            ));
        }

        let other_cols = other_shape[1];

        // Validate CSR format
        if indptr.len() != num_rows + 1 {
            return Err(crate::BackendError::InvalidInput(
                "Invalid CSR indptr array length".to_string(),
            ));
        }

        if data.len() != indices.len() {
            return Err(crate::BackendError::InvalidInput(
                "CSR data and indices arrays must have same length".to_string(),
            ));
        }

        if num_cols != other_shape[0] {
            return Err(crate::BackendError::InvalidInput(
                format!("Sparse matrix columns ({}) must match dense matrix rows ({})", num_cols, other_shape[0])
            ));
        }

        // Initialize result matrix (num_rows × other_cols)
        let mut result = vec![T::zero(); num_rows * other_cols];

        // Perform SPMM: result[i,j] += data[k] * other[indices[k], j] for each row i
        for row in 0..num_rows {
            let row_start = indptr[row];
            let row_end = indptr[row + 1];

            for k in row_start..row_end {
                let col = indices[k];
                let val = data[k];

                // Multiply this sparse entry with each column of the dense matrix
                for j in 0..other_cols {
                    let dense_idx = col * other_cols + j;
                    let result_idx = row * other_cols + j;

                    if dense_idx < other_data.len() && result_idx < result.len() {
                        result[result_idx] = result[result_idx] + val * other_data[dense_idx];
                    }
                }
            }
        }

        Ok(result)
    }

    fn spmv_csr(
        &self,
        data: &[Self::Data],
        indices: &[usize],
        indptr: &[usize],
        vector: &[Self::Data],
        num_rows: usize,
        _num_cols: usize,
    ) -> crate::Result<Vec<Self::Data>> {
        let mut result = vec![Self::Data::default(); num_rows];

        for row in 0..num_rows {
            let row_start = indptr[row];
            let row_end = indptr[row + 1];

            for idx in row_start..row_end {
                let col = indices[idx];
                let val = data[idx];
                // result[row] += val * vector[col]
                // We need to implement multiplication, but for now assume it's f32/f64
                // This is a simplified implementation
                result[row] = result[row] + val * vector[col];
            }
        }

        Ok(result)
    }

    fn coo_matmul_sparse(
        &self,
        _lhs_data: &[Self::Data],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs_data: &[Self::Data],
        _rhs_row: &[usize],
        _rhs_col: &[usize],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> crate::Result<coeus_storage::CooStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "coo_matmul_sparse".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn coo_matmul_dense(
        &self,
        _lhs_data: &[Self::Data],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs: &coeus_storage::DenseStorage<Self::Data>,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> crate::Result<coeus_storage::DenseStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "coo_matmul_dense".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn coo_add_sparse(
        &self,
        _lhs_data: &[Self::Data],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs_data: &[Self::Data],
        _rhs_row: &[usize],
        _rhs_col: &[usize],
        _m: usize,
        _n: usize,
    ) -> crate::Result<coeus_storage::CooStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "coo_add_sparse".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn coo_mul_sparse(
        &self,
        _lhs_data: &[Self::Data],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs_data: &[Self::Data],
        _rhs_row: &[usize],
        _rhs_col: &[usize],
        _m: usize,
        _n: usize,
    ) -> crate::Result<coeus_storage::CooStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "coo_mul_sparse".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn quantize(
        &self,
        input: &coeus_storage::DenseStorage<Self::Data>,
        levels: usize,
    ) -> crate::Result<coeus_storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
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

    // Missing required methods that I'm implementing with errors for now
    fn relu_dense(
        &self,
        input: &coeus_storage::DenseStorage<Self::Data>,
    ) -> crate::Result<coeus_storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd + Default,
    {
        // Element-wise ReLU: max(0, x)
        let input_data = input.as_slice();
        let mut result_data = Vec::with_capacity(input_data.len());

        let zero = Self::Data::zero();
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
        input: &coeus_storage::DenseStorage<Self::Data>,
    ) -> crate::Result<Self::Data>
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
        input: &coeus_storage::DenseStorage<Self::Data>,
    ) -> crate::Result<Self::Data>
    where
        Self::Data: PartialOrd,
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
        input: &coeus_storage::DenseStorage<Self::Data>,
    ) -> crate::Result<Self::Data>
    where
        Self::Data: PartialOrd,
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

    fn argmax_dense(&self, input: &coeus_storage::DenseStorage<Self::Data>) -> crate::Result<usize>
    where
        Self::Data: PartialOrd,
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

    fn argmin_dense(&self, input: &coeus_storage::DenseStorage<Self::Data>) -> crate::Result<usize>
    where
        Self::Data: PartialOrd,
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
        input: &coeus_storage::DenseStorage<Self::Data>,
    ) -> crate::Result<coeus_storage::DenseStorage<Self::Data>>
    where
        T: crate::DataType,
    {
        // Element-wise natural logarithm
        let input_data = input.as_slice();
        let mut result_data = Vec::with_capacity(input_data.len());

        for &x in input_data {
            // We need to convert to f64, compute log, then convert back
            let x_f64 = x.to_f64().unwrap_or(1.0);
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
        input: &coeus_storage::DenseStorage<Self::Data>,
    ) -> crate::Result<coeus_storage::DenseStorage<Self::Data>>
    where
        T: crate::DataType,
    {
        // Element-wise sine function
        let input_data = input.as_slice();
        let mut result_data = Vec::with_capacity(input_data.len());

        for &x in input_data {
            // Convert to f64, compute sin, then convert back
            let x_f64 = x.to_f64().unwrap_or(0.0);
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
        input: &coeus_storage::DenseStorage<Self::Data>,
    ) -> crate::Result<coeus_storage::DenseStorage<Self::Data>>
    where
        T: crate::DataType,
    {
        // Element-wise cosine function
        let input_data = input.as_slice();
        let mut result_data = Vec::with_capacity(input_data.len());

        for &x in input_data {
            // Convert to f64, compute cos, then convert back
            let x_f64 = x.to_f64().unwrap_or(0.0);
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
