//! GPU type traits for zero-cost GPU dispatch
//!
//! Provides marker traits and type conversions for GPU-accelerated operations.
//! Enables specialization for f32 while falling back to CPU for other types.

use crate::Result;
use dtype::float::{Float32, Float64};

/// Trait for types that can be executed on GPU
///
/// This trait provides the mechanism for zero-cost GPU dispatch.
/// Types implementing this trait can leverage GPU acceleration when available.
pub trait GpuFloat: Copy + Default + Send + Sync + 'static {
    /// Convert slice to f32 for GPU execution
    fn to_f32_slice(slice: &[Self]) -> Vec<f32>;
    
    /// Convert f32 result back to Self
    fn from_f32_slice(slice: &[f32]) -> Vec<Self>;
    
    /// Returns true if this type natively matches GPU execution type
    fn is_native_gpu_type() -> bool;

    /// Convert f32 to Self (for scalar return)
    fn from_f32(val: f32) -> Option<Self>;

    /// Convert usize to Self (for counts)
    fn from_usize(val: usize) -> Option<Self>;
}

impl GpuFloat for f32 {
    #[inline]
    fn to_f32_slice(slice: &[Self]) -> Vec<f32> {
        slice.to_vec()
    }
    
    #[inline]
    fn from_f32_slice(slice: &[f32]) -> Vec<Self> {
        slice.to_vec()
    }
    
    #[inline]
    fn is_native_gpu_type() -> bool {
        true
    }

    #[inline]
    fn from_f32(val: f32) -> Option<Self> {
        Some(val)
    }

    #[inline]
    fn from_usize(val: usize) -> Option<Self> {
        Some(val as f32)
    }
}

impl GpuFloat for Float32 {
    #[inline]
    fn to_f32_slice(slice: &[Self]) -> Vec<f32> {
        slice.iter().map(|x| x.0).collect()
    }
    
    #[inline]
    fn from_f32_slice(slice: &[f32]) -> Vec<Self> {
        slice.iter().map(|&x| Float32(x)).collect()
    }
    
    #[inline]
    fn is_native_gpu_type() -> bool {
        true
    }

    #[inline]
    fn from_f32(val: f32) -> Option<Self> {
        Some(Float32(val))
    }

    #[inline]
    fn from_usize(val: usize) -> Option<Self> {
        Some(Float32(val as f32))
    }
}

impl GpuFloat for f64 {
    #[inline]
    fn to_f32_slice(slice: &[Self]) -> Vec<f32> {
        slice.iter().map(|&x| x as f32).collect()
    }
    
    #[inline]
    fn from_f32_slice(slice: &[f32]) -> Vec<Self> {
        slice.iter().map(|&x| x as f64).collect()
    }
    
    #[inline]
    fn is_native_gpu_type() -> bool {
        false // f64 needs conversion
    }

    #[inline]
    fn from_f32(val: f32) -> Option<Self> {
        Some(val as f64)
    }

    #[inline]
    fn from_usize(val: usize) -> Option<Self> {
        Some(val as f64)
    }
}

impl GpuFloat for Float64 {
    #[inline]
    fn to_f32_slice(slice: &[Self]) -> Vec<f32> {
        slice.iter().map(|x| x.0 as f32).collect()
    }
    
    #[inline]
    fn from_f32_slice(slice: &[f32]) -> Vec<Self> {
        slice.iter().map(|&x| Float64(x as f64)).collect()
    }
    
    #[inline]
    fn is_native_gpu_type() -> bool {
        false
    }

    #[inline]
    fn from_f32(val: f32) -> Option<Self> {
        Some(Float64(val as f64))
    }

    #[inline]
    fn from_usize(val: usize) -> Option<Self> {
        Some(Float64(val as f64))
    }
}

// Integer types - less common for GPU ML but supported via conversion
impl GpuFloat for i32 {
    #[inline]
    fn to_f32_slice(slice: &[Self]) -> Vec<f32> {
        slice.iter().map(|&x| x as f32).collect()
    }
    
    #[inline]
    fn from_f32_slice(slice: &[f32]) -> Vec<Self> {
        slice.iter().map(|&x| x as i32).collect()
    }
    
    #[inline]
    fn is_native_gpu_type() -> bool {
        false
    }

    #[inline]
    fn from_f32(val: f32) -> Option<Self> {
        Some(val as i32)
    }

    #[inline]
    fn from_usize(val: usize) -> Option<Self> {
        Some(val as i32)
    }
}

impl GpuFloat for i64 {
    #[inline]
    fn to_f32_slice(slice: &[Self]) -> Vec<f32> {
        slice.iter().map(|&x| x as f32).collect()
    }
    
    #[inline]
    fn from_f32_slice(slice: &[f32]) -> Vec<Self> {
        slice.iter().map(|&x| x as i64).collect()
    }
    
    #[inline]
    fn is_native_gpu_type() -> bool {
        false
    }

    #[inline]
    fn from_f32(val: f32) -> Option<Self> {
        Some(val as i64)
    }

    #[inline]
    fn from_usize(val: usize) -> Option<Self> {
        Some(val as i64)
    }
}

/// GPU execution helper functions
pub mod gpu_ops {
    use super::*;
    use crate::gpu::dense_executor::get_gpu_executor;
    
    /// Execute addition on GPU if available, otherwise fall back to CPU
    pub fn add_gpu<T: GpuFloat + dtype::DataType>(
        lhs: &[T],
        rhs: &[T],
    ) -> Result<Vec<T>> {
        // Check if GPU executor is available
        if let Some(executor) = get_gpu_executor() {
            // Convert to f32, execute on GPU, convert back
            let lhs_f32 = T::to_f32_slice(lhs);
            let rhs_f32 = T::to_f32_slice(rhs);
            let result_f32 = executor.add(&lhs_f32, &rhs_f32)?;
            Ok(T::from_f32_slice(&result_f32))
        } else {
            // Fall back to CPU
            let mut result = vec![T::default(); lhs.len()];
            crate::cpu::arithmetic::add_primitive(lhs, rhs, &mut result)?;
            Ok(result)
        }
    }
    
    /// Execute multiplication on GPU if available
    pub fn mul_gpu<T: GpuFloat + dtype::DataType>(
        lhs: &[T],
        rhs: &[T],
    ) -> Result<Vec<T>> {
        if let Some(executor) = get_gpu_executor() {
            let lhs_f32 = T::to_f32_slice(lhs);
            let rhs_f32 = T::to_f32_slice(rhs);
            let result_f32 = executor.mul(&lhs_f32, &rhs_f32)?;
            Ok(T::from_f32_slice(&result_f32))
        } else {
            let mut result = vec![T::default(); lhs.len()];
            crate::cpu::arithmetic::mul_primitive(lhs, rhs, &mut result)?;
            Ok(result)
        }
    }
    
    /// Execute matrix multiplication on GPU if available
    pub fn matmul_gpu<T: GpuFloat + dtype::DataType>(
        lhs: &[T],
        rhs: &[T],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<T>> {
        if let Some(executor) = get_gpu_executor() {
            let lhs_f32 = T::to_f32_slice(lhs);
            let rhs_f32 = T::to_f32_slice(rhs);
            let result_f32 = executor.matmul(&lhs_f32, &rhs_f32, m, k, n)?;
            Ok(T::from_f32_slice(&result_f32))
        } else {
            let mut result = vec![T::default(); m * n];
            crate::cpu::linear_algebra::matmul_primitive(lhs, rhs, &mut result, m, k, n)?;
            Ok(result)
        }
    }
    
    /// Execute sparse matrix-vector multiplication on GPU if available
    pub fn spmv_csr_gpu<T: GpuFloat + dtype::DataType>(
        values: &[T],
        indices: &[usize],
        indptr: &[usize],
        vector: &[T],
        num_rows: usize,
    ) -> Result<Vec<T>> {
        if let Some(executor) = get_gpu_executor() {
            // Convert types for GPU
            let values_f32 = T::to_f32_slice(values);
            let indices_u32: Vec<u32> = indices.iter().map(|&x| x as u32).collect();
            let indptr_u32: Vec<u32> = indptr.iter().map(|&x| x as u32).collect();
            let vector_f32 = T::to_f32_slice(vector);
            
            let result_f32 = executor.spmv_csr(&values_f32, &indices_u32, &indptr_u32, &vector_f32, num_rows)?;
            Ok(T::from_f32_slice(&result_f32))
        } else {
            let mut result = vec![T::default(); num_rows];
            crate::cpu::sparse_kernels::spmv_csr_kernel(values, indices, indptr, vector, &mut result, num_rows)?;
            Ok(result)
        }
    }

    /// Execute sparse-matrix dense-matrix multiplication on GPU if available
    pub fn spmm_csr_gpu<T: GpuFloat + dtype::DataType>(
        values: &[T],
        indices: &[usize],
        indptr: &[usize],
        matrix_b: &[T],
        m: usize, // Rows of A
        n: usize, // Cols of B
    ) -> Result<Vec<T>> {
        if let Some(executor) = get_gpu_executor() {
            let values_f32 = T::to_f32_slice(values);
            let indices_u32: Vec<u32> = indices.iter().map(|&x| x as u32).collect();
            let indptr_u32: Vec<u32> = indptr.iter().map(|&x| x as u32).collect();
            let matrix_b_f32 = T::to_f32_slice(matrix_b);
            
            let result_f32 = executor.spmm_csr(&values_f32, &indices_u32, &indptr_u32, &matrix_b_f32, m, n)?;
            Ok(T::from_f32_slice(&result_f32))
        } else {
            // CPU Fallback
            let mut result = vec![T::default(); m * n];
            crate::cpu::sparse_kernels::spmm_csr_dense_kernel(
                values, indices, indptr, matrix_b, n, &mut result, m
            )?;
            Ok(result)
        }
    }

    /// Execute 2D convolution on GPU if available
    pub fn conv2d_gpu<T: GpuFloat + dtype::DataType>(
        input: &[T],
        weight: &[T],
        input_dims: &[u32; 4],
        weight_dims: &[u32; 4],
    ) -> Result<Vec<T>> {
        if let Some(executor) = get_gpu_executor() {
            let input_f32 = T::to_f32_slice(input);
            let weight_f32 = T::to_f32_slice(weight);
            
            let result_f32 = executor.conv2d(&input_f32, &weight_f32, input_dims, weight_dims)?;
            Ok(T::from_f32_slice(&result_f32))
        } else {
            Err(crate::BackendError::UnsupportedOperation {
                operation: "conv2d".to_string(),
                backend: "gpu".to_string(),
            })
        }
    }

    /// Execute unary element-wise operation on GPU if available
    /// op: 0=log, 1=sin, 2=cos, 3=exp, 4=sqrt, 5=tanh, 6=sigmoid, 7=relu, 8=tan, 9=asin, 10=acos, 11=atan, 12=sinh, 13=cosh, 14=abs, 15=ceil, 16=floor, 17=round
    pub fn unary_op_gpu<T: GpuFloat + dtype::DataType>(
        input: &[T],
        op: u32,
    ) -> Result<Vec<T>> {
        if let Some(executor) = get_gpu_executor() {
            let input_f32 = T::to_f32_slice(input);
            let result_f32 = executor.unary_op(&input_f32, op)?;
            Ok(T::from_f32_slice(&result_f32))
        } else {
            // Should be handled by caller falling back to CPU, but returns error if GPU requested explicitly and fails
             Err(crate::BackendError::UnsupportedOperation {
                operation: format!("unary_op_{}", op),
                backend: "gpu".to_string(),
            })
        }
    }

    /// Execute reduction operation on GPU
    /// op: 0=Sum, 1=Max, 2=Min
    pub fn reduce_gpu<T: GpuFloat + dtype::DataType>(
        input: &[T],
        op: u32,
    ) -> Result<T> {
        if let Some(executor) = get_gpu_executor() {
             let input_f32 = T::to_f32_slice(input);
            let partials = executor.reduce(&input_f32, op, input.len())?;
            
            // Finish reduction on CPU
            let result_f32 = match op {
                0 => partials.iter().sum::<f32>(),
                1 => partials.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                2 => partials.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                _ => return Err(crate::BackendError::InvalidInput(format!("Invalid reduction op: {}", op))),
            };

            T::from_f32(result_f32)
                .ok_or_else(|| crate::BackendError::InvalidInput("Failed to convert result".into()))
        } else {
             Err(crate::BackendError::UnsupportedOperation {
                operation: format!("reduce_{}", op),
                backend: "gpu".to_string(),
            })
        }
    }
}
