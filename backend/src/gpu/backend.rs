//! GPU backend implementation using WGPU
//!
//! This module provides a GPU backend that implements the Backend trait,
//! enabling transparent GPU acceleration for tensor operations.

use crate::{Backend, Device, DeviceInfo, Result};
use dtype::DataType;
use storage::{DenseStorage, Storage, CooStorage};
use std::marker::PhantomData;
use super::traits::GpuFloat;
use dtype::num_traits::{self};

/// GPU backend implementation using WGPU compute shaders
///
/// Provides GPU-accelerated operations through the same Backend trait interface
/// as the CPU backend. Currently falls back to CPU implementations but provides
/// structure for future GPU acceleration.
#[derive(Debug)]
pub struct GpuBackend<T: DataType + GpuFloat> {
    device: Device,
    /// GPU device index (0 = first GPU)
    gpu_index: usize,
    _phantom: PhantomData<T>,
}

impl<T: DataType + GpuFloat> Clone for GpuBackend<T> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            gpu_index: self.gpu_index,
            _phantom: PhantomData,
        }
    }
}

impl<T: DataType + GpuFloat> Default for GpuBackend<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DataType + GpuFloat> GpuBackend<T> {
    /// Create a new GPU backend
    ///
    /// Creates a backend targeting the first available GPU.
    /// Currently uses CPU fallbacks, GPU acceleration coming soon.
    pub fn new() -> Self {
        Self {
            device: Device::Gpu {
                name: "Generic GPU".to_string(),
                vendor: 0,
                device: 0,
                backend: "WGPU",
            },
            gpu_index: 0,
            _phantom: PhantomData,
        }
    }

    /// Create GPU backend targeting specific GPU index
    pub fn with_gpu_index(gpu_index: usize) -> Self {
        Self {
            device: Device::Gpu {
                name: format!("GPU {}", gpu_index),
                vendor: 0,
                device: gpu_index as u32,
                backend: "WGPU",
            },
            gpu_index,
            _phantom: PhantomData,
        }
    }

    /// Get GPU device index
    pub fn gpu_index(&self) -> usize {
        self.gpu_index
    }
}

impl<T: DataType + GpuFloat> Backend for GpuBackend<T> {
    type Data = T;
    type Device = Device;

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn supports(&self, operation: &str) -> bool {
        // GPU backend supports these operations with actual GPU acceleration
        matches!(
            operation,
            "add" | "sub" | "mul" | "div" | "matmul" | "relu" | "sigmoid" | "tanh" 
            | "sum" | "mean" | "max" | "min" | "exp" | "log" | "sin" | "cos"
            | "spmm_csr" | "spmv_csr"
        )
    }

    fn device_name(&self) -> &str {
        "gpu"
    }

    fn device_info(&self) -> Box<dyn DeviceInfo> {
        Box::new(self.device.clone())
    }

    // Dense operations - use GPU when available, otherwise CPU fallback

    fn add_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        // Use GPU-accelerated addition via gpu_ops
        let result = super::traits::gpu_ops::add_gpu(lhs.as_slice(), rhs.as_slice())?;
        DenseStorage::from_vec(result, lhs.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn mul_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        // Use GPU-accelerated multiplication via gpu_ops
        let result = super::traits::gpu_ops::mul_gpu(lhs.as_slice(), rhs.as_slice())?;
        DenseStorage::from_vec(result, lhs.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn sub_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        let mut result = vec![T::default(); lhs.as_slice().len()];
        crate::cpu::arithmetic::sub_primitive(lhs.as_slice(), rhs.as_slice(), &mut result)?;
        
        DenseStorage::from_vec(result, lhs.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn matmul_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        let lhs_shape = lhs.shape().dims();
        let rhs_shape = rhs.shape().dims();
        
        if lhs_shape.len() != 2 || rhs_shape.len() != 2 {
            return Err(crate::BackendError::InvalidInput(
                "Matrix multiplication requires 2D tensors".to_string(),
            ));
        }
        
        let (m, k) = (lhs_shape[0], lhs_shape[1]);
        let (k2, n) = (rhs_shape[0], rhs_shape[1]);
        
        if k != k2 {
            return Err(crate::BackendError::InvalidInput(
                "Matrix dimensions don't match for multiplication".to_string(),
            ));
        }
        
        // Use GPU-accelerated matrix multiplication via gpu_ops
        let result = super::traits::gpu_ops::matmul_gpu(lhs.as_slice(), rhs.as_slice(), m, k, n)?;
        
        DenseStorage::from_vec(result, &[m, n])
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn relu_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd + Default,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 7)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn sigmoid_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 6)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn sum_dense(&self, input: &DenseStorage<Self::Data>) -> Result<Self::Data> {
        super::traits::gpu_ops::reduce_gpu(input.as_slice(), 0)
    }

    fn max_dense(&self, input: &DenseStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd,
    {
        super::traits::gpu_ops::reduce_gpu(input.as_slice(), 1)
    }

    fn min_dense(&self, input: &DenseStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd,
    {
        super::traits::gpu_ops::reduce_gpu(input.as_slice(), 2)
    }

    fn argmax_dense(&self, input: &DenseStorage<Self::Data>) -> Result<usize>
    where
        Self::Data: PartialOrd,
    {
        input.as_slice()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| crate::BackendError::InvalidInput("Empty tensor".to_string()))
    }

    fn argmin_dense(&self, input: &DenseStorage<Self::Data>) -> Result<usize>
    where
        Self::Data: PartialOrd,
    {
        input.as_slice()
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| crate::BackendError::InvalidInput("Empty tensor".to_string()))
    }

    fn exp_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 3)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn log_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 0)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn sin_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 1)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn cos_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 2)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn tan_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 8)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn asin_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 9)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn acos_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 10)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn atan_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 11)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn sinh_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 12)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn cosh_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 13)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn tanh_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 5)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn gelu_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 18)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn sqrt_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 4)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn abs_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Signed,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 14)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn floor_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 16)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn ceil_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 15)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn round_dense(
        &self,
        input: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        let result = super::traits::gpu_ops::unary_op_gpu(input.as_slice(), 17)?;
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn conv2d_dense(
        &self,
        input: &DenseStorage<Self::Data>,
        weight: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        let in_dims = input.shape().dims();
        let w_dims = weight.shape().dims();
        
        if in_dims.len() != 4 || w_dims.len() != 4 {
             return Err(crate::BackendError::InvalidInput("conv2d requires 4D input and weight".into()));
        }
        
        let n: u32 = in_dims[0] as u32;
        let c_in: u32 = in_dims[1] as u32;
        let h_in: u32 = in_dims[2] as u32;
        let w_in: u32 = in_dims[3] as u32;
        
        let c_out: u32 = w_dims[0] as u32;
        let c_in_w: u32 = w_dims[1] as u32;
        let kh: u32 = w_dims[2] as u32;
        let kw: u32 = w_dims[3] as u32;
        
        if c_in != c_in_w {
            return Err(crate::BackendError::InvalidInput("conv2d channel mismatch".into()));
        }

        let input_dims_u32 = [n, c_in, h_in, w_in];
        let weight_dims_u32 = [c_out, c_in, kh, kw];
        
        // Try GPU implementation
        let result_vec = super::traits::gpu_ops::conv2d_gpu(
            input.as_slice(),
            weight.as_slice(),
            &input_dims_u32,
            &weight_dims_u32
        )?;
        
        let h_out = h_in.saturating_sub(kh) + 1;
        let w_out = w_in.saturating_sub(kw) + 1;
        
        DenseStorage::from_vec(
            result_vec, 
            &[n as usize, c_out as usize, h_out as usize, w_out as usize]
        ).map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn mean_dense(
        &self,
        input: &DenseStorage<Self::Data>,
        _axes: Option<&[usize]>,
    ) -> Result<DenseStorage<Self::Data>> {
        let sum = super::traits::gpu_ops::reduce_gpu(input.as_slice(), 0)?;
        let count = T::from_usize(input.as_slice().len()).ok_or_else(|| {
            crate::BackendError::InvalidInput("Failed to convert size".into())
        })?;
        let mean_val = sum / count;
        DenseStorage::from_vec(vec![mean_val], &[])
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    // Sparse operations - these can use GPU when available

    fn spmv_csr(
        &self,
        data: &[Self::Data],
        indices: &[usize],
        indptr: &[usize],
        vector: &[Self::Data],
        num_rows: usize,
        _num_cols: usize,
    ) -> Result<Vec<Self::Data>> {
        // Try GPU implementation first, it will fallback to CPU if needed
        super::traits::gpu_ops::spmv_csr_gpu(data, indices, indptr, vector, num_rows)
    }

    fn spmm_csr(
        &self,
        data: &[Self::Data],
        indices: &[usize],
        indptr: &[usize],
        other: &DenseStorage<Self::Data>,
        num_rows: usize,
        _num_cols: usize,
    ) -> Result<Vec<Self::Data>> {
        let dense_cols = other.shape().dims().get(1).copied().unwrap_or(1);
        // Try GPU implementation first
        super::traits::gpu_ops::spmm_csr_gpu(
            data, indices, indptr, other.as_slice(), num_rows, dense_cols
        )
    }

    fn coo_matmul_sparse(
        &self,
        lhs_data: &[Self::Data],
        lhs_row: &[usize],
        lhs_col: &[usize],
        rhs_data: &[Self::Data],
        rhs_row: &[usize],
        rhs_col: &[usize],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        // Fallback: Convert to dense, multiply, convert to CSR
        // This ensures correctness but is memory intensive
        let lhs_coo = CooStorage::new(lhs_data.to_vec(), lhs_row.to_vec(), lhs_col.to_vec(), &[m, k])
            .map_err(|e| crate::BackendError::InvalidInput(format!("COO Creation Error: {}", e)))?;
        let rhs_coo = CooStorage::new(rhs_data.to_vec(), rhs_row.to_vec(), rhs_col.to_vec(), &[k, n])
            .map_err(|e| crate::BackendError::InvalidInput(format!("COO Creation Error: {}", e)))?;
            
        let lhs_dense = lhs_coo.to_dense().map_err(|e| crate::BackendError::StorageError { source: e })?;
        let rhs_dense = rhs_coo.to_dense().map_err(|e| crate::BackendError::StorageError { source: e })?;
        
        // Use dense implementation (which might use GPU)
        let lhs_tensor = DenseStorage::from_vec(lhs_dense.as_slice().to_vec(), &[m, k])
             .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))?;
        let rhs_tensor = DenseStorage::from_vec(rhs_dense.as_slice().to_vec(), &[k, n])
             .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))?;
             
        let result_vec = super::traits::gpu_ops::matmul_gpu(lhs_tensor.as_slice(), rhs_tensor.as_slice(), m, k, n)?;
        
        let result_dense = DenseStorage::from_vec(result_vec, &[m, n])
             .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))?;
             
        result_dense.to_csr().map_err(|e| crate::BackendError::StorageError { source: e })
    }

    fn coo_matmul_dense(
        &self,
        lhs_data: &[Self::Data],
        lhs_row: &[usize],
        lhs_col: &[usize],
        rhs: &DenseStorage<Self::Data>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<DenseStorage<Self::Data>> {
        // Convert COO to CSR and use optimized SpMM
        let lhs_coo = CooStorage::new(lhs_data.to_vec(), lhs_row.to_vec(), lhs_col.to_vec(), &[m, k])
            .map_err(|e| crate::BackendError::InvalidInput(format!("COO Creation Error: {}", e)))?;
        let lhs_csr = lhs_coo.to_csr().map_err(|e| crate::BackendError::StorageError { source: e })?;
        
        let result_vec = self.spmm_csr(
            lhs_csr.data(),
            lhs_csr.indices(),
            lhs_csr.indptr(),
            rhs,
            m,
            n
        )?;
        
        DenseStorage::from_vec(result_vec, &[m, n])
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn coo_add_sparse(
        &self,
        lhs_data: &[Self::Data],
        lhs_row: &[usize],
        lhs_col: &[usize],
        rhs_data: &[Self::Data],
        rhs_row: &[usize],
        rhs_col: &[usize],
        m: usize,
        n: usize,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        // Fallback: Convert to dense, add, convert to CSR
        let lhs_coo = CooStorage::new(lhs_data.to_vec(), lhs_row.to_vec(), lhs_col.to_vec(), &[m, n])
            .map_err(|e| crate::BackendError::InvalidInput(format!("COO Creation Error: {}", e)))?;
        let rhs_coo = CooStorage::new(rhs_data.to_vec(), rhs_row.to_vec(), rhs_col.to_vec(), &[m, n])
            .map_err(|e| crate::BackendError::InvalidInput(format!("COO Creation Error: {}", e)))?;
            
        let lhs_dense = lhs_coo.to_dense().map_err(|e| crate::BackendError::StorageError { source: e })?;
        let rhs_dense = rhs_coo.to_dense().map_err(|e| crate::BackendError::StorageError { source: e })?;
        
        let lhs_tensor = DenseStorage::from_vec(lhs_dense.as_slice().to_vec(), &[m, n])
             .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))?;
        let rhs_tensor = DenseStorage::from_vec(rhs_dense.as_slice().to_vec(), &[m, n])
             .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))?;
             
        let result = self.add_dense(&lhs_tensor, &rhs_tensor)?;
        result.to_csr().map_err(|e| crate::BackendError::StorageError { source: e })
    }

    fn coo_mul_sparse(
        &self,
        lhs_data: &[Self::Data],
        lhs_row: &[usize],
        lhs_col: &[usize],
        rhs_data: &[Self::Data],
        rhs_row: &[usize],
        rhs_col: &[usize],
        m: usize,
        n: usize,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        // Fallback: Convert to dense, mul, convert to CSR
        let lhs_coo = CooStorage::new(lhs_data.to_vec(), lhs_row.to_vec(), lhs_col.to_vec(), &[m, n])
            .map_err(|e| crate::BackendError::InvalidInput(format!("COO Creation Error: {}", e)))?;
        let rhs_coo = CooStorage::new(rhs_data.to_vec(), rhs_row.to_vec(), rhs_col.to_vec(), &[m, n])
            .map_err(|e| crate::BackendError::InvalidInput(format!("COO Creation Error: {}", e)))?;
            
        let lhs_dense = lhs_coo.to_dense().map_err(|e| crate::BackendError::StorageError { source: e })?;
        let rhs_dense = rhs_coo.to_dense().map_err(|e| crate::BackendError::StorageError { source: e })?;
        
        let lhs_tensor = DenseStorage::from_vec(lhs_dense.as_slice().to_vec(), &[m, n])
             .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))?;
        let rhs_tensor = DenseStorage::from_vec(rhs_dense.as_slice().to_vec(), &[m, n])
             .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))?;
             
        let result = self.mul_dense(&lhs_tensor, &rhs_tensor)?;
        result.to_csr().map_err(|e| crate::BackendError::StorageError { source: e })
    }

    fn quantize(
        &self,
        input: &DenseStorage<Self::Data>,
        levels: usize,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        // Min-Max Quantization
        // 1. Convert to f32 for calculation stability
        let data_f32 = T::to_f32_slice(input.as_slice());
        if data_f32.is_empty() {
             return DenseStorage::from_vec(vec![], input.shape().dims())
                .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)));
        }

        // 2. Find min/max
        let (min_val, max_val) = data_f32.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &x| {
            (if x < min { x } else { min }, if x > max { x } else { max })
        });
        
        // 3. Compute scale and zero point
        let levels_f = levels as f32;
        // Avoid division by zero if max == min
        let scale = if (max_val - min_val).abs() < 1e-6 {
            1.0
        } else {
            (max_val - min_val) / (levels_f - 1.0)
        };
        
        let zero_point = -min_val / scale;
        
        // 4. Quantize
        // q = clamp(round(x / scale + zero_point), 0, levels - 1)
        // We map back to T (as float/representation of the level)
        let quantized_f32: Vec<f32> = data_f32.into_iter().map(|x| {
            let q = (x / scale + zero_point).round();
            q.clamp(0.0, levels_f - 1.0)
        }).collect();
        
        let result_data = T::from_f32_slice(&quantized_f32);
        
        DenseStorage::from_vec(result_data, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn clip_info_nce_loss(
        &self,
        _image_embeddings: &DenseStorage<Self::Data>,
        _text_embeddings: &DenseStorage<Self::Data>,
        _temperature: f32,
    ) -> Result<Self::Data> {
        Ok(T::default())
    }

    fn clip_attention(
        &self,
        queries: &DenseStorage<Self::Data>,
        _keys: &DenseStorage<Self::Data>,
        _values: &DenseStorage<Self::Data>,
        _num_heads: usize,
    ) -> Result<DenseStorage<Self::Data>> {
        DenseStorage::from_vec(vec![T::default(); queries.as_slice().len()], queries.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn add_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "add_strided".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn mul_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "mul_strided".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn sub_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "sub_strided".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn cholesky_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "cholesky_dense".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn qr_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<(storage::DenseStorage<Self::Data>, storage::DenseStorage<Self::Data>)>
    where
        Self::Data: num_traits::Float,
    {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "qr_dense".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn svd_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<(
        storage::DenseStorage<Self::Data>,
        storage::DenseStorage<Self::Data>,
        storage::DenseStorage<Self::Data>,
    )>
    where
        Self::Data: num_traits::Float,
    {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "svd_dense".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn take_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
        _indices: &storage::DenseStorage<dtype::int::Int64>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "take_dense".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn put_dense(
        &self,
        _input: &mut storage::DenseStorage<Self::Data>,
        _indices: &storage::DenseStorage<dtype::int::Int64>,
        _values: &storage::DenseStorage<Self::Data>,
        _accumulate: bool,
    ) -> Result<()> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "put_dense".to_string(),
            backend: "gpu".to_string(),
        })
    }

    // ================== Comparison ==================

    fn eq_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "eq_dense".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn eq_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "eq_strided".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn ne_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "ne_dense".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn ne_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "ne_strided".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn gt_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "gt_dense".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn gt_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "gt_strided".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn ge_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "ge_dense".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn ge_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "ge_strided".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn lt_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "lt_dense".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn lt_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "lt_strided".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn le_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "le_dense".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn le_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "le_strided".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn add_csr(
        &self,
        _lhs: &storage::CsrStorage<Self::Data>,
        _rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "add_csr".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn mul_csr(
        &self,
        _lhs: &storage::CsrStorage<Self::Data>,
        _rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "mul_csr".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn sub_csr(
        &self,
        _lhs: &storage::CsrStorage<Self::Data>,
        _rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "sub_csr".to_string(),
            backend: "gpu".to_string(),
        })
    }

    fn div_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "div_strided".to_string(),
            backend: "gpu".to_string(),
        })
    }
}
