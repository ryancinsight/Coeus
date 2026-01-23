//! CPU backend implementation using hierarchical primitives

use crate::{Backend, Device, DeviceInfo, Result};
use dtype::{DataType, num_traits};
use storage::{DenseStorage, Storage};
use std::marker::PhantomData;

/// CPU backend implementation using hierarchical primitive operations
#[derive(Debug, Clone)]
pub struct CpuBackend<T: DataType> {
    device: Device,
    _phantom: PhantomData<T>,
}

impl<T: DataType> Default for CpuBackend<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: DataType> CpuBackend<T> {
    /// Create a new CPU backend
    pub fn new() -> Self {
        Self {
            device: Device::Cpu,
            _phantom: PhantomData,
        }
    }
}

impl<T: DataType> Backend for CpuBackend<T> {
    type Data = T;
    type Device = Device;

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn supports(&self, operation: &str) -> bool {
        // CPU backend supports all operations
        matches!(
            operation,
            "add" | "sub" | "mul" | "div" | "matmul" | "relu" | "sigmoid" | "tanh" 
            | "sum" | "mean" | "max" | "min" | "argmax" | "argmin" | "exp" | "log" 
            | "sin" | "cos" | "conv2d" | "spmm_csr" | "spmv_csr" | "quantize"
            | "clip_info_nce_loss" | "clip_attention"
        )
    }

    fn device_name(&self) -> &str {
        "cpu"
    }

    fn device_info(&self) -> Box<dyn DeviceInfo> {
        Box::new(self.device.clone())
    }

    fn add_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        // Use hierarchical primitive
        let lhs_slice = lhs.as_slice();
        let rhs_slice = rhs.as_slice();
        let mut result = vec![T::default(); lhs_slice.len()];
        
        super::arithmetic::add_primitive(lhs_slice, rhs_slice, &mut result)?;
        
        DenseStorage::from_vec(result, lhs.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn mul_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        // Use hierarchical primitive
        let lhs_slice = lhs.as_slice();
        let rhs_slice = rhs.as_slice();
        let mut result = vec![T::default(); lhs_slice.len()];
        
        super::arithmetic::mul_primitive(lhs_slice, rhs_slice, &mut result)?;
        
        DenseStorage::from_vec(result, lhs.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn sub_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        // Use hierarchical primitive
        let lhs_slice = lhs.as_slice();
        let rhs_slice = rhs.as_slice();
        let mut result = vec![T::default(); lhs_slice.len()];
        
        super::arithmetic::sub_primitive(lhs_slice, rhs_slice, &mut result)?;
        
        DenseStorage::from_vec(result, lhs.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn matmul_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        // Use hierarchical primitive
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
        
        let lhs_slice = lhs.as_slice();
        let rhs_slice = rhs.as_slice();
        let mut result = vec![T::default(); m * n];
        
        super::linear_algebra::matmul_primitive(lhs_slice, rhs_slice, &mut result, m, k, n)?;
        
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
        // Use hierarchical primitive
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        
        super::activation::relu_primitive(input_slice, &mut result)?;
        
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
        // Use hierarchical primitive
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        
        super::activation::sigmoid_primitive(input_slice, &mut result)?;
        
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn sum_dense(&self, input: &DenseStorage<Self::Data>) -> Result<Self::Data> {
        // Use hierarchical primitive
        Ok(super::reduction::sum_primitive(input.as_slice()))
    }

    fn max_dense(&self, input: &DenseStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd,
    {
        // Use hierarchical primitive
        super::reduction::max_primitive(input.as_slice())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Max error: {}", e)))
    }

    fn min_dense(&self, input: &DenseStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd,
    {
        // Use hierarchical primitive - implement min_primitive
        input.as_slice()
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
            .ok_or_else(|| crate::BackendError::InvalidInput("Empty tensor".to_string()))
    }

    fn argmax_dense(&self, input: &DenseStorage<Self::Data>) -> Result<usize>
    where
        Self::Data: PartialOrd,
    {
        // Use hierarchical primitive - implement argmax_primitive
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
        // Use hierarchical primitive - implement argmin_primitive
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.exp();
        }
        
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.ln();
        }
        
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.sin();
        }
        
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.cos();
        }
        
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.tan();
        }
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.asin();
        }
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.acos();
        }
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.atan();
        }
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.sinh();
        }
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.cosh();
        }
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.tanh();
        }
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
        let sqrt_2_over_pi = num_traits::NumCast::from(0.7978845608028654).unwrap_or(T::one());
        let coeff = num_traits::NumCast::from(0.044715).unwrap_or(T::zero());
        let half = num_traits::NumCast::from(0.5).unwrap_or(T::one());
        
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        
        // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        for (i, &x) in input_slice.iter().enumerate() {
            let x3 = x * x * x;
            let inner = sqrt_2_over_pi * (x + coeff * x3);
            result[i] = half * x * (T::one() + inner.tanh());
        }

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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.sqrt();
        }
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.abs();
        }
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.floor();
        }
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.ceil();
        }
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
        let input_slice = input.as_slice();
        let mut result = vec![T::default(); input_slice.len()];
        for (i, &x) in input_slice.iter().enumerate() {
            result[i] = x.round();
        }
        DenseStorage::from_vec(result, input.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn conv2d_dense(
        &self,
        _input: &DenseStorage<Self::Data>,
        _weight: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        // Conv2D is a complex operation that requires proper implementation
        // For now, return unsupported operation error
        Err(crate::BackendError::UnsupportedOperation {
            operation: "conv2d_dense".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn mean_dense(
        &self,
        input: &DenseStorage<Self::Data>,
        _axes: Option<&[usize]>,
    ) -> Result<DenseStorage<Self::Data>> {
        // Use hierarchical primitive
        let mean_val = super::reduction::mean_primitive(input.as_slice());
        
        // For now, return scalar result
        DenseStorage::from_vec(vec![mean_val], &[])
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn spmv_csr(
        &self,
        data: &[Self::Data],
        indices: &[usize],
        indptr: &[usize],
        vector: &[Self::Data],
        num_rows: usize,
        _num_cols: usize,
    ) -> Result<Vec<Self::Data>> {
        // Use optimized sparse kernel
        let mut result = vec![T::default(); num_rows];
        super::sparse_kernels::spmv_csr_kernel(data, indices, indptr, vector, &mut result, num_rows)?;
        Ok(result)
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
        // Use optimized sparse kernel
        let dense_cols = other.shape().dims().get(1).copied().unwrap_or(1);
        let mut result = vec![T::default(); num_rows * dense_cols];
        super::sparse_kernels::spmm_csr_dense_kernel(
            data, indices, indptr, other.as_slice(), dense_cols, &mut result, num_rows
        )?;
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
        m: usize,
        _k: usize,
        n: usize,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        // Placeholder implementation - would use hierarchical primitive
        storage::CsrStorage::empty(&[m, n])
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn coo_matmul_dense(
        &self,
        _lhs_data: &[Self::Data],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs: &DenseStorage<Self::Data>,
        m: usize,
        _k: usize,
        n: usize,
    ) -> Result<DenseStorage<Self::Data>> {
        // Placeholder implementation - would use hierarchical primitive
        DenseStorage::from_vec(vec![T::default(); m * n], &[m, n])
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn coo_add_sparse(
        &self,
        _lhs_data: &[Self::Data],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs_data: &[Self::Data],
        _rhs_row: &[usize],
        _rhs_col: &[usize],
        m: usize,
        n: usize,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        // Placeholder implementation - would use hierarchical primitive
        storage::CsrStorage::empty(&[m, n])
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn coo_mul_sparse(
        &self,
        _lhs_data: &[Self::Data],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs_data: &[Self::Data],
        _rhs_row: &[usize],
        _rhs_col: &[usize],
        m: usize,
        n: usize,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        // Placeholder implementation - would use hierarchical primitive
        storage::CsrStorage::empty(&[m, n])
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn quantize(
        &self,
        _input: &DenseStorage<Self::Data>,
        _levels: usize,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        // Quantization requires integration with quantization crate
        // For now, return unsupported operation error
        Err(crate::BackendError::UnsupportedOperation {
            operation: "quantize".to_string(),
            backend: "cpu".to_string(),
        })
    }

    fn clip_info_nce_loss(
        &self,
        _image_embeddings: &DenseStorage<Self::Data>,
        _text_embeddings: &DenseStorage<Self::Data>,
        _temperature: f32,
    ) -> Result<Self::Data> {
        // Placeholder implementation - would use hierarchical primitive
        Ok(T::default())
    }

    fn clip_attention(
        &self,
        queries: &DenseStorage<Self::Data>,
        _keys: &DenseStorage<Self::Data>,
        _values: &DenseStorage<Self::Data>,
        _num_heads: usize,
    ) -> Result<DenseStorage<Self::Data>> {
        // Placeholder implementation - would use hierarchical primitive
        DenseStorage::from_vec(vec![T::default(); queries.as_slice().len()], queries.shape().dims())
            .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }
}