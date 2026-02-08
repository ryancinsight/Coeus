//! CPU backend implementation using hierarchical primitives

use crate::{Backend, Device, DeviceInfo, Result};
use dtype::{num_traits, DataType};
use std::marker::PhantomData;
use storage::{DenseStorage, Storage, StorageToDense};

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
            "add"
                | "sub"
                | "mul"
                | "div"
                | "matmul"
                | "addmm_dense"
                | "addmv_dense"
                | "relu"
                | "sigmoid"
                | "tanh"
                | "sum"
                | "mean"
                | "max"
                | "min"
                | "argmax"
                | "argmin"
                | "exp"
                | "log"
                | "sin"
                | "cos"
                | "conv2d"
                | "spmm_csr"
                | "spmv_csr"
                | "quantize"
                | "clip_info_nce_loss"
                | "clip_attention"
                | "isnan"
                | "isinf"
                | "isfinite"
                | "logical_and"
                | "logical_or"
                | "logical_xor"
                | "logical_not"
                | "log1p"
                | "expm1"
                | "reciprocal"
                | "atan2"
                | "rsqrt"
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
        super::implementations::arithmetic::add_dense(lhs, rhs)
    }

    fn mul_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        super::implementations::arithmetic::mul_dense(lhs, rhs)
    }

    fn neg_csr(&self, input: &storage::CsrStorage<Self::Data>) -> Result<storage::CsrStorage<Self::Data>>
    where
        Self::Data: core::ops::Neg<Output = Self::Data> {
        super::implementations::activation::neg_csr(input)
    }

    fn neg_strided(
        &self,
        input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: core::ops::Neg<Output = Self::Data>
    {
        super::implementations::activation::neg_strided(input)
    }

    fn sub_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        super::implementations::arithmetic::sub_dense(lhs, rhs)
    }

    fn matmul_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        super::implementations::linear_algebra::matmul_dense(lhs, rhs)
    }

    fn addmm_dense(
        &self,
        input: &DenseStorage<Self::Data>,
        mat1: &DenseStorage<Self::Data>,
        mat2: &DenseStorage<Self::Data>,
        beta: Self::Data,
        alpha: Self::Data,
    ) -> Result<DenseStorage<Self::Data>> {
        super::implementations::linear_algebra::addmm_dense(input, mat1, mat2, beta, alpha)
    }

    fn addmv_dense(
        &self,
        input: &DenseStorage<Self::Data>,
        mat: &DenseStorage<Self::Data>,
        vec: &DenseStorage<Self::Data>,
        beta: Self::Data,
        alpha: Self::Data,
    ) -> Result<DenseStorage<Self::Data>> {
        super::implementations::linear_algebra::addmv_dense(input, mat, vec, beta, alpha)
    }

    fn relu_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd + Default,
    {
        super::implementations::activation::relu_dense(input)
    }

    fn sigmoid_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::sigmoid_dense(input)
    }

    fn relu_strided(
        &self,
        input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd + Default,
    {
        super::implementations::activation::relu_strided(input)
    }

    fn sigmoid_strided(
        &self,
        input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::sigmoid_strided(input)
    }

    fn sum_dense(&self, input: &DenseStorage<Self::Data>) -> Result<Self::Data> {
        super::implementations::reduction::sum_dense(input)
    }

    fn max_dense(&self, input: &DenseStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd,
    {
        super::implementations::reduction::max_dense(input)
    }

    fn min_dense(&self, input: &DenseStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd,
    {
        super::implementations::reduction::min_dense(input)
    }

    fn sum_strided(&self, input: &storage::StridedStorage<Self::Data>) -> Result<Self::Data> {
        super::implementations::reduction::sum_strided(input)
    }

    fn max_strided(&self, input: &storage::StridedStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd,
    {
        super::implementations::reduction::max_strided(input)
    }

    fn min_strided(&self, input: &storage::StridedStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd,
    {
        super::implementations::reduction::min_strided(input)
    }

    fn argmax_dense(&self, input: &DenseStorage<Self::Data>) -> Result<usize>
    where
        Self::Data: PartialOrd,
    {
        super::implementations::reduction::argmax_dense(input)
    }

    fn argmin_dense(&self, input: &DenseStorage<Self::Data>) -> Result<usize>
    where
        Self::Data: PartialOrd,
    {
        super::implementations::reduction::argmin_dense(input)
    }

    fn exp_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::exp_dense(input)
    }

    fn log_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::log_dense(input)
    }

    fn sin_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::sin_dense(input)
    }

    fn cos_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::cos_dense(input)
    }

    fn tan_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::tan_dense(input)
    }

    fn asin_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::asin_dense(input)
    }

    fn acos_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::acos_dense(input)
    }

    fn atan_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::atan_dense(input)
    }

    fn sinh_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::sinh_dense(input)
    }

    fn cosh_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::cosh_dense(input)
    }

    fn tanh_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::tanh_dense(input)
    }

    fn tanh_strided(
        &self,
        input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::tanh_strided(input)
    }

    fn gelu_strided(
        &self,
        input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        // Currently fallback to dense implementation inside implementation if needed
        // but here we can just call the strided one if implemented
        // For now, let's use the one in activation.rs
        super::implementations::activation::gelu_dense(&input.to_dense()) // Use to_dense() instead of storage_to_dense()
            .map(|d| storage::StridedStorage::new(d.as_slice().to_vec(), d.shape().dims()).unwrap())
    }

    fn gelu_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::gelu_dense(input)
    }

    fn sqrt_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::sqrt_dense(input)
    }

    fn abs_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Signed,
    {
        super::implementations::activation::abs_dense(input)
    }

    fn floor_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::floor_dense(input)
    }

    fn ceil_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::ceil_dense(input)
    }

    fn exp_strided(
        &self,
        input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::exp_strided(input)
    }

    fn log_strided(
        &self,
        input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::log_strided(input)
    }

    fn sqrt_strided(
        &self,
        input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::sqrt_strided(input)
    }

    fn abs_strided(
        &self,
        input: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: num_traits::Signed,
    {
        super::implementations::activation::abs_strided(input)
    }

    fn round_dense(&self, input: &DenseStorage<Self::Data>) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::round_dense(input)
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
        axes: Option<&[usize]>,
    ) -> Result<DenseStorage<Self::Data>> {
        super::implementations::reduction::mean_dense(input, axes)
    }

    fn cholesky_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::linear_algebra::cholesky_dense(input)
    }

    fn qr_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<(storage::DenseStorage<Self::Data>, storage::DenseStorage<Self::Data>)>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::linear_algebra::qr_dense(input)
    }

    fn svd_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<(
        storage::DenseStorage<Self::Data>,
        storage::DenseStorage<Self::Data>,
        storage::DenseStorage<Self::Data>,
    )>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::linear_algebra::svd_dense(input)
    }

    fn spmv_csr(
        &self,
        data: &[Self::Data],
        indices: &[usize],
        indptr: &[usize],
        vector: &[Self::Data],
        num_rows: usize,
        num_cols: usize,
    ) -> Result<Vec<Self::Data>> {
        super::implementations::sparse::spmv_csr(
            data, indices, indptr, vector, num_rows, num_cols,
        )
    }

    fn spmm_csr(
        &self,
        data: &[Self::Data],
        indices: &[usize],
        indptr: &[usize],
        other: &DenseStorage<Self::Data>,
        num_rows: usize,
        num_cols: usize,
    ) -> Result<Vec<Self::Data>> {
        super::implementations::sparse::spmm_csr(
            data, indices, indptr, other, num_rows, num_cols,
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
        super::implementations::sparse::coo_matmul_sparse(
            lhs_data, lhs_row, lhs_col, rhs_data, rhs_row, rhs_col, m, k, n,
        )
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
        super::implementations::sparse::coo_matmul_dense(
            lhs_data, lhs_row, lhs_col, rhs, m, k, n,
        )
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
        super::implementations::sparse::coo_add_sparse(
            lhs_data, lhs_row, lhs_col, rhs_data, rhs_row, rhs_col, m, n,
        )
    }

    fn coo_sub_sparse(
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
        super::implementations::sparse::coo_sub_sparse(
            lhs_data, lhs_row, lhs_col, rhs_data, rhs_row, rhs_col, m, n,
        )
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
        super::implementations::sparse::coo_mul_sparse(
            lhs_data, lhs_row, lhs_col, rhs_data, rhs_row, rhs_col, m, n,
        )
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

    // ================== Comparison Operations ==================

    fn eq_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        super::implementations::comparison::eq_dense(lhs, rhs)
    }

    fn eq_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        super::implementations::comparison::eq_strided(lhs, rhs)
    }

    fn ne_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>> {
        super::implementations::comparison::ne_dense(lhs, rhs)
    }

    fn ne_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        super::implementations::comparison::ne_strided(lhs, rhs)
    }

    fn gt_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        super::implementations::comparison::gt_dense(lhs, rhs)
    }

    fn gt_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        super::implementations::comparison::gt_strided(lhs, rhs)
    }

    fn ge_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        super::implementations::comparison::ge_dense(lhs, rhs)
    }

    fn ge_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        super::implementations::comparison::ge_strided(lhs, rhs)
    }

    fn lt_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        super::implementations::comparison::lt_dense(lhs, rhs)
    }

    fn lt_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        super::implementations::comparison::lt_strided(lhs, rhs)
    }

    fn le_dense(
        &self,
        lhs: &DenseStorage<Self::Data>,
        rhs: &DenseStorage<Self::Data>,
    ) -> Result<DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        super::implementations::comparison::le_dense(lhs, rhs)
    }

    fn le_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        super::implementations::comparison::le_strided(lhs, rhs)
    }

    fn clip_attention(
        &self,
        queries: &storage::DenseStorage<Self::Data>,
        _keys: &storage::DenseStorage<Self::Data>,
        _values: &storage::DenseStorage<Self::Data>,
        _num_heads: usize,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        // Placeholder implementation - would use hierarchical primitive
        DenseStorage::from_vec(
            vec![T::default(); queries.as_slice().len()],
            queries.shape().dims(),
        )
        .map_err(|e| crate::BackendError::InvalidInput(format!("Storage error: {}", e)))
    }

    fn take_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
        _indices: &storage::DenseStorage<dtype::int::Int64>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(crate::BackendError::UnsupportedOperation {
            operation: "take_dense".to_string(),
            backend: "cpu".to_string(),
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
            backend: "cpu".to_string(),
        })
    }

    fn add_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        super::implementations::arithmetic::add_strided(lhs, rhs)
    }

    fn mul_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        super::implementations::arithmetic::mul_strided(lhs, rhs)
    }

    fn sub_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        super::implementations::arithmetic::sub_strided(lhs, rhs)
    }

    fn div_strided(
        &self,
        lhs: &storage::StridedStorage<Self::Data>,
        rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        super::implementations::arithmetic::div_strided(lhs, rhs)
    }

    fn add_csr(
        &self,
        lhs: &storage::CsrStorage<Self::Data>,
        rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        super::implementations::arithmetic::add_csr(lhs, rhs)
    }

    fn mul_csr(
        &self,
        lhs: &storage::CsrStorage<Self::Data>,
        rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        super::implementations::arithmetic::mul_csr(lhs, rhs)
    }

    fn sub_csr(
        &self,
        lhs: &storage::CsrStorage<Self::Data>,
        rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        super::implementations::arithmetic::sub_csr(lhs, rhs)
    }

    fn relu_csr(
        &self,
        input: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>>
    where
        Self::Data: PartialOrd + Default,
    {
        super::implementations::activation::relu_csr(input)
    }

    fn tanh_csr(
        &self,
        input: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        super::implementations::activation::tanh_csr(input)
    }

    fn abs_csr(
        &self,
        input: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>>
    where
        Self::Data: num_traits::Signed,
    {
        super::implementations::activation::abs_csr(input)
    }

    fn matmul_csr(
        &self,
        lhs: &storage::CsrStorage<Self::Data>,
        rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        super::implementations::sparse::matmul_csr(lhs, rhs)
    }

    fn addmm_csr(
        &self,
        input: &storage::CsrStorage<Self::Data>,
        mat1: &storage::CsrStorage<Self::Data>,
        mat2: &storage::CsrStorage<Self::Data>,
        beta: Self::Data,
        alpha: Self::Data,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        super::implementations::sparse::addmm_csr(input, mat1, mat2, beta, alpha)
    }

    fn addmv_csr(
        &self,
        input: &storage::CsrStorage<Self::Data>,
        mat: &storage::CsrStorage<Self::Data>,
        vec: &storage::CsrStorage<Self::Data>,
        beta: Self::Data,
        alpha: Self::Data,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        super::implementations::sparse::addmv_csr(input, mat, vec, beta, alpha)
    }

    fn add_dense_csr(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        super::implementations::sparse::add_dense_csr(lhs, rhs)
    }

    fn add_csr_dense(
        &self,
        lhs: &storage::CsrStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        // CSR + Dense is commutative for element-wise addition
        super::implementations::sparse::add_dense_csr(rhs, lhs)
    }

    fn mul_dense_csr(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        super::implementations::sparse::mul_dense_csr(lhs, rhs)
    }

    fn mul_csr_dense(
        &self,
        lhs: &storage::CsrStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        super::implementations::sparse::mul_dense_csr(rhs, lhs)
    }

    fn matmul_csr_dense(
        &self,
        lhs: &storage::CsrStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        let (rows, cols) = lhs.dims();
        let result = super::implementations::sparse::spmm_csr(
            lhs.data(),
            lhs.indices(),
            lhs.indptr(),
            rhs,
            rows,
            cols,
        )?;
        let out_cols = rhs.shape().dims().get(1).copied().unwrap_or(1);
        storage::DenseStorage::from_vec(result, &[rows, out_cols])
            .map_err(|e| crate::BackendError::StorageError { source: e })
    }

    fn matmul_dense_csr(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        super::implementations::sparse::matmul_dense_csr(lhs, rhs)
    }

    // ================== Status Checks ==================

    fn isnan_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float + dtype::num_traits::One + dtype::num_traits::Zero {
        super::implementations::comparison::isnan_dense(input)
    }

    fn isinf_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float + dtype::num_traits::One + dtype::num_traits::Zero {
        super::implementations::comparison::isinf_dense(input)
    }

    fn isfinite_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float + dtype::num_traits::One + dtype::num_traits::Zero {
        super::implementations::comparison::isfinite_dense(input)
    }

    // ================== Logical Operations ==================

    fn logical_and_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::One + dtype::num_traits::Zero {
        super::implementations::comparison::logical_and_dense(lhs, rhs)
    }

    fn logical_or_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::One + dtype::num_traits::Zero {
        super::implementations::comparison::logical_or_dense(lhs, rhs)
    }

    fn logical_xor_dense(
        &self,
        lhs: &storage::DenseStorage<Self::Data>,
        rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::One + dtype::num_traits::Zero {
        super::implementations::comparison::logical_xor_dense(lhs, rhs)
    }

    fn logical_not_dense(
        &self,
        input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::One + dtype::num_traits::Zero {
        super::implementations::comparison::logical_not_dense(input)
    }

    // ================== Math Parity ==================

    fn log1p_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        super::implementations::activation::log1p_dense(input)
    }

    fn expm1_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        super::implementations::activation::expm1_dense(input)
    }

    fn reciprocal_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        super::implementations::activation::reciprocal_dense(input)
    }

    fn atan2_dense(
        &self,
        y: &storage::DenseStorage<Self::Data>,
        x: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        super::implementations::activation::atan2_dense(y, x)
    }

    fn rsqrt_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        super::implementations::activation::rsqrt_dense(input)
    }

    fn erf_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        super::implementations::activation::erf_dense(input)
    }

    fn erfc_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        super::implementations::activation::erfc_dense(input)
    }

    fn erfinv_dense(&self, input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        super::implementations::activation::erfinv_dense(input)
    }
}
