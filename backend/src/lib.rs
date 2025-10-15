//! # Coeus Backend Abstractions
//!
//! Compute device abstractions enabling execution on CPU, GPU, and other accelerators.
//!
//! ## Architecture
//!
//! Backend traits separate compute substrate from tensor storage/dtype logic,
//! enabling zero-cost backend dispatch via static monomorphization.
//!
//! ### Backend Trait Hierarchy
//!
//! ```text
//! Backend
//! ├── CpuBackend      // Native CPU execution (SIMD-ready)
//! ├── GpuBackend      // GPU via wgpu (future)
//! └── NpuBackend      // Neural processors (future)
//! ```
//!
//! ## Design Principles (ADR-003)
//!
//! - **Zero-Cost Dispatch**: Static monomorphization eliminates runtime overhead
//! - **Send + Sync**: Thread-safe by construction for parallel execution
//! - **Extensibility**: New backends via trait implementation
//! - **Device Capability**: Runtime feature detection for optimal paths
//!
//! ## Safety
//!
//! All backend operations are memory-safe with zero unsafe code.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

use alloc::vec::Vec;

pub use coeus_dtype::DataType;
pub use coeus_storage::Storage;

/// Result type for backend operations
pub type Result<T> = core::result::Result<T, BackendError>;

/// Backend-specific errors
#[derive(Debug)]
pub enum BackendError {
    /// Unsupported operation for this backend
    UnsupportedOperation {
        operation: alloc::string::String,
        backend: alloc::string::String,
    },
    /// Invalid input parameters
    InvalidInput(alloc::string::String),
}

impl core::fmt::Display for BackendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BackendError::UnsupportedOperation { operation, backend } => {
                write!(f, "Unsupported {operation} operation for {backend} backend")
            }
            BackendError::InvalidInput(msg) => {
                write!(f, "Invalid input: {msg}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BackendError {}

#[cfg(feature = "std")]
impl From<coeus_storage::StorageError> for BackendError {
    fn from(err: coeus_storage::StorageError) -> Self {
        BackendError::InvalidInput(alloc::format!("Storage error: {}", err))
    }
}

pub mod cpu;
pub mod device;

#[cfg(all(feature = "gpu", feature = "std"))]
pub mod gpu;

#[cfg(all(feature = "npu", feature = "std"))]
pub mod npu;

#[cfg(all(feature = "tpu", feature = "std"))]
pub mod tpu;

pub use cpu::CpuBackend;
pub use device::{Device, DeviceInfo};

#[cfg(all(feature = "gpu", feature = "std"))]
pub use gpu::GpuBackend;

#[cfg(all(feature = "npu", feature = "std"))]
pub use npu::NpuBackend;

#[cfg(all(feature = "tpu", feature = "std"))]
pub use tpu::TpuBackend;

/// Core backend trait for compute device abstraction.
///
/// Defines the interface all compute backends must implement, enabling
/// zero-cost dispatch to different hardware substrates.
///
/// # Type Safety
///
/// Backends are `Send + Sync` by requirement, ensuring thread-safe execution
/// for parallel tensor operations.
///
/// # Examples
///
/// ```
/// use coeus_backend::{Backend, CpuBackend, DeviceInfo};
///
/// let backend = CpuBackend::new();
/// assert_eq!(backend.device().name(), "cpu");
/// ```
pub trait Backend: Clone + core::fmt::Debug + Send + Sync + Sized + 'static {
    /// Device information for this backend
    type DeviceType: DeviceInfo;

    /// Returns device information
    fn device(&self) -> &Self::DeviceType;

    /// Returns the device name as a string
    fn device_name(&self) -> &str {
        self.device().name()
    }

    /// Returns true if this backend supports the given operation
    ///
    /// Enables runtime feature detection for optimal code paths.
    fn supports(&self, operation: &str) -> bool;

    /// Perform element-wise addition on dense storage
    ///
    /// # Arguments
    /// * `lhs` - Left-hand side dense storage
    /// * `rhs` - Right-hand side dense storage
    ///
    /// # Returns
    /// New dense storage containing element-wise sum
    fn add_dense<T>(&self, lhs: &coeus_storage::DenseStorage<T>, rhs: &coeus_storage::DenseStorage<T>) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType;

    /// Perform element-wise multiplication on dense storage
    ///
    /// # Arguments
    /// * `lhs` - Left-hand side dense storage
    /// * `rhs` - Right-hand side dense storage
    ///
    /// # Returns
    /// New dense storage containing element-wise product
    fn mul_dense<T>(&self, lhs: &coeus_storage::DenseStorage<T>, rhs: &coeus_storage::DenseStorage<T>) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType;

    /// Perform matrix multiplication on dense storage
    ///
    /// # Arguments
    /// * `lhs` - Left-hand side matrix dense storage
    /// * `rhs` - Right-hand side matrix dense storage
    ///
    /// # Returns
    /// New dense storage containing matrix product
    fn matmul_dense<T>(&self, lhs: &coeus_storage::DenseStorage<T>, rhs: &coeus_storage::DenseStorage<T>) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType;

    /// Perform element-wise exponential on dense storage
    ///
    /// # Arguments
    /// * `input` - Input dense storage
    ///
    /// # Returns
    /// New dense storage containing exp(input)
    fn exp_dense<T>(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType;

    /// Perform element-wise logarithm on dense storage
    ///
    /// # Arguments
    /// * `input` - Input dense storage
    ///
    /// # Returns
    /// New dense storage containing log(input)
    fn log_dense<T>(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType;

    /// Perform element-wise sine on dense storage
    ///
    /// # Arguments
    /// * `input` - Input dense storage
    ///
    /// # Returns
    /// New dense storage containing sin(input)
    fn sin_dense<T>(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType;

    /// Perform element-wise cosine on dense storage
    ///
    /// # Arguments
    /// * `input` - Input dense storage
    ///
    /// # Returns
    /// New dense storage containing cos(input)
    fn cos_dense<T>(&self, input: &coeus_storage::DenseStorage<T>) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType;

    /// Perform 2D convolution on dense storage
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape (N, C_in, H_in, W_in)
    /// * `weight` - Weight tensor of shape (C_out, C_in, K_h, K_w)
    /// * `bias` - Optional bias tensor of shape (C_out,)
    /// * `stride` - Stride for height and width (stride_h, stride_w)
    /// * `padding` - Padding for height and width (pad_h, pad_w)
    ///
    /// # Returns
    /// Output tensor of shape (N, C_out, H_out, W_out)
    fn conv2d_dense<T>(
        &self,
        input: &coeus_storage::DenseStorage<T>,
        weight: &coeus_storage::DenseStorage<T>,
        bias: Option<&coeus_storage::DenseStorage<T>>,
        stride: (usize, usize),
        padding: (usize, usize),
        input_shape: &[usize],
        weight_shape: &[usize],
    ) -> crate::Result<coeus_storage::DenseStorage<T>>
    where
        T: crate::DataType;

    /// Perform sparse matrix multiplication (CSR format)
    ///
    /// # Arguments
    /// * `lhs_data` - Non-zero values of left matrix
    /// * `lhs_indices` - Column indices of left matrix
    /// * `lhs_indptr` - Row pointers of left matrix
    /// * `rhs_data` - Non-zero values of right matrix
    /// * `rhs_indices` - Column indices of right matrix
    /// * `rhs_indptr` - Row pointers of right matrix
    /// * `m` - Rows in left matrix
    /// * `k` - Columns in left matrix / rows in right matrix
    /// * `n` - Columns in right matrix
    ///
    /// # Returns
    /// Result matrix in COO format (data, row_indices, col_indices)
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
        T: crate::DataType;

    /// Perform sparse matrix-dense vector multiplication (CSR format)
    ///
    /// # Arguments
    /// * `matrix_data` - Non-zero values of sparse matrix
    /// * `matrix_indices` - Column indices of sparse matrix
    /// * `matrix_indptr` - Row pointers of sparse matrix
    /// * `vector` - Dense vector
    /// * `rows` - Number of rows in matrix
    /// * `cols` - Number of columns in matrix
    ///
    /// # Returns
    /// Dense result vector
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
        T: crate::DataType;

    /// Perform quantization of tensor data
    ///
    /// # Arguments
    /// * `input` - Input tensor data to quantize
    /// * `scale` - Quantization scale factor
    /// * `zero_point` - Quantization zero point
    /// * `bits` - Target quantization bitwidth (4, 8, or 16)
    /// * `scheme` - Quantization scheme (Affine or Symmetric)
    ///
    /// # Returns
    /// Quantized data as packed bytes
    fn quantize<T>(
        &self,
        input: &[T],
        scale: T,
        zero_point: T,
        bits: usize,
        scheme: &str,
    ) -> crate::Result<Vec<u8>>
    where
        T: crate::DataType;

    /// Perform dequantization of packed quantized data
    ///
    /// # Arguments
    /// * `quantized_data` - Packed quantized data
    /// * `scale` - Quantization scale factor
    /// * `zero_point` - Quantization zero point
    /// * `bits` - Quantization bitwidth (4, 8, or 16)
    /// * `scheme` - Quantization scheme (Affine or Symmetric)
    /// * `output_size` - Number of output elements
    ///
    /// # Returns
    /// Dequantized tensor data
    fn dequantize<T>(
        &self,
        quantized_data: &[u8],
        scale: T,
        zero_point: T,
        bits: usize,
        scheme: &str,
        output_size: usize,
    ) -> crate::Result<Vec<T>>
    where
        T: crate::DataType;

    /// Perform quantized matrix multiplication (GEMM)
    ///
    /// # Arguments
    /// * `lhs_data` - Left matrix quantized data (packed)
    /// * `lhs_scale` - Left matrix quantization scale
    /// * `lhs_zero_point` - Left matrix quantization zero point
    /// * `rhs_data` - Right matrix quantized data (packed)
    /// * `rhs_scale` - Right matrix quantization scale
    /// * `rhs_zero_point` - Right matrix quantization zero point
    /// * `bias` - Optional bias vector
    /// * `m` - Rows in left matrix
    /// * `k` - Columns in left matrix / rows in right matrix
    /// * `n` - Columns in right matrix
    /// * `bits` - Quantization bitwidth
    /// * `scheme` - Quantization scheme
    ///
    /// # Returns
    /// Result matrix in full precision
    fn quantized_matmul<T>(
        &self,
        lhs_data: &[u8],
        lhs_scale: T,
        lhs_zero_point: T,
        rhs_data: &[u8],
        rhs_scale: T,
        rhs_zero_point: T,
        bias: Option<&[T]>,
        m: usize,
        k: usize,
        n: usize,
        bits: usize,
        scheme: &str,
    ) -> crate::Result<Vec<T>>
    where
        T: crate::DataType;
}
