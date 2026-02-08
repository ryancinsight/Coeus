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
//! ├── CpuBackend<T>      // Native CPU execution (SIMD-ready)
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
//! - **Adaptive Selection**: Performance-driven backend selection based on workload characteristics
//!
//! ## Safety
//!
//! All backend operations are memory-safe with zero cost abstractions.

pub mod core;
pub mod selection;
pub mod monitoring;
pub mod runtime;

pub use self::core::{Backend, BackendType};
pub use self::selection::{BackendSelector, WorkloadCharacteristics};
pub use self::monitoring::PerformanceMonitor;
pub use self::runtime::*;

pub use dtype::{num_traits, DataType};
pub use storage::Storage;

// Code moved to submodules

// End of moved code
// Code moved to submodules

/// Performance summary for monitoring
#[derive(Debug)]
pub struct PerformanceSummary {
    pub average_memory_usage_mb: f64,
    pub average_gpu_utilization: f32,
    pub operation_count: usize,
    pub current_step: u64,
}

use std::fmt;

/// Result type for backend operations
pub type Result<T> = std::result::Result<T, BackendError>;

/// Backend-specific errors
#[derive(Debug)]
pub enum BackendError {
    /// Unsupported operation for this backend
    UnsupportedOperation { operation: String, backend: String },
    /// Invalid input parameters
    InvalidInput(String),
    /// Storage operation error
    StorageError { source: storage::StorageError },
    /// GPU-specific error
    GpuError(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::UnsupportedOperation { operation, backend } => {
                write!(f, "Unsupported {operation} operation for {backend} backend")
            }
            BackendError::InvalidInput(msg) => {
                write!(f, "Invalid input: {msg}")
            }
            BackendError::StorageError { source } => {
                write!(f, "Storage error: {source}")
            }
            BackendError::GpuError(msg) => {
                write!(f, "GPU error: {msg}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BackendError {}

#[cfg(feature = "std")]
impl From<storage::StorageError> for BackendError {
    fn from(err: storage::StorageError) -> Self {
        BackendError::InvalidInput(format!("Storage error: {}", err))
    }
}

// Backend trait is re-exported from core module
pub use self::selection::{MemoryAccessPattern, DataLocality, OperationType};
pub use self::core::{DeviceInfo, StubDevice};

/// Stub backend for compilation - provides minimal interface to allow dependent crate testing
#[derive(Debug, Clone)]
pub struct StubBackend<D: DataType> {
    _phantom: std::marker::PhantomData<D>,
}

impl<D: DataType> Default for StubBackend<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<D: DataType> StubBackend<D> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D: DataType> Backend for StubBackend<D> {
    type Data = D;
    type Device = StubDevice;

    fn device(&self) -> &Self::Device {
        static DEVICE: StubDevice = StubDevice;
        &DEVICE
    }

    fn supports(&self, _operation: &str) -> bool {
        true // Stub always supports operations
    }

    fn device_name(&self) -> &str {
        "stub"
    }

    fn device_info(&self) -> Box<dyn DeviceInfo> {
        Box::new(StubDevice)
    }

    fn add_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "add_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn mul_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "mul_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn add_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "add_strided".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn mul_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "mul_strided".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn matmul_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "matmul_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn relu_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd + Default,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "relu_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn sigmoid_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "sigmoid_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn sum_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<Self::Data> {
        Err(BackendError::UnsupportedOperation {
            operation: "sum_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn max_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "max_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn min_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<Self::Data>
    where
        Self::Data: PartialOrd,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "min_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn argmax_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<usize>
    where
        Self::Data: PartialOrd,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "argmax_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn argmin_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<usize>
    where
        Self::Data: PartialOrd,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "argmin_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn sub_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "sub_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn sub_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "sub_strided".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn div_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "div_strided".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn add_csr(
        &self,
        _lhs: &storage::CsrStorage<Self::Data>,
        _rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "add_csr".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn mul_csr(
        &self,
        _lhs: &storage::CsrStorage<Self::Data>,
        _rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "mul_csr".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn sub_csr(
        &self,
        _lhs: &storage::CsrStorage<Self::Data>,
        _rhs: &storage::CsrStorage<Self::Data>,
    ) -> Result<storage::CsrStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "sub_csr".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn exp_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "exp_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn log_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "log_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn sin_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "sin_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn cos_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "cos_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn tan_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "tan_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn asin_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "asin_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn acos_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "acos_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn atan_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "atan_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn sinh_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "sinh_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn cosh_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "cosh_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn tanh_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "tanh_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn gelu_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> 
    where
        Self::Data: num_traits::Float,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "gelu_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn sqrt_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "sqrt_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn abs_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "abs_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn floor_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "floor_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn ceil_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "ceil_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn round_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "round_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn cholesky_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: num_traits::Float,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "cholesky_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn qr_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<(storage::DenseStorage<Self::Data>, storage::DenseStorage<Self::Data>)>
    where
        Self::Data: num_traits::Float,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "qr_dense".to_string(),
            backend: "stub".to_string(),
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
        Err(BackendError::UnsupportedOperation {
            operation: "svd_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn take_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
        _indices: &storage::DenseStorage<dtype::int::Int64>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "take_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn put_dense(
        &self,
        _input: &mut storage::DenseStorage<Self::Data>,
        _indices: &storage::DenseStorage<dtype::int::Int64>,
        _values: &storage::DenseStorage<Self::Data>,
        _accumulate: bool,
    ) -> Result<()> {
        Err(BackendError::UnsupportedOperation {
            operation: "put_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn conv2d_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
        _weight: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "conv2d_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn mean_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
        _axes: Option<&[usize]>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "mean_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn spmm_csr(
        &self,
        _data: &[Self::Data],
        _indices: &[usize],
        _indptr: &[usize],
        _other: &storage::DenseStorage<Self::Data>,
        _num_rows: usize,
        _num_cols: usize,
    ) -> Result<Vec<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "spmm_csr".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn spmv_csr(
        &self,
        _data: &[Self::Data],
        _indices: &[usize],
        _indptr: &[usize],
        _vector: &[Self::Data],
        _num_rows: usize,
        _num_cols: usize,
    ) -> Result<Vec<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "spmv_csr".to_string(),
            backend: "stub".to_string(),
        })
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
    ) -> Result<storage::CsrStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "coo_matmul_sparse".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn coo_matmul_dense(
        &self,
        _lhs_data: &[Self::Data],
        _lhs_row: &[usize],
        _lhs_col: &[usize],
        _rhs: &storage::DenseStorage<Self::Data>,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "coo_matmul_dense".to_string(),
            backend: "stub".to_string(),
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
    ) -> Result<storage::CsrStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "coo_add_sparse".to_string(),
            backend: "stub".to_string(),
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
    ) -> Result<storage::CsrStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "coo_mul_sparse".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn quantize(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
        _levels: usize,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "quantize".to_string(),
            backend: "stub".to_string(),
        })
    }

    /// Compute CLIP InfoNCE loss for contrastive learning
    fn clip_info_nce_loss(
        &self,
        _image_embeddings: &storage::DenseStorage<Self::Data>,
        _text_embeddings: &storage::DenseStorage<Self::Data>,
        _temperature: f32,
    ) -> Result<Self::Data> {
        Err(BackendError::UnsupportedOperation {
            operation: "clip_info_nce_loss".to_string(),
            backend: "stub".to_string(),
        })
    }

    /// Compute CLIP attention mechanism
    fn clip_attention(
        &self,
        _queries: &storage::DenseStorage<Self::Data>,
        _keys: &storage::DenseStorage<Self::Data>,
        _values: &storage::DenseStorage<Self::Data>,
        _num_heads: usize,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "clip_attention".to_string(),
            backend: "stub".to_string(),
        })
    }

    // ================== Comparison ==================

    fn eq_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "eq_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn eq_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "eq_strided".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn ne_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "ne_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn ne_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>> {
        Err(BackendError::UnsupportedOperation {
            operation: "ne_strided".to_string(),
            backend: "stub".to_string(),
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
        Err(BackendError::UnsupportedOperation {
            operation: "gt_dense".to_string(),
            backend: "stub".to_string(),
        })
    }

    fn isnan_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float + dtype::num_traits::One + dtype::num_traits::Zero {
        Err(BackendError::UnsupportedOperation { operation: "isnan_dense".to_string(), backend: "stub".to_string() })
    }

    fn isinf_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float + dtype::num_traits::One + dtype::num_traits::Zero {
        Err(BackendError::UnsupportedOperation { operation: "isinf_dense".to_string(), backend: "stub".to_string() })
    }

    fn isfinite_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float + dtype::num_traits::One + dtype::num_traits::Zero {
        Err(BackendError::UnsupportedOperation { operation: "isfinite_dense".to_string(), backend: "stub".to_string() })
    }

    fn logical_and_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::One + dtype::num_traits::Zero {
        Err(BackendError::UnsupportedOperation { operation: "logical_and_dense".to_string(), backend: "stub".to_string() })
    }

    fn logical_or_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::One + dtype::num_traits::Zero {
        Err(BackendError::UnsupportedOperation { operation: "logical_or_dense".to_string(), backend: "stub".to_string() })
    }

    fn logical_xor_dense(
        &self,
        _lhs: &storage::DenseStorage<Self::Data>,
        _rhs: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::One + dtype::num_traits::Zero {
        Err(BackendError::UnsupportedOperation { operation: "logical_xor_dense".to_string(), backend: "stub".to_string() })
    }

    fn logical_not_dense(
        &self,
        _input: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::One + dtype::num_traits::Zero {
        Err(BackendError::UnsupportedOperation { operation: "logical_not_dense".to_string(), backend: "stub".to_string() })
    }

    fn log1p_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "log1p_dense".to_string(), backend: "stub".to_string() })
    }

    fn expm1_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "expm1_dense".to_string(), backend: "stub".to_string() })
    }

    fn reciprocal_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "reciprocal_dense".to_string(), backend: "stub".to_string() })
    }

    fn atan2_dense(
        &self,
        _y: &storage::DenseStorage<Self::Data>,
        _x: &storage::DenseStorage<Self::Data>,
    ) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "atan2_dense".to_string(), backend: "stub".to_string() })
    }

    fn rsqrt_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "rsqrt_dense".to_string(), backend: "stub".to_string() })
    }

    fn erf_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "erf_dense".to_string(), backend: "stub".to_string() })
    }

    fn erfc_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "erfc_dense".to_string(), backend: "stub".to_string() })
    }

    fn erfinv_dense(&self, _input: &storage::DenseStorage<Self::Data>) -> Result<storage::DenseStorage<Self::Data>>
    where Self::Data: dtype::num_traits::Float {
        Err(BackendError::UnsupportedOperation { operation: "erfinv_dense".to_string(), backend: "stub".to_string() })
    }

    fn gt_strided(
        &self,
        _lhs: &storage::StridedStorage<Self::Data>,
        _rhs: &storage::StridedStorage<Self::Data>,
    ) -> Result<storage::StridedStorage<Self::Data>>
    where
        Self::Data: PartialOrd,
    {
        Err(BackendError::UnsupportedOperation {
            operation: "gt_strided".to_string(),
            backend: "stub".to_string(),
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
        Err(BackendError::UnsupportedOperation {
            operation: "ge_dense".to_string(),
            backend: "stub".to_string(),
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
        Err(BackendError::UnsupportedOperation {
            operation: "ge_strided".to_string(),
            backend: "stub".to_string(),
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
        Err(BackendError::UnsupportedOperation {
            operation: "lt_dense".to_string(),
            backend: "stub".to_string(),
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
        Err(BackendError::UnsupportedOperation {
            operation: "lt_strided".to_string(),
            backend: "stub".to_string(),
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
        Err(BackendError::UnsupportedOperation {
            operation: "le_dense".to_string(),
            backend: "stub".to_string(),
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
        Err(BackendError::UnsupportedOperation {
            operation: "le_strided".to_string(),
            backend: "stub".to_string(),
        })
    }
}

/// Placeholder memory manager for backend selection
/// TODO: Replace with full memory management implementation
#[derive(Debug, Clone)]
pub struct MemoryManager;

/// Memory analysis hints for backend selection
#[derive(Debug)]
pub struct MemoryAnalysisResult {
    /// Recommended backend based on memory constraints
    pub recommended_backend: Option<BackendType>,
    /// Memory efficiency score (0.0-1.0)
    pub memory_efficiency: f32,
    /// Transfer cost estimate
    pub transfer_cost: f64,
    /// Fragmentation impact
    pub fragmentation_penalty: f32,
}

impl MemoryManager {
    /// Analyze memory constraints for backend selection
    pub async fn analyze_memory_for_selection(
        &self,
        _workload: &crate::distributed::DistributedWorkloadCharacteristics,
        _backends: &[BackendType],
    ) -> MemoryAnalysisResult {
        // Placeholder implementation - always return no recommendation
        MemoryAnalysisResult {
            recommended_backend: None,
            memory_efficiency: 0.5,
            transfer_cost: 0.0,
            fragmentation_penalty: 0.0,
        }
    }
}

/// Backend device
pub mod device;
pub mod distributed;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(not(feature = "gpu"))]
pub mod gpu {
    // Stub definition for when GPU feature is disabled
    // ...
    // Note: Use abbreviated stub for brevity in replace
    pub struct GpuBackend<T>(std::marker::PhantomData<T>);
    impl<T> GpuBackend<T> { pub fn new() -> Self { Self(std::marker::PhantomData) } }
}

pub mod cpu;

pub use cpu::CpuBackend;
pub use device::Device; // DeviceInfo is exported via core
#[cfg(feature = "gpu")]
pub use gpu::GpuBackend;

pub use distributed::{
    BackendSelectionDecision, CoordinationStats, DistributedBackendCoordinator,
    DistributedWorkloadAnalyzer, DistributedWorkloadCharacteristics, FaultToleranceState,
    MemoryConstraints,
};

// Memory management integration
pub mod memory_integration;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuBackend;
    use dtype::float::Float32;
    use std::vec;

    #[test]
    fn test_spmv_csr_basic() {
        let backend = CpuBackend::<Float32>::new();

        // Create a simple 3x3 sparse matrix in CSR format:
        // [[1, 0, 2],
        //  [0, 3, 0],
        //  [4, 0, 5]]
        // Maps to data=[1,2,3,4,5], indices=[0,2,1,0,2], indptr=[0,2,3,5]
        let data: Vec<Float32> = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
        ];
        let indices: Vec<usize> = vec![0, 2, 1, 0, 2];
        let indptr: Vec<usize> = vec![0, 2, 3, 5];

        // Create a dense vector [1, 2, 3]
        let vector: Vec<Float32> = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];

        // Perform SPMV: result should be [1*1 + 2*3, 3*2, 4*1 + 5*3] = [7, 6, 19]
        let result = backend
            .spmv_csr(&data, &indices, &indptr, &vector, 3, 3)
            .unwrap();

        assert_eq!(result.len(), 3);
        assert!((result[0].get() - 7.0).abs() < 1e-6);
        assert!((result[1].get() - 6.0).abs() < 1e-6);
        assert!((result[2].get() - 19.0).abs() < 1e-6);
    }

    #[test]
    fn test_backend_selector_creation() {
        let selector = BackendSelector::new();
        assert!(selector.available_backends().contains(&BackendType::Cpu));
    }

    #[test]
    fn test_backend_selection_small_element_wise() {
        let selector = BackendSelector::new();
        let workload = WorkloadCharacteristics {
            total_elements: 1000,
            access_pattern: MemoryAccessPattern::Dense,
            compute_intensity: 1.0,
            data_locality: DataLocality::High,
            operation_type: OperationType::ElementWise,
        };

        let selected = selector.select_backend(&workload);
        assert_eq!(selected, BackendType::Cpu);
    }

    #[test]
    fn test_backend_selection_large_matmul() {
        let selector = BackendSelector::new();
        let workload = WorkloadCharacteristics {
            total_elements: 2_000_000,
            access_pattern: MemoryAccessPattern::Dense,
            compute_intensity: 20.0,
            data_locality: DataLocality::High,
            operation_type: OperationType::MatrixMultiplication,
        };

        let selected = selector.select_backend(&workload);
        // GPUs should be preferred for large matrix multiplications
        assert_eq!(selected, BackendType::Gpu);
    }

    /*
    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new(1.0);

        monitor.record_memory_usage(512.0);
        monitor.record_utilization(85.0);
        monitor.record_operation_latency("matmul", 1500.0);

        let summary = monitor.get_performance_summary();
        assert_eq!(summary.average_memory_usage_mb, 512.0);
        assert_eq!(summary.average_gpu_utilization, 85.0);
        assert_eq!(summary.operation_count, 1);

        let total_training_time = 100_000.0;
        let overhead = monitor.calculate_gpu_overhead(total_training_time);
        // Overhead calculation may vary based on recorded metrics
        assert!((0.0..=50.0).contains(&overhead));
    }
    */

    #[test]
    fn test_matmul_mathematical_correctness() {
        // Test matrix multiplication against analytical results
        let backend = CpuBackend::<Float32>::new();

        // Test case: 2x3 @ 3x2 = 2x2
        let lhs_data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ];
        let rhs_data = vec![
            Float32::new(7.0),
            Float32::new(8.0),
            Float32::new(9.0),
            Float32::new(10.0),
            Float32::new(11.0),
            Float32::new(12.0),
        ];

        let lhs = storage::DenseStorage::from_vec(lhs_data, &[2, 3]).unwrap();
        let rhs = storage::DenseStorage::from_vec(rhs_data, &[3, 2]).unwrap();

        let result = backend.matmul_dense(&lhs, &rhs).unwrap();

        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //         = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
        //         = [[58, 64], [139, 154]]
        let expected_data = vec![
            Float32::new(58.0),
            Float32::new(64.0),
            Float32::new(139.0),
            Float32::new(154.0),
        ];
        let expected = storage::DenseStorage::from_vec(expected_data, &[2, 2]).unwrap();

        assert_eq!(result.shape().dims(), &[2, 2]);
        for (r, e) in result.as_slice().iter().zip(expected.as_slice().iter()) {
            assert!(
                (r.get() - e.get()).abs() < 1e-6,
                "Result: {}, Expected: {}",
                r.get(),
                e.get()
            );
        }
    }

    #[test]
    fn test_mean_reduction_correctness() {
        // Test mean reduction against analytical results
        let backend = CpuBackend::<Float32>::new();

        // 2x3 matrix: [[1, 2, 3], [4, 5, 6]]
        let data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ];
        let tensor = storage::DenseStorage::from_vec(data, &[2, 3]).unwrap();

        // Global mean: (1+2+3+4+5+6)/6 = 21/6 = 3.5
        let global_mean = backend.mean_dense(&tensor, None).unwrap();
        assert_eq!(global_mean.shape().dims(), &[]);
        assert!((global_mean.as_slice()[0].get() - 3.5).abs() < 1e-6);

        // Mean along axis 0 (reduce first dimension): [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
        let axis0_mean = backend.mean_dense(&tensor, Some(&[0])).unwrap();
        assert_eq!(axis0_mean.shape().dims(), &[3]);
        let expected_axis0 = [Float32::new(2.5), Float32::new(3.5), Float32::new(4.5)];
        for (r, e) in axis0_mean.as_slice().iter().zip(expected_axis0.iter()) {
            assert!((r.get() - e.get()).abs() < 1e-6);
        }

        // Mean along axis 1 (reduce second dimension): [(1+2+3)/3, (4+5+6)/3] = [2.0, 5.0]
        let axis1_mean = backend.mean_dense(&tensor, Some(&[1])).unwrap();
        assert_eq!(axis1_mean.shape().dims(), &[2]);
        let expected_axis1 = [Float32::new(2.0), Float32::new(5.0)];
        for (r, e) in axis1_mean.as_slice().iter().zip(expected_axis1.iter()) {
            assert!((r.get() - e.get()).abs() < 1e-6);
        }
    }

    #[test]
    fn test_element_wise_operations_precision() {
        // Test element-wise operations for numerical precision
        let backend = CpuBackend::<Float32>::new();

        let data = vec![
            Float32::new(1.5),
            Float32::new(-2.7),
            Float32::new(std::f32::consts::PI),
            Float32::new(-0.5),
            Float32::new(10.0),
            Float32::new(0.001),
        ];
        let tensor = storage::DenseStorage::from_vec(data.clone(), &[2, 3]).unwrap();

        // Test exp: e^1.5, e^-2.7, e^3.14, e^-0.5, e^10.0, e^0.001
        let exp_result = backend.exp_dense(&tensor).unwrap();
        for (i, &val) in data.iter().enumerate() {
            let expected = val.get().exp();
            let actual = exp_result.as_slice()[i].get();
            assert!(
                (actual - expected).abs() < 1e-6,
                "exp({}) = {} vs {}",
                val.get(),
                actual,
                expected
            );
        }

        // Test ReLU: max(0, x)
        let relu_result = backend.relu_dense(&tensor).unwrap();
        let expected_relu = [1.5, 0.0, std::f32::consts::PI, 0.0, 10.0, 0.001];
        for (i, &expected) in expected_relu.iter().enumerate() {
            let actual = relu_result.as_slice()[i].get();
            assert!(
                (actual - expected).abs() < 1e-6,
                "ReLU result[{}] = {} vs {}",
                i,
                actual,
                expected
            );
        }
    }

    #[test]
    fn test_sparse_coo_operations_correctness() {
        // Test COO sparse operations for mathematical correctness
        let backend = CpuBackend::<Float32>::new();

        // Test coo_add_sparse: [1, 0; 0, 2] + [0, 1; 3, 0] = [1, 1; 3, 2]
        let lhs_data = vec![Float32::new(1.0), Float32::new(2.0)];
        let lhs_row = vec![0, 1];
        let lhs_col = vec![0, 1];

        let rhs_data = vec![Float32::new(1.0), Float32::new(3.0)];
        let rhs_row = vec![0, 1];
        let rhs_col = vec![1, 0];

        let result = backend
            .coo_add_sparse(
                &lhs_data, &lhs_row, &lhs_col, &rhs_data, &rhs_row, &rhs_col, 2, 2,
            )
            .unwrap();

        // Should have 4 non-zero elements
        assert_eq!(result.nnz(), 4);
        assert_eq!(result.indices().len(), 4);
        assert_eq!(result.indptr().len(), 3);

        // Test coo_mul_sparse: element-wise multiplication
        let mul_result = backend
            .coo_mul_sparse(
                &lhs_data, &lhs_row, &lhs_col, &rhs_data, &rhs_row, &rhs_col, 2, 2,
            )
            .unwrap();
        // Only position (0,1) has non-zero values in both matrices: 0 * 1 = 0, so result should be empty or have zero elements
        // Actually, no positions have non-zero values in both matrices, so result should be empty
        assert_eq!(mul_result.nnz(), 0);
    }

    #[test]
    fn test_clip_info_nce_loss_validation() {
        // Test CLIP InfoNCE loss against simplified analytical case
        let backend = CpuBackend::<Float32>::new();

        // Simple 2x2 case: two embeddings per batch
        // image_embeddings: [[1, 0], [0, 1]]
        // text_embeddings: [[1, 0], [0, 1]]
        let image_data = vec![
            Float32::new(1.0),
            Float32::new(0.0),
            Float32::new(0.0),
            Float32::new(1.0),
        ];
        let text_data = vec![
            Float32::new(1.0),
            Float32::new(0.0),
            Float32::new(0.0),
            Float32::new(1.0),
        ];

        let image_tensor = storage::DenseStorage::from_vec(image_data, &[2, 2]).unwrap();
        let text_tensor = storage::DenseStorage::from_vec(text_data, &[2, 2]).unwrap();

        let temperature = 1.0f32;
        let loss = backend
            .clip_info_nce_loss(&image_tensor, &text_tensor, temperature)
            .unwrap();

        // For this case with identical normalized embeddings:
        // Each positive pair has similarity = 1.0, negative pairs have similarity = 0.0
        // The loss should be a positive value (since it's a contrastive loss)
        // We just verify it's reasonable and not NaN/Infinite
        assert!(
            loss.get() > 0.0 && loss.get() < 2.0,
            "CLIP InfoNCE loss should be positive and reasonable: {}",
            loss.get()
        );
        assert!(
            loss.get().is_finite(),
            "CLIP InfoNCE loss should be finite: {}",
            loss.get()
        );
    }

    #[test]
    fn test_reduction_operations_correctness() {
        // Test reduction operations (sum, max, min, argmax, argmin)
        let backend = CpuBackend::<Float32>::new();

        let data = vec![
            Float32::new(3.0),
            Float32::new(1.0),
            Float32::new(4.0),
            Float32::new(1.0),
            Float32::new(5.0),
            Float32::new(9.0),
        ];
        let tensor = storage::DenseStorage::from_vec(data, &[2, 3]).unwrap();

        // Sum: 3+1+4+1+5+9 = 23
        let sum_result = backend.sum_dense(&tensor).unwrap();
        assert!((sum_result.get() - 23.0).abs() < 1e-6);

        // Max: 9
        let max_result = backend.max_dense(&tensor).unwrap();
        assert!((max_result.get() - 9.0).abs() < 1e-6);

        // Min: 1
        let min_result = backend.min_dense(&tensor).unwrap();
        assert!((min_result.get() - 1.0).abs() < 1e-6);

        // Argmax: index 5 (9.0 at position [1,2] in row-major order)
        let argmax_result = backend.argmax_dense(&tensor).unwrap();
        assert_eq!(argmax_result, 5);

        // Argmin: index 1 or 3 (1.0 at positions [0,1] or [1,0])
        let argmin_result = backend.argmin_dense(&tensor).unwrap();
        assert!(argmin_result == 1 || argmin_result == 3);
    }

    #[test]
    fn test_csr_arithmetic_correctness() {
        let backend = CpuBackend::<Float32>::new();

        // Matrix A (3x3):
        // [[1, 0, 2],
        //  [0, 3, 0],
        //  [4, 0, 0]]
        let a_data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0), Float32::new(4.0)];
        let a_indices = vec![0, 2, 1, 0];
        let a_indptr = vec![0, 2, 3, 4];
        let a = storage::CsrStorage::new(a_data, a_indices, a_indptr, &[3, 3]).unwrap();

        // Matrix B (3x3):
        // [[0, 1, 1],
        //  [2, 0, 0],
        //  [0, 0, 5]]
        let b_data = vec![Float32::new(1.0), Float32::new(1.0), Float32::new(2.0), Float32::new(5.0)];
        let b_indices = vec![1, 2, 0, 2];
        let b_indptr = vec![0, 2, 3, 4];
        let b = storage::CsrStorage::new(b_data, b_indices, b_indptr, &[3, 3]).unwrap();

        // A + B = 
        // [[1, 1, 3],
        //  [2, 3, 0],
        //  [4, 0, 5]]
        let add_res = backend.add_csr(&a, &b).unwrap();
        assert_eq!(add_res.nnz(), 7);
        let dense_add = add_res.to_dense().unwrap();
        let expected_add = vec![1.0, 1.0, 3.0, 2.0, 3.0, 0.0, 4.0, 0.0, 5.0];
        for (i, &e) in expected_add.iter().enumerate() {
            assert!((dense_add.as_slice()[i].get() - e).abs() < 1e-6);
        }

        // A - B =
        // [[1, -1, 1],
        //  [-2, 3, 0],
        //  [4, 0, -5]]
        let sub_res = backend.sub_csr(&a, &b).unwrap();
        assert_eq!(sub_res.nnz(), 7);
        let dense_sub = sub_res.to_dense().unwrap();
        let expected_sub = vec![1.0, -1.0, 1.0, -2.0, 3.0, 0.0, 4.0, 0.0, -5.0];
        for (i, &e) in expected_sub.iter().enumerate() {
            assert!((dense_sub.as_slice()[i].get() - e).abs() < 1e-6);
        }

        // A * B =
        // [[0, 0, 2],
        //  [0, 0, 0],
        //  [0, 0, 0]]
        let mul_res = backend.mul_csr(&a, &b).unwrap();
        assert_eq!(mul_res.nnz(), 1);
        let dense_mul = mul_res.to_dense().unwrap();
        let expected_mul = vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for (i, &e) in expected_mul.iter().enumerate() {
            assert!((dense_mul.as_slice()[i].get() - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_div_strided_correctness() {
        let backend = CpuBackend::<Float32>::new();

        let lhs_data = vec![Float32::new(10.0), Float32::new(20.0), Float32::new(30.0), Float32::new(40.0)];
        let rhs_data = vec![Float32::new(2.0), Float32::new(2.0), Float32::new(2.0), Float32::new(2.0)];

        let lhs = storage::StridedStorage::new(lhs_data, &[2, 2]).unwrap();
        let rhs = storage::StridedStorage::new(rhs_data, &[2, 2]).unwrap();

        let res = backend.div_strided(&lhs, &rhs).unwrap();
        let expected = vec![5.0, 10.0, 15.0, 20.0];
        for (i, &e) in expected.iter().enumerate() {
            assert!((res.as_slice()[i].get() - e).abs() < 1e-6);
        }
    }
}
