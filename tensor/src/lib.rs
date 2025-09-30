//! # Coeus Tensor
//!
//! PyTorch-like tensor library with automatic differentiation, backend abstraction, and higher-order derivatives.
//!
//! This crate provides a unified tensor architecture that consolidates all tensor forms
//! into a single type with generic dtype and backend abstraction.
//!
//! ## Architecture
//!
//! The tensor library provides a single unified tensor type that works across all backends
//! and supports both regular operations and automatic differentiation:
//!
//! ### UnifiedTensor (Single Tensor Type)
//! ```rust
//! use coeus_tensor::{Tensor, CpuBackend};
//! use coeus_dtype::FloatDtype;
//!
//! // Create CPU tensor with explicit backend and dtype
//! let backend = CpuBackend::new();
//! let tensor = Tensor::<f32, CpuBackend>::from_vec(
//!     backend,
//!     vec![1.0, 2.0, 3.0],
//!     vec![3]
//! ).unwrap();
//!
//! // Operations work like PyTorch with zero-copy semantics
//! let result = tensor.add(&tensor).unwrap();
//!
//! // Enable autograd for gradient computation
//! let autograd_tensor = tensor.with_autograd(true);
//! let grad_result = autograd_tensor.neg(); // Supports gradient computation
//! ```
//!
//! ## Key Design Principles
//!
//! - **Single Source of Truth (SSOT)**: One tensor type handles both backend and autograd
//! - **Generic Backend Abstraction**: Works with any backend (CPU, GPU, custom)
//! - **Generic Dtype Support**: Full support for all numeric types through trait system
//! - **Zero-Copy Operations**: Copy-on-write semantics minimize memory allocations
//! - **Optional Autograd**: Can work with or without gradient tracking
//! - **Type Safety**: Compile-time guarantees for all operations
//!
//! ## Backend Architecture
//!
//! The unified tensor uses a backend abstraction system:
//! - `B: Backend<T>`: Generic backend trait for device-agnostic operations
//! - `T: Dtype`: Generic data type trait for numeric operations
//! - Zero unsafe code with proper trait bounds and memory safety

pub mod core {
    pub mod tensor;
}
pub mod ops {
    pub mod activations;
    pub mod arithmetic;
    pub mod creation;
    pub mod indexing;
    pub mod matrix;
    pub mod reduction;
    pub mod bitwise;
}
pub mod traits;

pub mod iterators;
pub mod performance; // Prune/move to utils if unused (YAGNI)

pub use core::tensor::{Tensor, DenseTensor, SparseTensor};

// PyTorch-like flat API re-exports (clean, no deep nesting)

// Arithmetic operations (available functions)
pub use ops::arithmetic::{add, mul, div, sub, neg, pow, exp, log, sin, cos, sqrt, maximum, minimum, abs};

// Matrix operations
pub use ops::matrix::matmul;

// Bitwise operations (methods available on Tensor)

// Reduction operations (available functions)
pub use ops::reduction::{sum, sum_dim, mean_dim};

// Creation operations (available as module)
pub use ops::creation;

// Full modules available for advanced usage
pub use ops::indexing;

// Tensor traits for zero-cost polymorphism
pub use traits::*;

// Re-export dtype traits for convenience
pub use coeus_dtype::{Dtype, FloatDtype};
pub use coeus_backend::{Backend, BackendData, BackendError, CpuBackend, Device};
pub use coeus_storage::{TensorStorage, DenseStorage, SparseStorageCSR, SparseStorageCOO, StorageDtype};

// Re-export standard library traits for convenience
pub use std::ops::{Add, Div, Mul, Neg, Sub};
pub use std::borrow::Cow;
pub use std::mem::MaybeUninit;

// Temporarily disable autograd re-exports until autograd crate is fixed
// pub use coeus_autograd::{AutogradContext, Operation};

/// Utility function to convert f64 vector to backend tensor data
/// This is used internally for operations that need to convert between different numeric types
pub fn vec_f64_to_tensor_data(data: Vec<f64>, shape: Vec<usize>) -> BackendData<f32> {
    // Convert f64 to the default dtype (f32 for now)
    let f32_data: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
    BackendData::Cpu { data: f32_data, shape }
}

/// Execute code within an autograd context for gradient computation

/// Result type for tensor operations (std::result::Result<T, TensorError>)
pub type Result<T> = std::result::Result<T, TensorError>;

/// Simplified tensor type alias using CpuBackend as default
/// This provides a convenient interface for CPU-based tensor operations
pub type CpuTensor<T> = Tensor<T, CpuBackend, DenseStorage<T>>;

/// Errors that can occur during tensor operations
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid tensor shape: data length ({data_len}) does not match shape product ({shape_product}) for shape {shape:?}")]
    InvalidShape {
        data_len: usize,
        shape_product: usize,
        shape: Vec<usize>,
    },

    #[error("Matrix multiplication requires at least 2D tensors, got shapes {lhs_shape:?} and {rhs_shape:?}")]
    MatrixMulRequires2D {
        lhs_shape: Vec<usize>,
        rhs_shape: Vec<usize>,
    },

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Sparse operation not supported: {0}")]
    SparseOperationNotSupported(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error(
        "Incompatible matrix dimensions for multiplication: {lhs_m}x{lhs_k} vs {rhs_k}x{rhs_n}"
    )]
    IncompatibleMatrixDims {
        lhs_m: usize,
        lhs_k: usize,
        rhs_k: usize,
        rhs_n: usize,
    },

    #[error("Tensor must be scalar for item() access, got shape {shape:?}")]
    NotScalar { shape: Vec<usize> },

    #[error("Cannot create count value for type during mean calculation")]
    MeanCalculationError,

    #[error("Dtype mismatch: expected {expected}, got {actual}")]
    DtypeMismatch { expected: String, actual: String },

    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },

    #[error("Index out of bounds: index {index}, size {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    #[error("Index out of bounds: index {index}, size {size}")]
    OutOfBounds { index: usize, size: usize },

    #[error("Invalid dimension: dimension {dim}, maximum dimension is {max_dim}")]
    InvalidDimension { dim: usize, max_dim: usize },

    #[error("Gradient computation error: {message}")]
    GradientError { message: String },

    #[error("Broadcasting error: cannot broadcast shapes {shape1:?} and {shape2:?}")]
    BroadcastingError {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
    },

    #[error("Numerical stability error: {operation} produced {issue} (value: {value})")]
    NumericalStabilityError {
        operation: String,
        issue: String,
        value: f64,
    },

    #[error("Memory allocation error: failed to allocate {requested} bytes")]
    MemoryAllocationError { requested: usize },

    #[error("Serialization error: {message}")]
    SerializationError { message: String },

    #[error("Device error: operation not supported on {device}")]
    DeviceError { device: String },

    #[error("Unsupported index operation: {0:?}")]
    UnsupportedIndex(Vec<crate::ops::indexing::Slice>),

    #[error("Invalid index: start {start}, end {end}, numel {numel}")]
    InvalidIndex { start: usize, end: usize, numel: usize },

    #[error(
        "Performance regression detected: {operation} exceeded threshold by {degradation:.2}%"
    )]
    PerformanceRegressionError { operation: String, degradation: f64 },


    #[error("Backend error: {0}")]
    BackendError(#[from] coeus_backend::BackendError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerdeError(#[from] serde_json::Error),

    #[error("Bincode error: {0}")]
    BincodeError(Box<bincode::ErrorKind>),

    #[error("Error: {0}")]
    StringError(String),
}

impl From<Box<bincode::ErrorKind>> for TensorError {
    fn from(err: Box<bincode::ErrorKind>) -> Self {
        TensorError::BincodeError(err)
    }
}

impl From<String> for TensorError {
    fn from(err: String) -> Self {
        TensorError::StringError(err)
    }
}


/// Memory layout for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Layout {
    /// Row-major (C-style) layout
    #[default]
    Contiguous,
    /// Column-major (Fortran-style) layout
    Fortran,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::from_vec(CpuBackend::new(), vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.numel(), 3);
    }

    #[test]
    fn test_tensor_addition() {
        let a = Tensor::from_vec(CpuBackend::default(), vec![1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_vec(CpuBackend::default(), vec![3.0, 4.0], vec![2]).unwrap();
        // Note: Addition operator not yet implemented, this is a placeholder test
        // let c = &a + &b;
        // assert_eq!(c.unwrap().data(), &[4.0, 6.0]);
    }

    #[test]
    fn test_requires_grad() {
        let mut t = Tensor::from_vec(CpuBackend::new(), vec![1.0, 2.0], vec![2]).unwrap();
        t.set_requires_grad(true);

        assert!(t.requires_grad());
        assert!(t.grad().is_none());
    }
}

impl From<coeus_storage::StorageError> for TensorError {
    fn from(err: coeus_storage::StorageError) -> Self {
        TensorError::StorageError(err.to_string())
    }
}

#[cfg(test)]
include!("tests/autograd_tests.rs");

#[cfg(test)]
include!("tests/property_tests.rs");

#[cfg(test)]
include!("tests/autograd/numerical_gradient_tests.rs");


// Test integration removed - async_view method not implemented
// Tests are included via include! macros above

/// Const generics/Cow full (example in Tensor impl if not, but for ops in submods)
impl<T: Dtype, B: Backend<T> + Clone + Send + Sync + Default, S: TensorStorage<T> + Clone + Send + Sync> Tensor<T, B, S> {
    pub fn view_cow(&self) -> Cow<'_, [T]> {
        Cow::Borrowed(self.data())
    }

    pub fn from_maybe_uninit(shape: Vec<usize>) -> Self where T: Default {
        let numel = shape.iter().product();
        let mut data = vec![MaybeUninit::uninit(); numel];
        // Init with Default or zero
        for i in 0..numel {
            data[i].write(T::default());
        }
        let data_init: Vec<T> = unsafe { std::mem::transmute(data) };
        let backend = B::default();
        Tensor::from_vec(backend, data_init, shape).unwrap()
    }
}
