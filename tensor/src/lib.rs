//! # Coeus Tensor
//!
//! PyTorch-like tensor library with automatic differentiation and higher-order derivatives.
//!
//! This crate provides:
//! - Tensor operations with operator overloads (`+`, `-`, `*`, `/`)
//! - Automatic differentiation with `requires_grad`
//! - Higher-order derivatives (Hessian matrices) for second-order optimization
//! - Iterator support with gradient flow and Hessian traversal
//! - Comprehensive mathematical operations
//! - GPU-ready architecture with SIMD acceleration
//! - Performance monitoring and regression detection
//! - Memory-efficient operations with zero-copy where possible
//!
//! ## Higher-Order Derivatives (Hessian Computation)
//!
//! The library supports second-order automatic differentiation through Hessian matrix computation:
//!
//! ### Basic Hessian Computation
//! ```rust
//! use coeus_tensor::Tensor;
//!
//! let x = Tensor::scalar(2.0);
//! let hessian = x.hessian().unwrap();
//!
//! // For f(x) = x², the Hessian is [[2.0]]
//! assert_eq!(hessian[0][0], 2.0);
//! ```
//!
//! ### Hessian Iterator Pattern
//! ```rust
//! use coeus_tensor::Tensor;
//!
//! let x = Tensor::scalar(1.0);
//! let mut hessian_iter = x.hessian_iter().unwrap();
//!
//! // Iterate through Hessian matrix elements
//! for ((row, col), value) in hessian_iter {
//!     println!("Hessian[{}][{}] = {}", row, col, value);
//! }
//! ```
//!
//! ### Cross-Hessian Computation
//! *Note: Cross-Hessian computation between different tensors is planned for future implementation*
//!
//! ## Numerical Methods and Accuracy
//!
//! The Hessian computation uses finite difference methods with configurable precision:
//! - Central differences for improved accuracy
//! - Small step sizes (default: 1e-5) for numerical stability
//! - Comprehensive validation against analytical derivatives
//!
//! ## Applications
//!
//! Higher-order derivatives are essential for:
//! - **Newton's Method**: Second-order optimization algorithms
//! - **Natural Gradients**: Improved convergence in machine learning
//! - **Hessian-free Methods**: Memory-efficient second-order optimization
//! - **Uncertainty Quantification**: Statistical analysis of gradients
//! - **Bayesian Optimization**: Modeling parameter uncertainty

pub mod arithmetic_ops;
pub mod core;
pub mod iterators;
pub mod ops;
pub mod performance;
pub mod serialization;

pub use coeus_dtype::{
    DataContainer, DataType, Dtype, FloatDtype, IntDtype, NumericDtype, QuantizedDtype,
};
pub use core::{apply_pending_gradients, store_pending_gradient, with_autograd_context, Tensor};
pub use ops::*;

// Re-export standard library traits for convenience
pub use std::ops::{Add, Div, Mul, Neg, Sub};

/// Result type for tensor operations
pub type Result<T> = std::result::Result<T, TensorError>;

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

    #[error("Invalid dimension: dimension {dim}, maximum dimension is {max_dim}")]
    InvalidDimension { dim: usize, max_dim: usize },

    #[error("Gradient computation error: {message}")]
    GradientError { message: String },

    #[error("Autograd error: {0}")]
    AutogradError(#[from] coeus_autograd::AutogradError),

    #[error("Backend error: {0}")]
    BackendError(#[from] coeus_backend::BackendError),
}

/// Global device type (for future GPU support)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Device {
    #[default]
    Cpu,
    // Cuda, // Future GPU support
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
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.numel(), 3);
    }

    #[test]
    fn test_tensor_addition() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);
        let c = &a + &b;

        assert_eq!(c.unwrap().data(), &[4.0, 6.0]);
    }

    #[test]
    fn test_requires_grad() {
        let mut t = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        t.set_requires_grad(true);

        assert!(t.requires_grad());
        assert!(t.grad().is_none());
    }
}

#[cfg(test)]
include!("tests/autograd_tests.rs");
