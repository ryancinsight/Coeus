//! # Coeus Tensor
//!
//! PyTorch-like tensor library with automatic differentiation, backend abstraction, and higher-order derivatives.
//!
//! This crate provides a unified tensor architecture that consolidates all tensor forms
//! into a single type with simplified generic backend abstraction.
//!
//! ## Architecture
//!
//! The tensor library provides a single unified tensor type that works across all backends
//! and supports both regular operations and automatic differentiation:
//!
//! ### Simplified Tensor API
//! ```rust
//! use tensor::Tensor;
//! use backend::CpuBackend;
//! use dtype::float::Float32;
//! use storage::DenseStorage;
//!
//! let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
//!     vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
//!     &[3]
//! ).unwrap();
//!
//! // Operations work like PyTorch with zero-copy semantics
//! let result = &tensor + &tensor;
//!
//! // Enable autograd for gradient computation
//! let autograd_tensor = tensor.requires_grad_(true);
//! let grad_result = -&autograd_tensor; // Supports gradient computation
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
//! The simplified tensor uses a backend abstraction with associated types:
//! - `B: Backend<Data = T, Device = D>`: Backend trait with associated data and device types
//! - Generic methods support any storage type (dense/sparse) through trait bounds
//! - Zero unsafe code with proper trait bounds and memory safety
//! - Complete sparse tensor support through generic operations

// Module declarations
extern crate alloc;

pub mod ops;

// Temporarily disabled due to alloc issues
pub mod functions;
pub mod minimal_tensor;
pub mod tensor_backend_dispatch;
pub mod tensor_backend_integration_tests;
pub mod tensor_core;
pub mod tensor_impl;

// Additional modules
pub mod elementwise;
pub mod error;
pub mod indexing;
pub mod shape_ops;

// Re-export full tensor implementation
pub use tensor_core::{AsAny, DifferentiableFunction, Function, OperationName, Tensor};

// Re-export error types and utilities

// Re-export storage utilities
pub use storage::{Shape, StorageFromVec, StorageToDense};

// Minimal API for testing - full API to be implemented later

// Advanced zero-copy optimizations - temporarily disabled
// pub mod zero_copy;
// pub mod simd_ops;

// Re-export dtype traits for convenience
pub use backend::{Backend, BackendError, Device};
pub use dtype::float::Float32;
pub use dtype::{traits::FloatExt, DataType};
pub use storage::{CooStorage, CscStorage, CsrStorage, DenseStorage, Storage};

// Re-export CpuBackend with Float32 default for convenience
pub use backend::CpuBackend;

// Result type for tensor operations
pub type Result<T> = std::result::Result<T, TensorError>;

// Re-export convenience functions from ops::creation
pub use ops::creation::{cat, randn};

/// Creates a thread-safe gradient storage container
pub fn grad_rwlock<T>(value: T) -> std::sync::RwLock<T> {
    std::sync::RwLock::new(value)
}

// Re-export error types
pub use error::TensorError;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::arithmetic::*;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;

    // ===== TENSOR CREATION TESTS =====

    #[test]
    fn test_tensor_creation_from_vec() {
        let data = vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ];
        let tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data, &[2, 2])
                .unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 2]);
        assert_eq!(tensor.len(), 4);
        assert!(!tensor.is_empty());
    }

    #[test]
    fn test_tensor_creation_from_vec_with_backend() {
        let data = vec![Float32::new(1.0), Float32::new(2.0)];
        let backend = CpuBackend::default();
        let tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
                data,
                &[2],
                backend,
            )
            .unwrap();
        assert_eq!(tensor.shape().dims(), &[2]);
        assert_eq!(tensor.len(), 2);
    }

    #[test]
    fn test_tensor_creation_zeros() {
        let tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[3, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[3, 3]);
        assert_eq!(tensor.len(), 9);
        // Check all elements are zero
        for &val in tensor.as_slice() {
            assert_eq!(val.get(), 0.0);
        }
    }

    #[test]
    fn test_tensor_creation_ones() {
        let tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 3]).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.len(), 6);
        // Check all elements are one
        for &val in tensor.as_slice() {
            assert_eq!(val.get(), 1.0);
        }
    }

    #[test]
    fn test_tensor_creation_from_slice() {
        let data = [Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_slice(&data, &[3])
                .unwrap();
        assert_eq!(tensor.shape().dims(), &[3]);
        assert_eq!(tensor.as_slice(), &data);
    }

    // ===== ARITHMETIC OPERATIONS TESTS =====

    #[test]
    fn test_tensor_arithmetic_add() {
        let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();
        let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(3.0), Float32::new(4.0)],
            &[2],
        )
        .unwrap();

        let result = add(&a, &b).unwrap();
        let expected = [Float32::new(4.0), Float32::new(6.0)];
        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_tensor_arithmetic_sub() {
        let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(5.0), Float32::new(7.0)],
            &[2],
        )
        .unwrap();
        let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(3.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();

        let result = sub(&a, &b).unwrap();
        let expected = [Float32::new(2.0), Float32::new(5.0)];
        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_tensor_arithmetic_mul() {
        let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(2.0), Float32::new(3.0)],
            &[2],
        )
        .unwrap();
        let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(4.0), Float32::new(5.0)],
            &[2],
        )
        .unwrap();

        let result = mul(&a, &b).unwrap();
        let expected = [Float32::new(8.0), Float32::new(15.0)];
        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_tensor_arithmetic_div() {
        let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(8.0), Float32::new(15.0)],
            &[2],
        )
        .unwrap();
        let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(4.0), Float32::new(5.0)],
            &[2],
        )
        .unwrap();

        let result = div(&a, &b).unwrap();
        let expected = [Float32::new(2.0), Float32::new(3.0)];
        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_tensor_arithmetic_neg() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(-2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let result = neg(&tensor).unwrap();
        let expected = [Float32::new(-1.0), Float32::new(2.0), Float32::new(-3.0)];
        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_tensor_arithmetic_maximum() {
        let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(4.0), Float32::new(2.0)],
            &[3],
        )
        .unwrap();
        let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(3.0), Float32::new(2.0), Float32::new(5.0)],
            &[3],
        )
        .unwrap();

        let result = maximum(&a, &b).unwrap();
        let expected = [Float32::new(3.0), Float32::new(4.0), Float32::new(5.0)];
        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_tensor_arithmetic_minimum() {
        let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(4.0), Float32::new(2.0)],
            &[3],
        )
        .unwrap();
        let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(3.0), Float32::new(2.0), Float32::new(5.0)],
            &[3],
        )
        .unwrap();

        let result = minimum(&a, &b).unwrap();
        let expected = [Float32::new(1.0), Float32::new(2.0), Float32::new(2.0)];
        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_tensor_arithmetic_pow() {
        let base = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(2.0), Float32::new(3.0)],
            &[2],
        )
        .unwrap();
        let exponent = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(3.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();

        let result = pow(&base, &exponent).unwrap();
        let expected = [Float32::new(8.0), Float32::new(9.0)]; // 2^3 = 8, 3^2 = 9
        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_tensor_arithmetic_abs() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-1.0), Float32::new(2.0), Float32::new(-3.0)],
            &[3],
        )
        .unwrap();

        let result = abs(&tensor).unwrap();
        let expected = [Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        assert_eq!(result.as_slice(), &expected[..]);
    }

    // ===== MATRIX OPERATIONS TESTS =====

    #[test]
    fn test_tensor_matrix_multiplication() {
        let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
            ],
            &[2, 2],
        )
        .unwrap();
        let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(5.0),
                Float32::new(6.0),
                Float32::new(7.0),
                Float32::new(8.0),
            ],
            &[2, 2],
        )
        .unwrap();

        let result = a.matmul(&b).unwrap();
        assert_eq!(result.shape().dims(), &[2, 2]);
        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        let expected = [
            Float32::new(19.0),
            Float32::new(22.0),
            Float32::new(43.0),
            Float32::new(50.0),
        ];
        assert_eq!(result.as_slice(), &expected[..]);
    }

    // ===== REDUCTION OPERATIONS TESTS =====

    #[test]
    fn test_tensor_reductions_sum_all() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
            ],
            &[2, 2],
        )
        .unwrap();

        let result = tensor.sum_all();
        assert_eq!(result.shape().dims(), &[1]);
        assert_eq!(result.as_slice()[0].get(), 10.0);
    }

    #[test]
    fn test_tensor_reductions_mean_all() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
            ],
            &[2, 2],
        )
        .unwrap();

        let result = tensor.mean_all();
        assert_eq!(result.shape().dims(), &[1]);
        assert_eq!(result.as_slice()[0].get(), 2.5);
    }

    #[test]
    fn test_tensor_reductions_sum_dims() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
                Float32::new(5.0),
                Float32::new(6.0),
                Float32::new(7.0),
                Float32::new(8.0),
            ],
            &[2, 4],
        )
        .unwrap();

        // Sum along dimension 0 (rows)
        let result = tensor.sum_dims(Some(&[0]), false).unwrap();
        assert_eq!(result.shape().dims(), &[4]);
        let expected = [
            Float32::new(6.0),
            Float32::new(8.0),
            Float32::new(10.0),
            Float32::new(12.0),
        ]; // [1+5, 2+6, 3+7, 4+8]
        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_tensor_reductions_mean_dims() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.0),
                Float32::new(2.0),
                Float32::new(3.0),
                Float32::new(4.0),
                Float32::new(5.0),
                Float32::new(6.0),
                Float32::new(7.0),
                Float32::new(8.0),
            ],
            &[2, 4],
        )
        .unwrap();

        // Mean along dimension 1 (columns)
        let result = tensor.mean_dims(Some(&[1]), false).unwrap();
        assert_eq!(result.shape().dims(), &[2]);
        let expected = [Float32::new(2.5), Float32::new(6.5)]; // [(1+2+3+4)/4, (5+6+7+8)/4]
        assert_eq!(result.as_slice(), &expected[..]);
    }

    // ===== TENSOR METHODS TESTS =====

    #[test]
    fn test_tensor_properties() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        assert_eq!(tensor.len(), 3);
        assert!(!tensor.is_empty());
        assert_eq!(tensor.shape().dims(), &[3]);
    }

    #[test]
    fn test_tensor_scalar_operations() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        // Scalar multiplication
        let result = tensor.mul_scalar(Float32::new(2.0)).unwrap();
        let expected = [Float32::new(2.0), Float32::new(4.0), Float32::new(6.0)];
        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_tensor_clamp() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.5), Float32::new(1.5), Float32::new(2.5)],
            &[3],
        )
        .unwrap();

        let result = tensor.clamp(Float32::new(1.0), Float32::new(2.0)).unwrap();
        let expected = [Float32::new(1.0), Float32::new(1.5), Float32::new(2.0)];
        assert_eq!(result.as_slice(), &expected[..]);
    }

    #[test]
    fn test_tensor_clone() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();

        let cloned = tensor.clone();
        assert_eq!(tensor.shape().dims(), cloned.shape().dims());
        assert_eq!(tensor.len(), cloned.len());
        assert_eq!(tensor.as_slice(), cloned.as_slice());
    }

    #[test]
    fn test_tensor_debug_display() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();

        let debug_str = format!("{:?}", tensor);
        assert!(debug_str.contains("Tensor"));
        assert!(debug_str.contains("[2]"));
    }

    // ===== ERROR HANDLING TESTS =====

    #[test]
    fn test_tensor_error_handling() {
        // Test invalid shape (wrong number of elements)
        let result = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2, 2], // Needs 4 elements, only has 2
        );
        assert!(result.is_err());

        // Test invalid shape (zero dimension)
        let result = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0)],
            &[0, 1], // Zero dimension
        );
        assert!(result.is_err());
    }

    // ===== BROADCASTING TESTS =====

    #[test]
    fn test_tensor_broadcasting_scalar() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();
        let scalar = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(10.0)],
            &[1],
        )
        .unwrap();

        let result = add(&tensor, &scalar).unwrap();
        let expected = [Float32::new(11.0), Float32::new(12.0)];
        assert_eq!(result.as_slice(), &expected[..]);
    }

    // ===== ELEMENTWISE OPERATIONS TESTS =====

    #[test]
    fn test_tensor_elementwise_sqrt() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(4.0), Float32::new(9.0)],
            &[3],
        )
        .unwrap();

        // Test through elementwise operations (if available)
        // This would test the elementwise module functions
        let result = pow(
            &tensor,
            &Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
                vec![Float32::new(0.5), Float32::new(0.5), Float32::new(0.5)],
                &[3],
            )
            .unwrap(),
        )
        .unwrap();

        let expected = [Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        for (i, &val) in result.as_slice().iter().enumerate() {
            assert!((val.get() - expected[i].get()).abs() < 1e-6);
        }
    }

    // ===== GRADIENT AND AUTODIFF TESTS =====

    #[test]
    fn test_tensor_autograd_creation() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();

        // Test that tensor can be created without gradients
        // grad() returns an error if no gradient is set
        assert!(tensor.grad().is_err());
        assert!(tensor.grad_fn().is_none());
    }

    #[test]
    fn test_tensor_gradient_accumulation() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();

        // Set a gradient
        let grad = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.1), Float32::new(0.2)],
            &[2],
        )
        .unwrap();

        tensor.set_grad(grad).unwrap();
        let retrieved_grad = tensor.grad().unwrap();
        assert_eq!(retrieved_grad.as_slice()[0].get(), 0.1);
        assert_eq!(retrieved_grad.as_slice()[1].get(), 0.2);
    }

    #[test]
    fn test_tensor_zero_grad() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();

        // Set a gradient first
        let grad = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.1), Float32::new(0.2)],
            &[2],
        )
        .unwrap();
        tensor.set_grad(grad).unwrap();

        // Zero gradients
        tensor.zero_grad().unwrap();

        // Check that gradient access now fails (since zero_grad clears the gradient)
        assert!(tensor.grad().is_err());
    }

    // ===== EDGE CASES =====

    #[test]
    fn test_tensor_empty_creation() {
        // Test that we can't create a tensor with wrong shape (more elements than shape allows)
        let result = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[1], // Shape only allows 1 element, but we provide 2
        );
        assert!(result.is_err()); // Should fail with shape mismatch
    }

    #[test]
    fn test_tensor_single_element() {
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(42.0)],
            &[1],
        )
        .unwrap();
        assert_eq!(tensor.shape().dims(), &[1]);
        assert_eq!(tensor.len(), 1);
        assert_eq!(tensor.as_slice()[0].get(), 42.0);
    }

    #[test]
    fn test_tensor_large_dimensions() {
        let tensor =
            Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 3, 4])
                .unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3, 4]);
        assert_eq!(tensor.len(), 24);
        // All elements should be zero
        for &val in tensor.as_slice() {
            assert_eq!(val.get(), 0.0);
        }
    }
}
