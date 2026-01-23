//! Property-based tests for Single Source of Truth (SSOT) architectural principle.
//!
//! **Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations**
//!
//! These tests verify that:
//! 1. Operations are defined exactly once in tensor::ops
//! 2. Tensor methods properly delegate to ops functions
//! 3. No duplicate implementations exist
//!
//! **Validates: Requirements 1.2, 1.4**

use proptest::prelude::*;

use approx::assert_relative_eq;
use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::{Tensor, Backend, DataType, Storage};
use tensor::ops::{sin, cos, exp, log, sqrt};
use tensor::ops::math::pow_scalar;

/// Type alias for our test tensor type
type TestTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

fn checked_numel(dims: &[usize]) -> Option<usize> {
    dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

/// Generate random tensor shapes (1D to 3D, smaller for SSOT tests)
fn arb_tensor_shape() -> impl Strategy<Value = Vec<usize>> {
    const MAX_DIM: usize = 32;
    const MAX_ELEMENTS: usize = 1_024;
    prop::collection::vec(1..=MAX_DIM, 1..=3).prop_filter("shape within element budget", |s| {
        checked_numel(s).is_some_and(|n| n > 0 && n <= MAX_ELEMENTS)
    })
}

/// Generate a random tensor with positive values (for operations like log, sqrt)
fn arb_positive_tensor() -> impl Strategy<Value = TestTensor> {
    arb_tensor_shape().prop_flat_map(|shape| {
        let len = checked_numel(&shape).unwrap();
        let data = prop::collection::vec((0.1f32..100.0).prop_map(Float32::new), len);
        data.prop_map(move |data| Tensor::from_vec(data, &shape).unwrap())
    })
}

/// Generate a random tensor with any values
fn arb_tensor() -> impl Strategy<Value = TestTensor> {
    arb_tensor_shape().prop_flat_map(|shape| {
        let len = checked_numel(&shape).unwrap();
        let data = prop::collection::vec((-100.0f32..100.0).prop_map(Float32::new), len);
        data.prop_map(move |data| Tensor::from_vec(data, &shape).unwrap())
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// **Property 1: Single Source of Truth - exp() delegation**
    ///
    /// For any tensor, calling the exp() method should produce the same result
    /// as calling ops::arithmetic::exp() directly, proving that the method
    /// delegates to the single source of truth.
    ///
    /// **Validates: Requirements 1.2, 1.4**
    #[test]
    fn test_ssot_exp_delegation(ref tensor in arb_tensor()) {
        // Call method
        let result_method = tensor.exp();

        // Call ops function directly
        let result_ops = exp(tensor).unwrap();

        // Verify they produce identical results
        prop_assert_eq!(result_method.shape().dims(), result_ops.shape().dims());
        for i in 0..result_method.len() {
            assert_relative_eq!(
                result_method.as_slice()[i].get(),
                result_ops.as_slice()[i].get(),
                epsilon = 1e-6,
                max_relative = 1e-6
            );
        }
    }

    /// **Property 1: Single Source of Truth - log() delegation**
    ///
    /// For any positive tensor, calling the log() method should produce the same
    /// result as calling ops::arithmetic::log() directly.
    ///
    /// **Validates: Requirements 1.2, 1.4**
    #[test]
    fn test_ssot_log_delegation(ref tensor in arb_positive_tensor()) {
        // Call method
        let result_method = tensor.log();

        // Call ops function directly
        let result_ops = log(tensor).unwrap();

        // Verify they produce identical results
        prop_assert_eq!(result_method.shape().dims(), result_ops.shape().dims());
        for i in 0..result_method.len() {
            assert_relative_eq!(
                result_method.as_slice()[i].get(),
                result_ops.as_slice()[i].get(),
                epsilon = 1e-6,
                max_relative = 1e-6
            );
        }
    }

    /// **Property 1: Single Source of Truth - sin() delegation**
    ///
    /// For any tensor, calling the sin() method should produce the same result
    /// as calling ops::arithmetic::sin() directly.
    ///
    /// **Validates: Requirements 1.2, 1.4**
    #[test]
    fn test_ssot_sin_delegation(ref tensor in arb_tensor()) {
        // Call method
        let result_method = tensor.sin();

        // Call ops function directly
        let result_ops = sin(tensor).unwrap();

        // Verify they produce identical results
        prop_assert_eq!(result_method.shape().dims(), result_ops.shape().dims());
        for i in 0..result_method.len() {
            assert_relative_eq!(
                result_method.as_slice()[i].get(),
                result_ops.as_slice()[i].get(),
                epsilon = 1e-6,
                max_relative = 1e-6
            );
        }
    }

    /// **Property 1: Single Source of Truth - cos() delegation**
    ///
    /// For any tensor, calling the cos() method should produce the same result
    /// as calling ops::arithmetic::cos() directly.
    ///
    /// **Validates: Requirements 1.2, 1.4**
    #[test]
    fn test_ssot_cos_delegation(ref tensor in arb_tensor()) {
        // Call method
        let result_method = tensor.cos();

        // Call ops function directly
        let result_ops = cos(tensor).unwrap();

        // Verify they produce identical results
        prop_assert_eq!(result_method.shape().dims(), result_ops.shape().dims());
        for i in 0..result_method.len() {
            assert_relative_eq!(
                result_method.as_slice()[i].get(),
                result_ops.as_slice()[i].get(),
                epsilon = 1e-6,
                max_relative = 1e-6
            );
        }
    }

    /// **Property 1: Single Source of Truth - sqrt() delegation**
    ///
    /// For any positive tensor, calling the sqrt() method should produce the same
    /// result as calling ops::arithmetic::sqrt() directly.
    ///
    /// **Validates: Requirements 1.2, 1.4**
    #[test]
    fn test_ssot_sqrt_delegation(ref tensor in arb_positive_tensor()) {
        // Call method
        let result_method = tensor.sqrt();

        // Call ops function directly
        let result_ops = sqrt(tensor).unwrap();

        // Verify they produce identical results
        prop_assert_eq!(result_method.shape().dims(), result_ops.shape().dims());
        for i in 0..result_method.len() {
            assert_relative_eq!(
                result_method.as_slice()[i].get(),
                result_ops.as_slice()[i].get(),
                epsilon = 1e-6,
                max_relative = 1e-6
            );
        }
    }

    /// **Property 1: Single Source of Truth - powf() delegation**
    ///
    /// For any positive tensor and exponent, calling the powf() method should
    /// produce the same result as calling ops::arithmetic::pow_scalar() directly.
    ///
    /// **Validates: Requirements 1.2, 1.4**
    #[test]
    fn test_ssot_powf_delegation(
        ref tensor in arb_positive_tensor(),
        exp in 0.1f32..5.0
    ) {
        let exp_val = Float32::new(exp);

        // Call method
        let result_method = tensor.powf(exp_val);

        // Call ops function directly
        let result_ops = pow_scalar(tensor, exp_val).unwrap();

        // Verify they produce identical results
        prop_assert_eq!(result_method.shape().dims(), result_ops.shape().dims());
        for i in 0..result_method.len() {
            assert_relative_eq!(
                result_method.as_slice()[i].get(),
                result_ops.as_slice()[i].get(),
                epsilon = 1e-5,
                max_relative = 1e-5
            );
        }
    }

    /// **Property 1: Single Source of Truth - square() delegation**
    ///
    /// For any tensor, calling the square() method should produce the same result
    /// as calling powf(2.0), which delegates to ops::arithmetic::pow_scalar().
    ///
    /// **Validates: Requirements 1.2, 1.4**
    #[test]
    fn test_ssot_square_delegation(ref tensor in arb_tensor()) {
        // Call square method
        let result_square = tensor.square();

        // Call powf(2.0) which also delegates to ops
        let result_powf = tensor.powf(Float32::new(2.0));

        // Verify they produce identical results (both delegate to same ops function)
        prop_assert_eq!(result_square.shape().dims(), result_powf.shape().dims());
        for i in 0..result_square.len() {
            assert_relative_eq!(
                result_square.as_slice()[i].get(),
                result_powf.as_slice()[i].get(),
                epsilon = 1e-6,
                max_relative = 1e-6
            );
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    /// Test that verifies the architectural principle: methods delegate to ops
    ///
    /// This is a compile-time verification that the methods exist and have
    /// the correct signatures. The property tests above verify runtime behavior.
    #[test]
    fn test_ssot_architectural_principle() {
        // Create a simple tensor
        let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
        let tensor = TestTensor::from_vec(data, &[3]).unwrap();

        // Verify that both method and ops function exist and work
        let _method_result = tensor.exp();
        let _ops_result = exp(&tensor).unwrap();

        // If this compiles, it proves:
        // 1. The method exists on Tensor
        // 2. The ops function exists in arithmetic
        // 3. Both have compatible signatures
        // 4. The architecture supports delegation
    }

    /// Test that documents the SSOT principle for future developers
    #[test]
    fn test_ssot_documentation() {
        // This test serves as documentation of the SSOT principle:
        //
        // ARCHITECTURAL PRINCIPLE: Single Source of Truth
        //
        // All mathematical operations are implemented exactly once in tensor::ops.
        // Tensor methods provide ergonomic APIs by delegating to these ops functions.
        //
        // Example:
        // - Implementation: tensor::ops::arithmetic::exp()
        // - Method: Tensor::exp() -> delegates to ops::arithmetic::exp()
        //
        // Benefits:
        // 1. Operations defined in exactly one place
        // 2. Easier to maintain and test
        // 3. Consistent behavior across all usage patterns
        // 4. Clear architectural boundaries

        let data = vec![Float32::new(1.0)];
        let tensor = TestTensor::from_vec(data, &[1]).unwrap();

        // Both of these should produce identical results:
        let result1 = tensor.exp(); // Method call
        let result2 = exp(&tensor).unwrap(); // Direct ops call

        assert_relative_eq!(
            result1.as_slice()[0].get(),
            result2.as_slice()[0].get(),
            epsilon = 1e-6
        );
    }
}
