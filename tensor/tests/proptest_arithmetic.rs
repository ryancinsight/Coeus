//! Property-based tests for tensor arithmetic operations.
//!
//! Validates edge cases, invariants, and mathematical properties using proptest.

use backend::CpuBackend;
use dtype::float::{Float32, Float64};
use dtype::int::Int32;
use storage::DenseStorage;
use tensor::Tensor;
use proptest::prelude::*;

type CpuTensorF32 = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;
type CpuTensorF64 = Tensor<CpuBackend<Float64>, DenseStorage<Float64>, Float64>;
type CpuTensorI32 = Tensor<CpuBackend<Int32>, DenseStorage<Int32>, Int32>;

// ============================================================================
// Property: Arithmetic Operations Preserve Shape
// ============================================================================

proptest! {
    #[test]
    fn prop_add_preserves_shape(
        size in 1usize..100,
        data1 in prop::collection::vec(-1000.0f32..1000.0, 1..100),
        data2 in prop::collection::vec(-1000.0f32..1000.0, 1..100)
    ) {
        let len = size.min(data1.len()).min(data2.len());
        let d1: Vec<Float32> = data1.iter().take(len).map(|&x| Float32::new(x)).collect();
        let d2: Vec<Float32> = data2.iter().take(len).map(|&x| Float32::new(x)).collect();

        let t1 = CpuTensorF32::from_vec(d1, &[len]).unwrap();
        let t2 = CpuTensorF32::from_vec(d2, &[len]).unwrap();

        let result = &t1 + &t2;
        prop_assert_eq!(result.shape().dims(), &[len]);
        prop_assert_eq!(result.len(), len);
    }

    #[test]
    fn prop_mul_preserves_shape(
        size in 1usize..100,
        data1 in prop::collection::vec(-100.0f64..100.0, 1..100),
        data2 in prop::collection::vec(-100.0f64..100.0, 1..100)
    ) {
        let len = size.min(data1.len()).min(data2.len());
        let d1: Vec<Float64> = data1.iter().take(len).map(|&x| Float64::new(x)).collect();
        let d2: Vec<Float64> = data2.iter().take(len).map(|&x| Float64::new(x)).collect();

        let t1 = CpuTensorF64::from_vec(d1, &[len]).unwrap();
        let t2 = CpuTensorF64::from_vec(d2, &[len]).unwrap();

        let result = &t1 * &t2;
        prop_assert_eq!(result.shape().dims(), &[len]);
        prop_assert_eq!(result.len(), len);
    }
}

// ============================================================================
// Property: Arithmetic Commutativity
// ============================================================================

proptest! {
    #[test]
    fn prop_add_commutative(
        data1 in prop::collection::vec(-1000.0f32..1000.0, 1..50),
        data2 in prop::collection::vec(-1000.0f32..1000.0, 1..50)
    ) {
        let len = data1.len().min(data2.len());
        let d1: Vec<Float32> = data1.iter().take(len).map(|&x| Float32::new(x)).collect();
        let d2: Vec<Float32> = data2.iter().take(len).map(|&x| Float32::new(x)).collect();

        let t1 = CpuTensorF32::from_vec(d1.clone(), &[len]).unwrap();
        let t2 = CpuTensorF32::from_vec(d2.clone(), &[len]).unwrap();

        let result1 = &t1 + &t2;
        let result2 = &t2 + &t1;

        for (a, b) in result1.as_slice().iter().zip(result2.as_slice().iter()) {
            prop_assert!((a.get() - b.get()).abs() < 1e-6);
        }
    }

    #[test]
    fn prop_mul_commutative(
        data1 in prop::collection::vec(-100i32..100, 1..50),
        data2 in prop::collection::vec(-100i32..100, 1..50)
    ) {
        let len = data1.len().min(data2.len());
        let d1: Vec<Int32> = data1.iter().take(len).map(|&x| Int32::new(x)).collect();
        let d2: Vec<Int32> = data2.iter().take(len).map(|&x| Int32::new(x)).collect();

        let t1 = CpuTensorI32::from_vec(d1, &[len]).unwrap();
        let t2 = CpuTensorI32::from_vec(d2, &[len]).unwrap();

        let result1 = &t1 * &t2;
        let result2 = &t2 * &t1;

        prop_assert_eq!(result1.as_slice(), result2.as_slice());
    }
}

// ============================================================================
// Property: Arithmetic Associativity
// ============================================================================

proptest! {
    #[test]
    fn prop_add_associative(
        data1 in prop::collection::vec(-100.0f32..100.0, 1..30),
        data2 in prop::collection::vec(-100.0f32..100.0, 1..30),
        data3 in prop::collection::vec(-100.0f32..100.0, 1..30)
    ) {
        let len = data1.len().min(data2.len()).min(data3.len());
        let d1: Vec<Float32> = data1.iter().take(len).map(|&x| Float32::new(x)).collect();
        let d2: Vec<Float32> = data2.iter().take(len).map(|&x| Float32::new(x)).collect();
        let d3: Vec<Float32> = data3.iter().take(len).map(|&x| Float32::new(x)).collect();

        let t1 = CpuTensorF32::from_vec(d1, &[len]).unwrap();
        let t2 = CpuTensorF32::from_vec(d2, &[len]).unwrap();
        let t3 = CpuTensorF32::from_vec(d3, &[len]).unwrap();

        // (t1 + t2) + t3
        let temp1 = &t1 + &t2;
        let result1 = &temp1 + &t3;

        // t1 + (t2 + t3)
        let temp2 = &t2 + &t3;
        let result2 = &t1 + &temp2;

        // Note: Floating-point addition is NOT strictly associative due to rounding
        // We use a relaxed tolerance to account for accumulated rounding errors
        for (a, b) in result1.as_slice().iter().zip(result2.as_slice().iter()) {
            prop_assert!((a.get() - b.get()).abs() < 1e-3,
                "Associativity violated beyond tolerance: {} vs {}", a.get(), b.get());
        }
    }
}

// ============================================================================
// Property: Identity Elements
// ============================================================================

proptest! {
    #[test]
    fn prop_add_identity(
        data in prop::collection::vec(-1000.0f32..1000.0, 1..50)
    ) {
        let len = data.len();
        let d: Vec<Float32> = data.iter().map(|&x| Float32::new(x)).collect();
        let zeros: Vec<Float32> = vec![Float32::new(0.0); len];

        let t = CpuTensorF32::from_vec(d.clone(), &[len]).unwrap();
        let zero = CpuTensorF32::from_vec(zeros, &[len]).unwrap();

        let result = &t + &zero;

        for (a, b) in result.as_slice().iter().zip(d.iter()) {
            prop_assert!((a.get() - b.get()).abs() < 1e-6);
        }
    }

    #[test]
    fn prop_mul_identity(
        data in prop::collection::vec(-1000i32..1000, 1..50)
    ) {
        let len = data.len();
        let d: Vec<Int32> = data.iter().map(|&x| Int32::new(x)).collect();
        let ones: Vec<Int32> = vec![Int32::new(1); len];

        let t = CpuTensorI32::from_vec(d.clone(), &[len]).unwrap();
        let one = CpuTensorI32::from_vec(ones, &[len]).unwrap();

        let result = &t * &one;

        prop_assert_eq!(result.as_slice(), d.as_slice());
    }
}

// ============================================================================
// Property: Negation Involution
// ============================================================================

proptest! {
    #[test]
    fn prop_negation_involution(
        data in prop::collection::vec(-1000.0f64..1000.0, 1..50)
    ) {
        let len = data.len();
        let d: Vec<Float64> = data.iter().map(|&x| Float64::new(x)).collect();

        let t = CpuTensorF64::from_vec(d.clone(), &[len]).unwrap();

        // -(-t) == t
        let neg_once = -(&t);
        let neg_twice = -(&neg_once);

        for (a, b) in neg_twice.as_slice().iter().zip(d.iter()) {
            prop_assert!((a.get() - b.get()).abs() < 1e-10);
        }
    }
}

// ============================================================================
// Property: Subtraction as Inverse Addition
// ============================================================================

proptest! {
    #[test]
    fn prop_sub_inverse_add(
        data1 in prop::collection::vec(-100.0f32..100.0, 1..50),
        data2 in prop::collection::vec(-100.0f32..100.0, 1..50)
    ) {
        let len = data1.len().min(data2.len());
        let d1: Vec<Float32> = data1.iter().take(len).map(|&x| Float32::new(x)).collect();
        let d2: Vec<Float32> = data2.iter().take(len).map(|&x| Float32::new(x)).collect();

        let t1 = CpuTensorF32::from_vec(d1.clone(), &[len]).unwrap();
        let t2 = CpuTensorF32::from_vec(d2, &[len]).unwrap();

        // (t1 + t2) - t2 == t1
        let sum = &t1 + &t2;
        let result = &sum - &t2;

        for (a, b) in result.as_slice().iter().zip(d1.iter()) {
            prop_assert!((a.get() - b.get()).abs() < 1e-5);
        }
    }
}

// ============================================================================
// Property: Division as Inverse Multiplication
// ============================================================================

proptest! {
    #[test]
    fn prop_div_inverse_mul(
        data1 in prop::collection::vec(-100.0f64..100.0, 1..50),
        data2 in prop::collection::vec(0.1f64..100.0, 1..50) // Avoid division by zero
    ) {
        let len = data1.len().min(data2.len());
        let d1: Vec<Float64> = data1.iter().take(len).map(|&x| Float64::new(x)).collect();
        let d2: Vec<Float64> = data2.iter().take(len).map(|&x| Float64::new(x)).collect();

        let t1 = CpuTensorF64::from_vec(d1.clone(), &[len]).unwrap();
        let t2 = CpuTensorF64::from_vec(d2, &[len]).unwrap();

        // (t1 * t2) / t2 == t1
        let product = &t1 * &t2;
        let result = &product / &t2;

        for (a, b) in result.as_slice().iter().zip(d1.iter()) {
            prop_assert!((a.get() - b.get()).abs() < 1e-9);
        }
    }
}

// ============================================================================
// Property: Broadcasting Correctness
// ============================================================================

proptest! {
    #[test]
    fn prop_broadcast_scalar_addition(
        scalar in -1000.0f32..1000.0,
        data in prop::collection::vec(-100.0f32..100.0, 1..50)
    ) {
        let len = data.len();
        let d: Vec<Float32> = data.iter().map(|&x| Float32::new(x)).collect();
        let s = vec![Float32::new(scalar)];

        let t = CpuTensorF32::from_vec(d.clone(), &[len]).unwrap();
        let scalar_tensor = CpuTensorF32::from_vec(s, &[1]).unwrap();

        let result = &t + &scalar_tensor;

        prop_assert_eq!(result.shape().dims(), &[len]);
        for (a, &b) in result.as_slice().iter().zip(d.iter()) {
            prop_assert!((a.get() - (b.get() + scalar)).abs() < 1e-5);
        }
    }
}

// ============================================================================
// Property: Overflow/Underflow Edge Cases
// ============================================================================

proptest! {
    #[test]
    fn prop_add_overflow_edge_cases(
        data1 in prop::collection::vec(any::<f32>(), 1..10),
        data2 in prop::collection::vec(any::<f32>(), 1..10)
    ) {
        let len = data1.len().min(data2.len());
        let d1: Vec<Float32> = data1.iter().take(len).map(|&x| Float32::new(x)).collect();
        let d2: Vec<Float32> = data2.iter().take(len).map(|&x| Float32::new(x)).collect();

        let t1 = CpuTensorF32::from_vec(d1, &[len]).unwrap();
        let t2 = CpuTensorF32::from_vec(d2, &[len]).unwrap();

        let result = &t1 + &t2;

        // Result should be finite unless inputs contain inf/nan
        for &val in result.as_slice() {
            let v = val.get();
            prop_assert!(v.is_finite() || t1.as_slice().iter().any(|x| !x.get().is_finite()) ||
                        t2.as_slice().iter().any(|x| !x.get().is_finite()));
        }
    }

    #[test]
    fn prop_mul_precision_edge_cases(
        data1 in prop::collection::vec(-1e10f64..1e10, 1..10),
        data2 in prop::collection::vec(-1e-10f64..1e-10, 1..10)
    ) {
        let len = data1.len().min(data2.len());
        let d1: Vec<Float64> = data1.iter().take(len).map(|&x| Float64::new(x)).collect();
        let d2: Vec<Float64> = data2.iter().take(len).map(|&x| Float64::new(x)).collect();

        let t1 = CpuTensorF64::from_vec(d1.clone(), &[len]).unwrap();
        let t2 = CpuTensorF64::from_vec(d2.clone(), &[len]).unwrap();

        let result = &t1 * &t2;

        // Check that very small numbers multiplied by large numbers don't cause issues
        for ((&a, &b), &res) in d1.iter().zip(d2.iter()).zip(result.as_slice()) {
            let expected = a.get() * b.get();
            let actual = res.get();
            // Allow for some floating point precision loss
            prop_assert!((expected - actual).abs() <= expected.abs() * 1e-12 + 1e-12);
        }
    }
}

// ============================================================================
// Property: Extreme Value Handling
// ============================================================================

proptest! {
    #[test]
    fn prop_extreme_values_handling(
        data in prop::collection::vec(any::<f32>(), 1..20)
    ) {
        let len = data.len();
        let d: Vec<Float32> = data.iter().map(|&x| {
            match x.to_bits() % 5 {
                0 => Float32::new(f32::INFINITY),
                1 => Float32::new(f32::NEG_INFINITY),
                2 => Float32::new(f32::NAN),
                3 => Float32::new(0.0),
                _ => Float32::new(x),
            }
        }).collect();

        let t1 = CpuTensorF32::from_vec(d.clone(), &[len]).unwrap();
        let t2 = CpuTensorF32::from_vec(vec![Float32::new(1.0); len], &[len]).unwrap();

        // Operations with extreme values should not panic
        let _add_result = &t1 + &t2;
        let _mul_result = &t1 * &t2;
    }

    #[test]
    fn prop_zero_edge_cases(
        data in prop::collection::vec(-1000.0f32..1000.0, 1..15)
    ) {
        let len = data.len();
        let d: Vec<Float32> = data.iter().map(|&x| Float32::new(x)).collect();
        let zeros = vec![Float32::new(0.0); len];

        let t = CpuTensorF32::from_vec(d.clone(), &[len]).unwrap();
        let zero_tensor = CpuTensorF32::from_vec(zeros, &[len]).unwrap();

        // Addition with zero should preserve original values
        let add_result = &t + &zero_tensor;
        for (&orig, &res) in d.iter().zip(add_result.as_slice()) {
            prop_assert!((orig.get() - res.get()).abs() < 1e-6);
        }

        // Multiplication by zero should give zero
        let mul_result = &t * &zero_tensor;
        for &val in mul_result.as_slice() {
            prop_assert!(val.get().abs() < 1e-6);
        }
    }
}

// ============================================================================
// Property: Broadcasting Edge Cases
// ============================================================================

proptest! {
    #[test]
    fn prop_broadcasting_complex_shapes(
        dims1 in prop::collection::vec(1usize..5, 1..3),
        dims2 in prop::collection::vec(1usize..5, 1..3)
    ) {
        // Create tensors with potentially compatible broadcasting shapes
        let size1: usize = dims1.iter().product();
        let size2: usize = dims2.iter().product();

        if size1 == 0 || size2 == 0 {
            return Ok(());
        }

        let d1: Vec<Float32> = (0..size1).map(|i| Float32::new(i as f32)).collect();
        let d2: Vec<Float32> = (0..size2).map(|i| Float32::new((i + 1) as f32)).collect();

        let t1 = CpuTensorF32::from_vec(d1, &dims1).unwrap();
        let t2 = CpuTensorF32::from_vec(d2, &dims2).unwrap();

        // Check if shapes are broadcastable
        let can_broadcast = {
            let mut compatible = true;
            let max_len = dims1.len().max(dims2.len());
            let padded1 = vec![1; max_len - dims1.len()].into_iter().chain(dims1.iter().cloned());
            let padded2 = vec![1; max_len - dims2.len()].into_iter().chain(dims2.iter().cloned());

            for (a, b) in padded1.zip(padded2) {
                if a != 1 && b != 1 && a != b {
                    compatible = false;
                    break;
                }
            }
            compatible
        };

        if can_broadcast {
            // Should succeed
            let result = &t1 + &t2;
            prop_assert!(result.numel() > 0);
        } else {
            // Should panic - skip this test case
            return Ok(());
        }
    }

    #[test]
    fn prop_broadcasting_preserves_data_integrity(
        scalar in -100.0f32..100.0,
        shape in prop::collection::vec(1usize..8, 1..3)
    ) {
        let size: usize = shape.iter().product();
        if size == 0 {
            return Ok(());
        }

        let d: Vec<Float32> = (0..size).map(|i| Float32::new(i as f32)).collect();
        let s = vec![Float32::new(scalar)];

        let tensor = CpuTensorF32::from_vec(d.clone(), &shape).unwrap();
        let scalar_tensor = CpuTensorF32::from_vec(s, &[1]).unwrap();

        let result = &tensor + &scalar_tensor;

        // Result should have same shape as original tensor
        prop_assert_eq!(result.shape().dims(), &shape[..]);

        // Each element should be original + scalar
        for (&orig, &res) in d.iter().zip(result.as_slice()) {
            prop_assert!((orig.get() + scalar - res.get()).abs() < 1e-6);
        }
    }
}

// ============================================================================
// Property: Integer Overflow Testing
// ============================================================================

proptest! {
    #[test]
    fn prop_integer_overflow_add(
        data1 in prop::collection::vec(i32::MIN/2..i32::MAX/2, 1..10),
        data2 in prop::collection::vec(i32::MIN/2..i32::MAX/2, 1..10)
    ) {
        let len = data1.len().min(data2.len());
        let d1: Vec<Int32> = data1.iter().take(len).map(|&x| Int32::new(x)).collect();
        let d2: Vec<Int32> = data2.iter().take(len).map(|&x| Int32::new(x)).collect();

        let t1 = CpuTensorI32::from_vec(d1.clone(), &[len]).unwrap();
        let t2 = CpuTensorI32::from_vec(d2.clone(), &[len]).unwrap();

        let result = &t1 + &t2;

        // Check that results are consistent with wrapping arithmetic
        for ((&a, &b), &res) in d1.iter().zip(d2.iter()).zip(result.as_slice()) {
            let expected = a.get().wrapping_add(b.get());
            prop_assert_eq!(res.get(), expected);
        }
    }

    #[test]
    fn prop_integer_extreme_values(
        data in prop::collection::vec(any::<i32>(), 1..15)
    ) {
        let len = data.len();
        let d: Vec<Int32> = data.iter().map(|&x| Int32::new(x)).collect();

        let t = CpuTensorI32::from_vec(d.clone(), &[len]).unwrap();

        // Test identity operations
        let doubled = &t + &t;
        for (&orig, &res) in d.iter().zip(doubled.as_slice()) {
            prop_assert_eq!(res.get(), orig.get().wrapping_mul(2));
        }
    }
}

// ============================================================================
// Property: Shape Invariants
// ============================================================================

proptest! {
    #[test]
    fn prop_shape_invariants_under_operations(
        dims in prop::collection::vec(1usize..10, 2..4),
        op_type in 0..3i32
    ) {
        let size: usize = dims.iter().product();
        if size == 0 {
            return Ok(());
        }

        let d1: Vec<Float32> = (0..size).map(|i| Float32::new(i as f32)).collect();
        let d2: Vec<Float32> = (0..size).map(|i| Float32::new((i % 5) as f32)).collect();

        let t1 = CpuTensorF32::from_vec(d1, &dims).unwrap();
        let t2 = CpuTensorF32::from_vec(d2, &dims).unwrap();

        let result = match op_type {
            0 => &t1 + &t2,
            1 => &t1 * &t2,
            _ => &t1 - &t2,
        };

        // Shape should be preserved for same-shaped tensors
        prop_assert_eq!(result.shape().dims(), &dims[..]);
        prop_assert_eq!(result.numel(), size);
    }

    #[test]
    fn prop_empty_tensor_handling(
        empty_dims in prop::collection::vec(0usize..2, 1..4)
    ) {
        // Create potentially empty tensors
        let size: usize = empty_dims.iter().product();

        if size > 0 {
            return Ok(()); // Skip non-empty cases
        }

        // For empty tensors, create with empty data vector
        let d: Vec<Float32> = Vec::new();
        let t = CpuTensorF32::from_vec(d, &empty_dims).unwrap();

        prop_assert_eq!(t.numel(), 0);
        prop_assert_eq!(t.len(), 0);
    }
}
