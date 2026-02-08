//! Property-based tests for tensor operations using proptest.
//!
//! These tests validate mathematical invariants, edge cases, and correctness
//! properties that are difficult to test with traditional unit tests.

use proptest::prelude::*;

use approx::assert_relative_eq;
use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::ops::{arithmetic, reduction, linalg};
use tensor::{Tensor, Backend};

/// Type alias for our test tensor type
type TestTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

fn checked_numel(dims: &[usize]) -> Option<usize> {
    dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
}

/// Generate random tensor shapes (1D to 4D)
fn arb_tensor_shape() -> impl Strategy<Value = Vec<usize>> {
    const MAX_DIM: usize = 64;
    const MAX_ELEMENTS: usize = 4_096;
    prop::collection::vec(1..=MAX_DIM, 1..=4).prop_filter("shape within element budget", |s| {
        checked_numel(s).is_some_and(|n| n > 0 && n <= MAX_ELEMENTS)
    })
}

/// Generate a random tensor with given shape
fn arb_tensor() -> impl Strategy<Value = TestTensor> {
    arb_tensor_shape().prop_flat_map(|shape| {
        let len = checked_numel(&shape).unwrap();
        let data = prop::collection::vec((-1.0e3f32..1.0e3).prop_map(Float32::new), len);
        data.prop_map(move |data| Tensor::from_vec(data, &shape).unwrap())
    })
}

/// Generate two tensors with compatible shapes for element-wise operations
fn arb_tensor_pair_same_shape() -> impl Strategy<Value = (TestTensor, TestTensor)> {
    arb_tensor_shape().prop_flat_map(|shape| {
        let size = checked_numel(&shape).unwrap();
        let data1 = prop::collection::vec((-1.0e3f32..1.0e3).prop_map(Float32::new), size);
        let data2 = prop::collection::vec((-1.0e3f32..1.0e3).prop_map(Float32::new), size);

        (data1, data2).prop_map(move |(d1, d2)| {
            (
                Tensor::from_vec(d1, &shape).unwrap(),
                Tensor::from_vec(d2, &shape).unwrap(),
            )
        })
    })
}

/// Generate tensors for matrix multiplication (compatible shapes)
fn arb_matrix_mult_pair() -> impl Strategy<Value = (TestTensor, TestTensor)> {
    (1..=50usize, 1..=50usize, 1..=50usize).prop_flat_map(|(m, k, n)| {
        let a_size = m * k;
        let b_size = k * n;
        let a_data = prop::collection::vec((-1.0e3f32..1.0e3).prop_map(Float32::new), a_size);
        let b_data = prop::collection::vec((-1.0e3f32..1.0e3).prop_map(Float32::new), b_size);

        (a_data, b_data).prop_map(move |(a_d, b_d)| {
            (
                Tensor::from_vec(a_d, &[m, k]).unwrap(),
                Tensor::from_vec(b_d, &[k, n]).unwrap(),
            )
        })
    })
}

proptest! {
    /// Test that tensor addition is commutative: a + b = b + a
    #[test]
    fn test_addition_commutative((ref a, ref b) in arb_tensor_pair_same_shape()) {
        let sum1 = arithmetic::add(a, b).unwrap();
        let sum2 = arithmetic::add(b, a).unwrap();

        prop_assert_eq!(sum1.shape().dims(), sum2.shape().dims());
        for i in 0..sum1.len() {
            assert_relative_eq!(
                sum1.as_slice()[i].get(),
                sum2.as_slice()[i].get(),
                epsilon = 1e-6,
                max_relative = 1e-6
            );
        }
    }

    /// Test that tensor addition is associative: (a + b) + c = a + (b + c)
    #[test]
    fn test_addition_associative((ref a, ref b) in arb_tensor_pair_same_shape()) {
        let c_data: Vec<Float32> = (0..a.len()).map(|_| Float32::new(1.5)).collect();
        let c = Tensor::<_, DenseStorage<_>, _>::from_vec(c_data, a.shape().dims()).unwrap();

        let ab = arithmetic::add(a, b).unwrap();
        let sum1 = arithmetic::add(&ab, &c).unwrap();
        let bc = arithmetic::add(b, &c).unwrap();
        let sum2 = arithmetic::add(a, &bc).unwrap();

        prop_assert_eq!(sum1.shape().dims(), sum2.shape().dims());
        for i in 0..sum1.len() {
            assert_relative_eq!(
                sum1.as_slice()[i].get(),
                sum2.as_slice()[i].get(),
                epsilon = 1e-4,
                max_relative = 1e-6
            );
        }
    }

    /// Test that tensor multiplication is commutative: a * b = b * a
    #[test]
    fn test_multiplication_commutative((ref a, ref b) in arb_tensor_pair_same_shape()) {
        let prod1 = arithmetic::mul(a, b).unwrap();
        let prod2 = arithmetic::mul(b, a).unwrap();

        prop_assert_eq!(prod1.shape().dims(), prod2.shape().dims());
        for i in 0..prod1.len() {
            assert_relative_eq!(
                prod1.as_slice()[i].get(),
                prod2.as_slice()[i].get(),
                epsilon = 1e-6
            );
        }
    }

    /// Test that addition with zero is identity: a + 0 = a
    #[test]
    fn test_addition_identity(ref a in arb_tensor()) {
        let zero_data: Vec<Float32> = (0..a.len()).map(|_| Float32::new(0.0)).collect();
        let zero = Tensor::<_, DenseStorage<_>, _>::from_vec(zero_data, a.shape().dims()).unwrap();

        let result = arithmetic::add(a, &zero).unwrap();

        prop_assert_eq!(result.shape().dims(), a.shape().dims());
        for i in 0..result.len() {
            assert_relative_eq!(
                result.as_slice()[i].get(),
                a.as_slice()[i].get(),
                epsilon = 1e-6
            );
        }
    }

    /// Test that multiplication by one is identity: a * 1 = a
    #[test]
    fn test_multiplication_identity(ref a in arb_tensor()) {
        let one_data: Vec<Float32> = (0..a.len()).map(|_| Float32::new(1.0)).collect();
        let one = Tensor::<_, DenseStorage<_>, _>::from_vec(one_data, a.shape().dims()).unwrap();

        let result = arithmetic::mul(a, &one).unwrap();

        prop_assert_eq!(result.shape().dims(), a.shape().dims());
        for i in 0..result.len() {
            assert_relative_eq!(
                result.as_slice()[i].get(),
                a.as_slice()[i].get(),
                epsilon = 1e-6
            );
        }
    }

    /// Test that multiplication by zero gives zero: a * 0 = 0
    #[test]
    fn test_multiplication_zero(ref a in arb_tensor()) {
        let zero_data: Vec<Float32> = (0..a.len()).map(|_| Float32::new(0.0)).collect();
        let zero = Tensor::<_, DenseStorage<_>, _>::from_vec(zero_data, a.shape().dims()).unwrap();

        let result = arithmetic::mul(a, &zero).unwrap();

        for i in 0..result.len() {
            assert_relative_eq!(
                result.as_slice()[i].get(),
                0.0,
                epsilon = 1e-6
            );
        }
    }

    /// Test matrix multiplication shape compatibility
    #[test]
    fn test_matrix_mult_shape((ref a, ref b) in arb_matrix_mult_pair()) {
        let result = linalg::matmul(a, b).unwrap();
        let expected_shape = &[a.shape().dims()[0], b.shape().dims()[1]];
        prop_assert_eq!(result.shape().dims(), expected_shape);
    }

    /// Test tensor transpose shape
    #[test]
    fn test_transpose_shape(ref a in arb_tensor()) {
        let original_shape = a.shape().dims();

        if original_shape.len() == 2 {
            let transposed = tensor::ops::transpose(a, 0, 1).unwrap();
            let expected_shape = &[original_shape[1], original_shape[0]];
            prop_assert_eq!(transposed.shape().dims(), expected_shape);
        }
    }

    /// Test that transpose is involution: (A^T)^T = A
    #[test]
    fn test_transpose_involution(ref a in arb_tensor()) {
        if a.shape().dims().len() == 2 {
            let transposed = tensor::ops::transpose(a, 0, 1).unwrap();
            let double_transposed = tensor::ops::transpose(&transposed, 0, 1).unwrap();

            prop_assert_eq!(double_transposed.shape().dims(), a.shape().dims());
            for i in 0..a.len() {
                assert_relative_eq!(
                    double_transposed.as_slice()[i].get(),
                    a.as_slice()[i].get(),
                    epsilon = 1e-6
                );
            }
        }
    }

    /// Test element-wise power operation
    #[test]
    fn test_power_operation(ref a in arb_tensor()) {
        let power = a.powf(Float32::new(2.0));

        prop_assert_eq!(power.shape().dims(), a.shape().dims());
        for i in 0..power.len() {
            let original = a.as_slice()[i].get();
            let expected = original * original;
            assert_relative_eq!(
                power.as_slice()[i].get(),
                expected,
                epsilon = 1e-6
            );
        }
    }

    /// Test element-wise square root (only for non-negative values)
    #[test]
    fn test_sqrt_operation(ref a in arb_tensor()) {
        let clamped = a.clamp(Float32::new(-1.0e10), Float32::new(1.0e10)).unwrap();
        let squared = clamped.powf(Float32::new(2.0));
        let sqrt_result = squared.sqrt();

        prop_assert_eq!(sqrt_result.shape().dims(), a.shape().dims());
        for i in 0..sqrt_result.len() {
            let original = clamped.as_slice()[i].get();
            assert_relative_eq!(
                sqrt_result.as_slice()[i].get(),
                original.abs(),
                epsilon = 1e-5
            );
        }
    }

    /// Test sum reduction
    #[test]
    fn test_sum_reduction(ref a in arb_tensor()) {
        let sum_result = reduction::sum(a, None, false).unwrap();

        // Sum should be scalar (empty shape [])
        prop_assert_eq!(sum_result.shape().dims(), &[]);

        // Manual sum verification
        let manual_sum: f32 = a.as_slice().iter().map(|x| x.get()).sum();
        assert_relative_eq!(
            sum_result.as_slice()[0].get(),
            manual_sum,
            epsilon = 1e-4
        );
    }

    /// Test mean reduction
    #[test]
    fn test_mean_reduction(ref a in arb_tensor()) {
        let mean_result = reduction::mean(a, None, false).unwrap();

        // Mean should be scalar (empty shape [])
        prop_assert_eq!(mean_result.shape().dims(), &[]);

        // Manual mean verification
        let manual_sum: f32 = a.as_slice().iter().map(|x| x.get()).sum();
        let manual_mean = manual_sum / a.len() as f32;
        assert_relative_eq!(
            mean_result.as_slice()[0].get(),
            manual_mean,
            epsilon = 1e-4
        );
    }

    /// Test broadcasting with scalars
    #[test]
    fn test_scalar_broadcasting(ref a in arb_tensor()) {
        let scalar = Float32::new(3.5);
        let scalar_t = Tensor::<_, DenseStorage<_>, _>::from_vec(vec![scalar], &[1]).unwrap();
        let result = arithmetic::add(a, &scalar_t).unwrap();

        prop_assert_eq!(result.shape().dims(), a.shape().dims());
        for i in 0..result.len() {
            let expected = a.as_slice()[i].get() + scalar.get();
            assert_relative_eq!(
                result.as_slice()[i].get(),
                expected,
                epsilon = 1e-6
            );
        }
    }

    /// Test that cloning preserves data
    #[test]
    fn test_clone_preserves_data(ref a in arb_tensor()) {
        let cloned = a.clone();

        prop_assert_eq!(cloned.shape().dims(), a.shape().dims());
        prop_assert_eq!(cloned.len(), a.len());

        for i in 0..a.len() {
            assert_relative_eq!(
                cloned.as_slice()[i].get(),
                a.as_slice()[i].get(),
                epsilon = 1e-10
            );
        }
    }

    /// Test tensor reshape preserves total elements
    #[test]
    fn test_reshape_preserves_elements(ref a in arb_tensor()) {
        if a.len() >= 4 && a.len() % 2 == 0 {
            let new_shape_usize = [2usize, a.len() / 2];
            let new_shape_isize = [
                2isize,
                isize::try_from(new_shape_usize[1]).unwrap_or(isize::MAX),
            ];
            let reshaped = a.reshape(&new_shape_isize).unwrap();

            prop_assert_eq!(reshaped.len(), a.len());
            prop_assert_eq!(reshaped.shape().dims(), &new_shape_usize[..]);

            // Data should be preserved in row-major order
            for i in 0..a.len() {
                assert_relative_eq!(
                    reshaped.as_slice()[i].get(),
                    a.as_slice()[i].get(),
                    epsilon = 1e-10
                );
            }
        }
    }

    /// Test that negation works: -(-a) = a
    #[test]
    fn test_negation_involution(ref a in arb_tensor()) {
        let neg = -a;
        let double_neg = -&neg;

        prop_assert_eq!(double_neg.shape().dims(), a.shape().dims());
        for i in 0..a.len() {
            assert_relative_eq!(
                double_neg.as_slice()[i].get(),
                a.as_slice()[i].get(),
                epsilon = 1e-6
            );
        }
    }

    /// Test reciprocal: a * (1/a) = 1 (for non-zero elements)
    #[test]
    fn test_reciprocal(ref a in arb_tensor()) {
        let one = Float32::new(1.0);
        let recip_data: Vec<Float32> = a.as_slice().iter().copied().map(|x| one / x).collect();
        let recip =
            Tensor::<_, DenseStorage<_>, _>::from_vec_with_backend(recip_data, a.shape().dims(), a.backend().clone())
                .unwrap();
        let product = arithmetic::mul(a, &recip).unwrap();

        prop_assert_eq!(product.shape().dims(), a.shape().dims());
        for i in 0..product.len() {
            let original = a.as_slice()[i].get();
            if original.abs() > 1e-6 {  // Avoid division by very small numbers
                assert_relative_eq!(
                    product.as_slice()[i].get(),
                    1.0,
                    epsilon = 1e-4
                );
            }
        }
    }

    /// Test exp and log are inverses: log(exp(a)) = a
    #[test]
    fn test_exp_log_inverse(ref a in arb_tensor()) {
        // Keep values in reasonable range for exp
        let clamped_a = a.clamp(Float32::new(-10.0), Float32::new(10.0)).unwrap();

        let exp_result = clamped_a.exp();
        let log_result = exp_result.log();

        prop_assert_eq!(log_result.shape().dims(), a.shape().dims());
        for i in 0..log_result.len() {
            assert_relative_eq!(
                log_result.as_slice()[i].get(),
                clamped_a.as_slice()[i].get(),
                epsilon = 1e-3
            );
        }
    }
}
