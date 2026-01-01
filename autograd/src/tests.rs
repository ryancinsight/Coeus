//! Tests for the autograd system

use crate::ops::{add, backward_with_grad, matmul, mul, nll_loss};
use backend::CpuBackend;
use dtype::float::Float32;
use storage::{CsrStorage, DenseStorage};
use tensor::Tensor;

type TestTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;
type SparseTestTensor = Tensor<CpuBackend<Float32>, CsrStorage<Float32>, Float32>;

// Property-based testing imports
#[cfg(test)]
use approx::assert_relative_eq;
#[cfg(test)]
use num_traits::Float as NumFloat;
#[cfg(test)]
use proptest::prelude::*;

#[test]
fn test_basic_addition_grad() {
    // Create tensors with gradient tracking
    let mut x = TestTensor::from_vec(vec![Float32::new(2.0)], &[1]).unwrap();
    x = x.requires_grad_(true);

    let mut y = TestTensor::from_vec(vec![Float32::new(3.0)], &[1]).unwrap();
    y = y.requires_grad_(true);

    // Perform addition
    let z = add::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>(&x, &y).unwrap();

    // Check forward pass
    assert_relative_eq!(z.as_slice()[0].get(), 5.0, epsilon = 1e-6);
    println!("z.requires_grad(): {}", z.requires_grad());

    // Backward pass with gradient
    let grad_output = TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
    println!("Calling backward_with_grad");
    backward_with_grad(&z, grad_output).unwrap();
    println!("backward_with_grad completed");

    // Check gradients
    let x_grad = x.grad().unwrap();
    let y_grad = y.grad().unwrap();

    assert_relative_eq!(x_grad.as_slice()[0].get(), 1.0, epsilon = 1e-6);
    assert_relative_eq!(y_grad.as_slice()[0].get(), 1.0, epsilon = 1e-6);
}

#[test]
fn test_tensor_gradient_methods() {
    let mut x = TestTensor::from_vec(vec![Float32::new(2.0), Float32::new(-1.0)], &[2]).unwrap();
    x = x.requires_grad_(true);

    // Test zero_grad
    x.zero_grad().unwrap();

    // Test is_nan (should be false for normal values)
    assert!(!x.is_nan());

    // Test is_inf (should be false for finite values)
    assert!(!x.is_inf());

    // Test clamp
    let clamped = x.clamp(Float32::new(-0.5), Float32::new(1.5)).unwrap();
    assert_relative_eq!(clamped.as_slice()[0].get(), 1.5, epsilon = 1e-6); // 2.0 clamped to 1.5
    assert_relative_eq!(clamped.as_slice()[1].get(), -0.5, epsilon = 1e-6); // -1.0 clamped to -0.5
}

#[test]
fn test_nan_inf_detection() {
    // Create tensor with NaN
    let nan_val = Float32::new(f32::NAN);
    let inf_val = Float32::new(f32::INFINITY);

    let x = TestTensor::from_vec(vec![nan_val, inf_val, Float32::new(1.0)], &[3]).unwrap();

    assert!(x.is_nan()); // Contains NaN
    assert!(x.is_inf()); // Contains infinity
}

#[test]
fn test_gradient_accumulation() {
    let mut x = TestTensor::from_vec(vec![Float32::new(2.0)], &[1]).unwrap();
    x = x.requires_grad_(true);

    let mut y = TestTensor::from_vec(vec![Float32::new(3.0)], &[1]).unwrap();
    y = y.requires_grad_(true);

    // First computation: z1 = x + y
    let z1 = add(&x, &y).unwrap();
    let grad1 = TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
    backward_with_grad(&z1, grad1).unwrap();

    // Second computation: z2 = x + y (same operation)
    let z2 = add(&x, &y).unwrap();
    let grad2 = TestTensor::from_vec(vec![Float32::new(1.0)], &[1]).unwrap();
    backward_with_grad(&z2, grad2).unwrap();

    // Gradients should accumulate
    let x_grad = x.grad().unwrap();
    let y_grad = y.grad().unwrap();

    assert_relative_eq!(x_grad.as_slice()[0].get(), 2.0, epsilon = 1e-6); // 1.0 + 1.0
    assert_relative_eq!(y_grad.as_slice()[0].get(), 2.0, epsilon = 1e-6); // 1.0 + 1.0
}

#[test]
fn test_no_grad_tensors() {
    // Create tensors without gradient tracking
    let x = TestTensor::from_vec(vec![Float32::new(2.0)], &[1]).unwrap();
    let y = TestTensor::from_vec(vec![Float32::new(3.0)], &[1]).unwrap();

    // Perform addition
    let z = add::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>(&x, &y).unwrap();

    // Should not have gradients
    assert!(!x.requires_grad());
    assert!(!y.requires_grad());
    assert!(!z.requires_grad());
}

#[test]
fn test_matmul_backward() {
    // Test matrix multiplication backward pass
    let mut a = TestTensor::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[2, 2],
    )
    .unwrap();
    a = a.requires_grad_(true);

    let mut b = TestTensor::from_vec(
        vec![
            Float32::new(5.0),
            Float32::new(6.0),
            Float32::new(7.0),
            Float32::new(8.0),
        ],
        &[2, 2],
    )
    .unwrap();
    b = b.requires_grad_(true);

    let c = crate::ops::matmul(&a, &b).unwrap();
    let grad_output = TestTensor::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(0.0),
            Float32::new(0.0),
            Float32::new(1.0),
        ],
        &[2, 2],
    )
    .unwrap();

    crate::ops::backward_with_grad(&c, grad_output).unwrap();

    let a_grad = a.grad().unwrap();
    let b_grad = b.grad().unwrap();

    // For C = A @ B, ∂C/∂A = grad_output @ B^T
    // B^T = [[5, 7], [6, 8]]
    // grad_output @ B^T = [[1, 0], [0, 1]] @ [[5, 7], [6, 8]] = [[5, 7], [6, 8]]
    assert_relative_eq!(a_grad.as_slice()[0].get(), 5.0, epsilon = 1e-6); // ∂C/∂A[0,0]
    assert_relative_eq!(a_grad.as_slice()[1].get(), 7.0, epsilon = 1e-6); // ∂C/∂A[0,1]
    assert_relative_eq!(a_grad.as_slice()[2].get(), 6.0, epsilon = 1e-6); // ∂C/∂A[1,0]
    assert_relative_eq!(a_grad.as_slice()[3].get(), 8.0, epsilon = 1e-6); // ∂C/∂A[1,1]

    // For C = A @ B, ∂C/∂B = A^T @ grad_output
    // A^T = [[1, 3], [2, 4]]
    // A^T @ grad_output = [[1, 3], [2, 4]] @ [[1, 0], [0, 1]] = [[1, 3], [2, 4]]
    assert_relative_eq!(b_grad.as_slice()[0].get(), 1.0, epsilon = 1e-6); // ∂C/∂B[0,0]
    assert_relative_eq!(b_grad.as_slice()[1].get(), 3.0, epsilon = 1e-6); // ∂C/∂B[0,1]
    assert_relative_eq!(b_grad.as_slice()[2].get(), 2.0, epsilon = 1e-6); // ∂C/∂B[1,0]
    assert_relative_eq!(b_grad.as_slice()[3].get(), 4.0, epsilon = 1e-6); // ∂C/∂B[1,1]
}

#[test]
fn test_exp_backward() {
    // Test exponential backward pass: d/dx exp(x) = exp(x)
    let mut x = TestTensor::from_vec(vec![Float32::new(0.0), Float32::new(1.0)], &[2]).unwrap();
    x = x.requires_grad_(true);

    let y = crate::ops::exp(&x).unwrap();
    let grad_output =
        TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();

    crate::ops::backward_with_grad(&y, grad_output).unwrap();

    let x_grad = x.grad().unwrap();

    // d/dx exp(x) = exp(x), so gradient = grad_output * exp(x)
    // exp(0) = 1, exp(1) ≈ 2.718
    // gradient[0] = 1.0 * 1.0 = 1.0
    // gradient[1] = 2.0 * exp(1) ≈ 2.0 * 2.718 ≈ 5.436
    assert!((x_grad.as_slice()[0].get() - 1.0).abs() < 1e-6);
    assert!((x_grad.as_slice()[1].get() - 2.0 * std::f32::consts::E).abs() < 1e-5);
}

#[test]
fn test_log_backward() {
    // Test logarithm backward pass: d/dx log(x) = 1/x
    let mut x = TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();
    x = x.requires_grad_(true);

    let y = crate::ops::log(&x).unwrap();
    let grad_output =
        TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();

    crate::ops::backward_with_grad(&y, grad_output).unwrap();

    let x_grad = x.grad().unwrap();

    // d/dx log(x) = 1/x, so gradient = grad_output * (1/x)
    // gradient[0] = 1.0 * (1/1) = 1.0
    // gradient[1] = 2.0 * (1/2) = 1.0
    assert!((x_grad.as_slice()[0].get() - 1.0).abs() < 1e-6);
    assert!((x_grad.as_slice()[1].get() - 1.0).abs() < 1e-6);
}

#[test]
fn test_sin_backward() {
    // Test sine backward pass: d/dx sin(x) = cos(x)
    let mut x = TestTensor::from_vec(
        vec![Float32::new(0.0), Float32::new(std::f32::consts::PI / 2.0)],
        &[2],
    )
    .unwrap();
    x = x.requires_grad_(true);

    let y = crate::ops::sin(&x).unwrap();
    let grad_output =
        TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();

    crate::ops::backward_with_grad(&y, grad_output).unwrap();

    let x_grad = x.grad().unwrap();

    // d/dx sin(x) = cos(x), so gradient = grad_output * cos(x)
    // cos(0) = 1, cos(π/2) = 0
    // gradient[0] = 1.0 * 1.0 = 1.0
    // gradient[1] = 2.0 * 0.0 = 0.0
    assert!((x_grad.as_slice()[0].get() - 1.0).abs() < 1e-6);
    assert!((x_grad.as_slice()[1].get() - 0.0).abs() < 1e-6);
}

#[test]
fn test_cos_backward() {
    // Test cosine backward pass: d/dx cos(x) = -sin(x)
    let mut x = TestTensor::from_vec(
        vec![Float32::new(0.0), Float32::new(std::f32::consts::PI / 2.0)],
        &[2],
    )
    .unwrap();
    x = x.requires_grad_(true);

    let y = crate::ops::cos(&x).unwrap();
    let grad_output =
        TestTensor::from_vec(vec![Float32::new(1.0), Float32::new(2.0)], &[2]).unwrap();

    crate::ops::backward_with_grad(&y, grad_output).unwrap();

    let x_grad = x.grad().unwrap();

    // d/dx cos(x) = -sin(x), so gradient = grad_output * (-sin(x))
    // -sin(0) = 0, -sin(π/2) = -1
    // gradient[0] = 1.0 * 0.0 = 0.0
    // gradient[1] = 2.0 * (-1.0) = -2.0
    assert!((x_grad.as_slice()[0].get() - 0.0).abs() < 1e-6);
    assert!((x_grad.as_slice()[1].get() + 2.0).abs() < 1e-6);
}

#[test]
fn test_nll_loss_input_validation() {
    // Test that NLL loss properly validates inputs and rejects invalid indices

    // Create log-probabilities: 3 classes, 2 batches
    // Manual softmax-like: class 0 has high prob, others low
    let log_probs = TestTensor::from_vec(
        vec![
            Float32::new(-0.1), // batch 0, class 0: high prob (log)
            Float32::new(-2.3), // batch 0, class 1: low prob
            Float32::new(-2.3), // batch 0, class 2: low prob
            Float32::new(-2.3), // batch 1, class 0: low prob
            Float32::new(-0.1), // batch 1, class 1: high prob (log)
            Float32::new(-2.3), // batch 1, class 2: low prob
        ],
        &[2, 3],
    )
    .unwrap();

    // Valid targets: batch 0 -> class 0, batch 1 -> class 1
    let targets = TestTensor::from_vec(
        vec![
            Float32::new(0.0), // batch 0 picks class 0 (valid integer)
            Float32::new(1.0), // batch 1 picks class 1 (valid integer)
        ],
        &[2],
    )
    .unwrap();

    // Should succeed with valid inputs
    let loss_val = nll_loss(&log_probs, &targets).unwrap();
    assert!(loss_val.as_slice()[0].get() > 0.0); // NLL loss should be positive

    // Test error cases

    // Case 1: Non-integer target
    let bad_targets = TestTensor::from_vec(
        vec![
            Float32::new(0.5), // batch 0: non-integer (should be 0, 1, or 2)
            Float32::new(1.0),
        ],
        &[2],
    )
    .unwrap();

    let bad_loss = nll_loss(&log_probs, &bad_targets);
    assert!(bad_loss.is_err());
    assert!(bad_loss.unwrap_err().to_string().contains("not an integer"));

    // Case 2: Target out of range (negative)
    let bad_targets2 = TestTensor::from_vec(
        vec![
            Float32::new(-1.0), // batch 0: negative index
            Float32::new(1.0),
        ],
        &[2],
    )
    .unwrap();

    let bad_loss2 = nll_loss(&log_probs, &bad_targets2);
    assert!(bad_loss2.is_err());
    assert!(bad_loss2.unwrap_err().to_string().contains("out of range"));

    // Case 3: Target out of range (too high)
    let bad_targets3 = TestTensor::from_vec(
        vec![
            Float32::new(3.0), // batch 0: class 3 doesn't exist (only 0,1,2)
            Float32::new(1.0),
        ],
        &[2],
    )
    .unwrap();

    let bad_loss3 = nll_loss(&log_probs, &bad_targets3);
    assert!(bad_loss3.is_err());
    assert!(bad_loss3.unwrap_err().to_string().contains("out of range"));

    // Case 4: NaN in log probabilities
    let nan_log_probs = TestTensor::from_vec(
        vec![
            Float32::new(f32::NAN), // NaN log prob
            Float32::new(-2.3),
            Float32::new(-2.3),
            Float32::new(-2.3),
            Float32::new(-0.1),
            Float32::new(-2.3),
        ],
        &[2, 3],
    )
    .unwrap();

    let nan_loss = nll_loss(&nan_log_probs, &targets);
    assert!(nan_loss.is_err());
    assert!(nan_loss
        .unwrap_err()
        .to_string()
        .contains("Invalid log probability"));
}

#[test]
fn test_nll_loss_backward() {
    // Test that NLL loss backward pass computes correct gradients

    // Simple 2-class, 1-batch example
    let mut log_probs = TestTensor::from_vec(
        vec![
            Float32::new(0.0),            // log(P(class 0)) = log(1) = 0
            Float32::new(-f32::INFINITY), // log(P(class 1)) = log(0) = -inf (impossible)
        ],
        &[1, 2],
    )
    .unwrap();
    log_probs = log_probs.requires_grad_(true);

    let targets = TestTensor::from_vec(vec![Float32::new(0.0)], &[1]).unwrap(); // Pick class 0

    let loss = nll_loss(&log_probs, &targets).unwrap();
    assert!(loss.requires_grad());

    // Backward pass
    let grad_output = TestTensor::from_vec(vec![Float32::new(1.0)], &[]).unwrap();
    crate::ops::backward_with_grad(&loss, grad_output).unwrap();

    // Check gradients: derivative of NLL w.r.t. log_probs should be -1 at target position
    let log_probs_grad = log_probs.grad().unwrap();
    assert_relative_eq!(log_probs_grad.as_slice()[0].get(), -1.0, epsilon = 1e-6); // Target position gets -1
    assert_relative_eq!(log_probs_grad.as_slice()[1].get(), 0.0, epsilon = 1e-6);
    // Non-target positions get 0
}

#[test]
fn test_checkpoint_basic() {
    use crate::checkpointing::checkpoint;

    let input = TestTensor::from_vec(vec![Float32::new(2.0)], &[1]).unwrap();

    let result = checkpoint(
        |x: &TestTensor| {
            Ok(x.clone()) // Identity function
        },
        &input,
    )
    .unwrap();

    assert_relative_eq!(result.as_slice()[0].get(), 2.0, epsilon = 1e-6);
}

#[test]
fn test_sparse_matmul_backward() {
    let backend = CpuBackend::default();

    // Create 2x2 identity matrix (sparse)
    // [[1, 0], [0, 1]]
    let data = vec![Float32::new(1.0), Float32::new(1.0)];
    let indices = vec![0, 1];
    let indptr = vec![0, 1, 2];
    let shape = vec![2, 2];
    let storage = CsrStorage::new(data, indices, indptr, &shape).unwrap();

    let lhs = SparseTestTensor::from_storage(storage.clone(), backend.clone());
    let lhs = lhs.requires_grad_(true);

    let rhs = SparseTestTensor::from_storage(storage.clone(), backend.clone());
    let rhs = rhs.requires_grad_(true);

    // Perform matmul: I * I = I
    let result = matmul(&lhs, &rhs).unwrap();

    // Verify forward result
    let result_dense = result.to_dense_generic().unwrap();
    let result_data = result_dense.as_slice();

    assert!(
        (result_data[0].get() - 1.0).abs() < 1e-6,
        "Element (0,0) mismatch"
    );
    assert!(
        (result_data[1].get() - 0.0).abs() < 1e-6,
        "Element (0,1) mismatch"
    );
    assert!(
        (result_data[2].get() - 0.0).abs() < 1e-6,
        "Element (1,0) mismatch"
    );
    assert!(
        (result_data[3].get() - 1.0).abs() < 1e-6,
        "Element (1,1) mismatch"
    );

    // Backward pass
    // Create sparse gradient of ones (dense representation in CSR)
    // [[1, 1], [1, 1]] -> indices [0, 1, 0, 1], indptr [0, 2, 4]
    let grad_data = vec![Float32::new(1.0); 4];
    let grad_indices = vec![0, 1, 0, 1];
    let grad_indptr = vec![0, 2, 4];
    let grad_shape = vec![2, 2];
    let grad_storage = CsrStorage::new(grad_data, grad_indices, grad_indptr, &grad_shape).unwrap();
    let grad_output = SparseTestTensor::from_storage(grad_storage, backend.clone());

    backward_with_grad(&result, grad_output).unwrap();

    // Verify gradients
    // For C = A * B, if A=I, B=I, dL/dC=Ones
    // dL/dA = Ones * I^T = Ones
    // dL/dB = I^T * Ones = Ones

    let lhs_grad = lhs.grad().unwrap();
    // lhs_grad is SparseTestTensor (CsrStorage)

    let lhs_grad_dense = lhs_grad.to_dense_generic().unwrap();
    let grad_data = lhs_grad_dense.as_slice();
    for v in grad_data.iter().take(4) {
        assert_relative_eq!(v.get(), 1.0, epsilon = 1e-6);
    }
}

// Property-based testing generators and tests for gradient correctness
#[cfg(test)]
mod proptest_tests {
    use super::*;
    use num_traits::ToPrimitive;

    /// Generate random tensor data with reasonable values for autograd
    fn arb_tensor_data(size: usize) -> impl Strategy<Value = Vec<Float32>> {
        prop::collection::vec(
            prop::num::f32::NORMAL.prop_map(|f| Float32::new(f.clamp(-1e3, 1e3))),
            size,
        )
    }

    /// Generate compatible log-probabilities and targets for NLL loss
    fn arb_nll_loss_pair() -> impl Strategy<Value = (TestTensor, TestTensor)> {
        (1..=4usize, 2..=6usize).prop_flat_map(|(batch_size, num_classes)| {
            let logprobs_size = batch_size * num_classes;
            // Generate log probabilities in reasonable range [-10, 0] for numerical stability
            let logprobs_data =
                prop::collection::vec((-10.0..0.0f32).prop_map(Float32::new), logprobs_size);
            let targets_data = prop::collection::vec(
                (0..num_classes).prop_map(|i| {
                    let Some(i_f32) = i.to_f32() else {
                        panic!("target class index {i} cannot be represented as f32");
                    };
                    Float32::new(i_f32)
                }),
                batch_size,
            );

            (
                logprobs_data,
                targets_data,
                Just(batch_size),
                Just(num_classes),
            )
                .prop_map(move |(lp_data, t_data, bsz, ncls)| {
                    (
                        Tensor::from_vec(lp_data, &[bsz, ncls]).unwrap(),
                        Tensor::from_vec(t_data, &[bsz]).unwrap(),
                    )
                })
        })
    }

    /// Generate tensor pairs of same shape for element-wise operations
    fn arb_tensor_pair_same_shape() -> impl Strategy<Value = (TestTensor, TestTensor)> {
        (1..=100usize, 1..=100usize).prop_flat_map(|(size, _)| {
            let data1 = arb_tensor_data(size);
            let data2 = arb_tensor_data(size);

            (data1, data2, Just(size)).prop_map(|(d1, d2, sz)| {
                (
                    Tensor::from_vec(d1, &[sz]).unwrap(),
                    Tensor::from_vec(d2, &[sz]).unwrap(),
                )
            })
        })
    }

    /// Generate pairs for matrix multiplication
    fn arb_matmul_pair() -> impl Strategy<Value = (TestTensor, TestTensor)> {
        (1..=10usize, 1..=10usize, 1..=10usize).prop_flat_map(|(m, k, n)| {
            let a_size = m * k;
            let b_size = k * n;
            let a_data = arb_tensor_data(a_size);
            let b_data = arb_tensor_data(b_size);

            (a_data, b_data, Just(m), Just(k), Just(n)).prop_map(move |(a_d, b_d, m_, k_, n_)| {
                (
                    Tensor::from_vec(a_d, &[m_, k_]).unwrap(),
                    Tensor::from_vec(b_d, &[k_, n_]).unwrap(),
                )
            })
        })
    }

    proptest! {
        /// Test gradient correctness for addition: d/dx (a + b) = 1 for both inputs
        #[test]
        fn test_addition_gradient_correctness((ref a, ref b) in arb_tensor_pair_same_shape()) {
            let a_grad: TestTensor = a.clone().requires_grad_(true);
            let b_grad: TestTensor = b.clone().requires_grad_(true);

            println!("a_grad requires_grad: {}, b_grad requires_grad: {}", a_grad.requires_grad(), b_grad.requires_grad());

            let c = add(&a_grad, &b_grad).unwrap();

            println!("c requires_grad: {}, c grad_fn: {:?}", c.requires_grad(), c.grad_fn());

            // Backward with unit gradient
            let grad_out = Tensor::ones(c.shape().dims()).unwrap();
            println!("Calling backward_with_grad");
            backward_with_grad(&c, grad_out).unwrap();

            // Check that gradients are all ones (broadcasted)
            let a_actual_grad = a_grad.grad().unwrap();
            let b_actual_grad = b_grad.grad().unwrap();

            for i in 0..a_actual_grad.len() {
                assert_relative_eq!(a_actual_grad.as_slice()[i].get(), 1.0, epsilon = 1e-6);
            }
            for i in 0..b_actual_grad.len() {
                assert_relative_eq!(b_actual_grad.as_slice()[i].get(), 1.0, epsilon = 1e-6);
            }
        }

        /// Test gradient correctness for multiplication
        #[test]
        fn test_multiplication_gradient_correctness((ref a, ref b) in arb_tensor_pair_same_shape()) {
            let a_grad: TestTensor = a.clone().requires_grad_(true);
            let b_grad: TestTensor = b.clone().requires_grad_(true);

            let c = mul(&a_grad, &b_grad).unwrap();

            // Backward with unit gradient
            let grad_out = Tensor::ones(c.shape().dims()).unwrap();
            backward_with_grad(&c, grad_out).unwrap();

            // Check gradients: d/dx (a*b) = b, d/dy (a*b) = a
            let a_actual_grad = a_grad.grad().unwrap();
            let b_actual_grad = b_grad.grad().unwrap();

            for i in 0..a_actual_grad.len() {
                assert_relative_eq!(
                    a_actual_grad.as_slice()[i].get(),
                    b.as_slice()[i].get(),
                    epsilon = 1e-5
                );
                assert_relative_eq!(
                    b_actual_grad.as_slice()[i].get(),
                    a.as_slice()[i].get(),
                    epsilon = 1e-5
                );
            }
        }

        /// Test gradient correctness for matrix multiplication
        #[test]
        fn test_matmul_gradient_correctness((ref a, ref b) in arb_matmul_pair()) {
            let a_grad: TestTensor = a.clone().requires_grad_(true);
            let b_grad: TestTensor = b.clone().requires_grad_(true);

            let c = matmul(&a_grad, &b_grad).unwrap();

            // Backward with unit gradient
            let grad_out = Tensor::ones(c.shape().dims()).unwrap();
            backward_with_grad(&c, grad_out).unwrap();

            // Gradients should be computed without errors
            let a_actual_grad = a_grad.grad().unwrap();
            let b_actual_grad = b_grad.grad().unwrap();

            // For now, just ensure no panics and finite gradients
            prop_assert!(a_actual_grad.as_slice().iter().all(|&x: &Float32| x.is_finite()));
            prop_assert!(b_actual_grad.as_slice().iter().all(|&x: &Float32| x.is_finite()));
        }

        /// Test NLL Loss gradient correctness
        #[test]
        fn test_nll_loss_gradient_correctness((ref log_probs, ref targets) in arb_nll_loss_pair()) {
            let log_probs_grad = log_probs.clone().requires_grad_(true);

            // Extract dimensions from tensors
            let batch_size = log_probs.shape().dims()[0];
            let n_classes = log_probs.shape().dims()[1];

            // Forward pass
            let loss = nll_loss(&log_probs_grad, targets).unwrap();

            // Backward pass
            let grad_output = Tensor::ones(loss.shape().dims()).unwrap();
            backward_with_grad(&loss, grad_output).unwrap();

            let grad = log_probs_grad.grad().unwrap();
            let grad_slice = grad.as_slice();
            let targets_slice = targets.as_slice();

            // Verify gradients: -1/N at target index, 0 elsewhere
            let Some(batch_size_f32) = batch_size.to_f32() else {
                panic!("batch_size {batch_size} cannot be represented as f32");
            };
            let scale = -1.0 / batch_size_f32;

            for (i, target) in targets_slice.iter().enumerate() {
                let Some(target_idx) = target.to_usize() else {
                    panic!("target value {} cannot be converted to usize", target.get());
                };
                for c in 0..n_classes {
                    let flat_idx = i * n_classes + c;
                    let expected = if c == target_idx { scale } else { 0.0 };
                    // Use a slightly larger epsilon for accumulated errors or float precision
                    assert!(
                        (grad_slice[flat_idx].get() - expected).abs() < 1e-4,
                        "Gradient mismatch at batch {} class {}: expected {}, got {}",
                        i,
                        c,
                        expected,
                        grad_slice[flat_idx].get()
                    );
                }
            }
        }

        /// Test gradient stability: gradients should be finite for reasonable inputs
        #[test]
        fn test_gradient_stability((ref a, ref b) in arb_tensor_pair_same_shape()) {
            // Test various operations for gradient stability
            let a_grad = a.clone().requires_grad_(true);
            let b_grad = b.clone().requires_grad_(true);

            // Test addition
            let add_result = add(&a_grad, &b_grad).unwrap();
            let grad_out = Tensor::ones(add_result.shape().dims()).unwrap();
            let add_backward = backward_with_grad(&add_result, grad_out.clone());
            if add_backward.is_ok() {
                let add_grad_a = a_grad.grad().unwrap();
                let add_grad_b = b_grad.grad().unwrap();
                prop_assert!(add_grad_a.as_slice().iter().all(|&x: &Float32| x.is_finite()));
                prop_assert!(add_grad_b.as_slice().iter().all(|&x: &Float32| x.is_finite()));
            }

            // Reset gradients
            a_grad.zero_grad().unwrap();
            b_grad.zero_grad().unwrap();

            // Test multiplication
            let mul_result = mul(&a_grad, &b_grad).unwrap();
            let mul_backward = backward_with_grad(&mul_result, grad_out);
            if mul_backward.is_ok() {
                let mul_grad_a = a_grad.grad().unwrap();
                let mul_grad_b = b_grad.grad().unwrap();
                prop_assert!(mul_grad_a.as_slice().iter().all(|&x: &Float32| x.is_finite()));
                prop_assert!(mul_grad_b.as_slice().iter().all(|&x: &Float32| x.is_finite()));
            }

            // Reset gradients
            a_grad.zero_grad().unwrap();
            b_grad.zero_grad().unwrap();
        }

        /// Test gradient accumulation: gradients should add up correctly
        #[test]
        fn test_gradient_accumulation_property((ref a, ref b) in arb_tensor_pair_same_shape()) {
            let a_grad = a.clone().requires_grad_(true);

            // Create multiple loss terms that depend on 'a'
            let loss1 = add(&a_grad, b).unwrap();
            let loss2 = add(&a_grad, b).unwrap();

            // Backward both losses
            let grad_out = Tensor::from_vec(vec![Float32::new(1.0); a.len()], &[a.len()]).unwrap();
            backward_with_grad(&loss1, grad_out.clone()).unwrap();
            backward_with_grad(&loss2, grad_out).unwrap();

            // Gradient should be accumulated (2.0 for each element)
            let accumulated_grad = a_grad.grad().unwrap();
            for i in 0..accumulated_grad.len() {
                assert_relative_eq!(accumulated_grad.as_slice()[i].get(), 2.0, epsilon = 1e-6);
            }
        }

        /// Test that non-differentiable operations don't break the system
        #[test]
        fn test_non_differentiable_operations_robustness((ref a, ref b) in arb_tensor_pair_same_shape()) {
            // Test operations that don't require gradients
            let result = add(a, b).unwrap();
            prop_assert!(!result.requires_grad());

            // Even without gradients, operations should not panic
            let grad_out = Tensor::ones(result.shape().dims()).unwrap();
            let backward_result = backward_with_grad(&result, grad_out);
            prop_assert!(backward_result.is_ok()); // Should succeed gracefully
        }
    }
}
