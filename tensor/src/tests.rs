use super::*;
use crate::ops::arithmetic::*;
use crate::ops::{pow, abs};
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

    let result = crate::ops::linalg::matmul(&a, &b).unwrap();
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
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[2, 3, 4]).unwrap();
    assert_eq!(tensor.shape().dims(), &[2, 3, 4]);
    assert_eq!(tensor.len(), 24);
    // All elements should be zero
    for &val in tensor.as_slice() {
        assert_eq!(val.get(), 0.0);
    }
}
