//! Concurrency tests to verify thread safety of tensor operations
//!
//! These tests verify that tensor operations work correctly under concurrent access.

use backend::CpuBackend;
use dtype::float::Float32;
use std::sync::Arc;
use std::thread;
use storage::DenseStorage;
use tensor::Tensor;

type CpuTensorF32 = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

/// Test concurrent reads of a shared tensor
#[test]
fn test_concurrent_tensor_reads() {
    let tensor = Arc::new(create_test_tensor());

    let tensor_clone1 = Arc::clone(&tensor);
    let tensor_clone2 = Arc::clone(&tensor);

    let handle1 = thread::spawn(move || {
        let _data = tensor_clone1.as_slice();
        let _shape = tensor_clone1.shape();
        let _numel = tensor_clone1.numel();
    });

    let handle2 = thread::spawn(move || {
        let _data = tensor_clone2.as_slice();
        let _shape = tensor_clone2.shape();
        let _numel = tensor_clone2.numel();
    });

    handle1.join().unwrap();
    handle2.join().unwrap();
}

/// Test concurrent arithmetic operations on separate tensors
#[test]
fn test_concurrent_arithmetic_separate_tensors() {
    let tensor1 = create_test_tensor();
    let tensor2 = create_test_tensor();
    let tensor3 = create_test_tensor();
    let tensor4 = create_test_tensor();

    let handle1 = thread::spawn(move || {
        let _result = &tensor1 + &tensor2;
    });

    let handle2 = thread::spawn(move || {
        let _result = &tensor3 * &tensor4;
    });

    handle1.join().unwrap();
    handle2.join().unwrap();
}

/// Test concurrent operations on cloned tensors
#[test]
fn test_concurrent_operations_cloned_tensors() {
    let base_tensor = create_test_tensor();

    let tensor1 = base_tensor.clone();
    let tensor2 = base_tensor.clone();

    let handle1 = thread::spawn(move || {
        let _result = &tensor1 + &tensor1; // Add to itself
    });

    let handle2 = thread::spawn(move || {
        let _result = &tensor2 * &tensor2; // Multiply by itself
    });

    handle1.join().unwrap();
    handle2.join().unwrap();
}

/// Test concurrent SIMD operations
#[test]
fn test_concurrent_simd_operations() {
    let tensor1 = create_test_tensor();
    let tensor2 = create_test_tensor();

    let handle1 = thread::spawn(move || {
        let _result = tensor1.relu_simd().unwrap();
    });

    let handle2 = thread::spawn(move || {
        let _result = tensor2.sum_simd().unwrap();
    });

    handle1.join().unwrap();
    handle2.join().unwrap();
}

/// Test concurrent broadcasting operations
#[test]
fn test_concurrent_broadcasting() {
    let tensor = create_test_tensor();
    let scalar = vec![Float32::new(2.0)];
    let scalar_tensor = CpuTensorF32::from_vec(scalar, &[1]).unwrap();

    let tensor_clone1 = tensor.clone();
    let scalar_clone1 = scalar_tensor.clone();

    let tensor_clone2 = tensor.clone();
    let scalar_clone2 = scalar_tensor.clone();

    let handle1 = thread::spawn(move || {
        let _result = &tensor_clone1 + &scalar_clone1;
    });

    let handle2 = thread::spawn(move || {
        let _result = &tensor_clone2 * &scalar_clone2;
    });

    handle1.join().unwrap();
    handle2.join().unwrap();
}

/// Test concurrent chunking operations
#[test]
fn test_concurrent_chunking() {
    let tensor = create_test_tensor();

    let tensor_clone1 = tensor.clone();
    let tensor_clone2 = tensor.clone();

    let handle1 = thread::spawn(move || {
        let chunks: Vec<_> = tensor_clone1.chunks(0, 2).collect();
        assert!(!chunks.is_empty());
    });

    let handle2 = thread::spawn(move || {
        let _view = tensor_clone2.backend_clone();
    });

    handle1.join().unwrap();
    handle2.join().unwrap();
}

/// Helper function to create a test tensor
fn create_test_tensor() -> CpuTensorF32 {
    let data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
    ];
    CpuTensorF32::from_vec(data, &[2, 2]).unwrap()
}
