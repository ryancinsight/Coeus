// Test file to verify generic abstraction capabilities work correctly
// This file tests that all components can be instantiated with different B<S<T>> combinations

use coeus_backend::CpuBackend;
use coeus_storage::{DenseStorage, CsrStorage};
use coeus_dtype::float::Float32;
use coeus_nn::{Sequential, Linear, MSELoss};
use coeus_nn::loss::mse_loss;

fn test_sequential_dense() {
    // Test Sequential with dense storage
    let mut seq = Sequential::<CpuBackend, DenseStorage<Float32>, Float32>::new();
    seq.add_module("linear".to_string(), Linear::<CpuBackend, DenseStorage<Float32>, Float32>::new(10, 5).unwrap());

    let input = coeus_tensor::Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[2, 10]).unwrap();
    let output = seq.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[2, 5]);
}

fn test_sequential_sparse() {
    // Test Sequential with sparse storage (if sparse components are implemented)
    // This would test the sparse pathway once sparse NN components are available
    // let mut seq = Sequential::<CpuBackend, CsrStorage<Float32>, Float32>::new();
    // For now, just test that the type compiles
    let _seq: Sequential<CpuBackend, CsrStorage<Float32>, Float32> = Sequential::new();
}

fn test_loss_dense() {
    // Test loss functions with dense storage
    let predictions = coeus_tensor::Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0)],
        &[2]
    ).unwrap();
    let targets = coeus_tensor::Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.5), Float32::new(2.5)],
        &[2]
    ).unwrap();

    let loss = mse_loss(&predictions, &targets).unwrap();
    assert_eq!(loss.shape().dims(), &[]);
}

fn main() {
    test_sequential_dense();
    test_sequential_sparse();
    test_loss_dense();
    println!("All generic abstraction tests passed!");
}
