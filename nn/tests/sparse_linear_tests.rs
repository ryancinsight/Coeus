//! Tests for SparseLinear layer

use coeus_backend::CpuBackend;
use coeus_dtype::float::Float32;
use coeus_nn::{Linear, Module, Parameter, SparseLinear};
use coeus_storage::Storage;
use coeus_storage::{CsrStorage, DenseStorage};
use coeus_tensor::Tensor;

// Note: Direct CsrStorage parameter creation has architectural issues
// SparseLinear uses dense storage with sparse computation pattern

#[test]
fn test_sparse_linear_creation() {
    // Create a sparse linear layer with 10 input features, 5 output features, 90% sparsity
    let sparse_layer = SparseLinear::<CpuBackend<Float32>, Float32>::new(10, 5, 0.9, true).unwrap();

    // Check dimensions
    assert_eq!(sparse_layer.in_features, 10);
    assert_eq!(sparse_layer.out_features, 5);
    assert!((sparse_layer.sparsity - 0.9).abs() < 1e-6);

    // Check that weight is sparse (should have much less than 50 non-zero elements)
    let weight_nnz = sparse_layer.csr_data.as_ref().unwrap().data.len();
    println!("Weight nnz: {}", weight_nnz);
    assert!(
        weight_nnz < 25,
        "Weight should be sparse, but has {} non-zero elements",
        weight_nnz
    );

    // Check sparsity ratio (random initialization is approximate)
    let total_elements = 10 * 5;
    let actual_sparsity = 1.0 - (weight_nnz as f64 / total_elements as f64);
    assert!(
        actual_sparsity > 0.5,
        "Sparsity should be > 0.5, but got {}",
        actual_sparsity
    );
}

#[test]
fn test_sparse_linear_forward() {
    // Create sparse linear layer
    let sparse_layer = SparseLinear::<CpuBackend<Float32>, Float32>::new(4, 3, 0.8, true).unwrap();

    // Create input tensor [batch_size=2, in_features=4]
    let input_data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0), // batch 0
        Float32::new(5.0),
        Float32::new(6.0),
        Float32::new(7.0),
        Float32::new(8.0), // batch 1
    ];
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[2, 4],
    )
    .unwrap();

    // Forward pass
    let output = sparse_layer.forward(&input).unwrap();

    // Check output shape
    assert_eq!(output.shape().dims(), &[2, 3]);

    // Output should be non-zero (since input is non-zero and weights are initialized)
    let output_slice = output.storage_ref().as_slice();
    assert!(!output_slice.iter().all(|&x| x.0 == 0.0));
}

#[test]
fn test_sparse_linear_memory_efficiency() {
    // Compare memory usage of sparse vs dense layers
    let in_features = 1000;
    let out_features = 500;
    let sparsity = 0.95;

    let sparse_layer = SparseLinear::<CpuBackend<Float32>, Float32>::new(
        in_features,
        out_features,
        sparsity,
        false,
    )
    .unwrap();
    let dense_layer = coeus_nn::Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        out_features,
        in_features,
    )
    .unwrap();

    // Check sparse layer has much fewer stored elements
    let sparse_nnz = sparse_layer.csr_data.as_ref().unwrap().data.len();
    let dense_elements = in_features * out_features;

    let expected_nnz = ((1.0 - sparsity) * dense_elements as f64) as usize;
    assert!(
        sparse_nnz <= expected_nnz + 1000,
        "Sparse layer nnz {} should be close to expected {}",
        sparse_nnz,
        expected_nnz
    );
    assert!(
        sparse_nnz < dense_elements / 2,
        "Sparse layer should use much less memory: {} vs {}",
        sparse_nnz,
        dense_elements
    );
}

#[test]
fn test_sparse_linear_to_dense() {
    let sparse_layer = SparseLinear::<CpuBackend<Float32>, Float32>::new(4, 3, 0.8, true).unwrap();
    let dense_layer = sparse_layer.to_dense().unwrap();

    // Both should have same dimensions
    assert_eq!(dense_layer.in_features, sparse_layer.in_features);
    assert_eq!(dense_layer.out_features, sparse_layer.out_features);

    // The conversion should succeed and produce a valid dense layer
    assert!(dense_layer.in_features > 0);
    assert!(dense_layer.out_features > 0);
}
