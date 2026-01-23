use coeus_sparse::activation::gelu::SparseGelu;
use coeus_sparse::activation::tanh::SparseTanh;
use backend::gpu::GpuBackend;
use backend::Backend;
use dtype::float::Float32;
use storage::{CsrStorage, Storage};
use dtype::num_traits::Float;

#[test]
fn test_gpu_sparse_gelu() {
    let backend = GpuBackend::<Float32>::new();
    if backend.device_name() == "cpu" {
        // Just verify it compiles and runs on fallback
    }

    // Sparse vector: [0, 1.0, 0, -1.0, 2.0]
    // Indices: [1, 3, 4]
    // Values: [1.0, -1.0, 2.0]
    let values = vec![Float32(1.0), Float32(-1.0), Float32(2.0)];
    let indices = vec![1, 3, 4];
    let indptr = vec![0, 3]; // 1 row
    
    let csr = CsrStorage::new(values.clone(), indices, indptr, &[1, 5]).unwrap();
    
    let result = csr.gelu_sparse(&backend).unwrap();
    
    // Check structure constraints preserved
    assert_eq!(result.indices(), csr.indices());
    assert_eq!(result.indptr(), csr.indptr());
    
    // Check values match approx GELU
    // GELU(1.0) ≈ 0.8413
    // GELU(-1.0) ≈ -0.1587
    // GELU(2.0) ≈ 1.9545
    let res_vals = result.data();
    assert!((res_vals[0].0 - 0.8413).abs() < 1e-3);
    assert!((res_vals[1].0 - -0.1586).abs() < 1e-3);
    assert!((res_vals[2].0 - 1.9545).abs() < 1e-3);
}

#[test]
fn test_gpu_sparse_tanh() {
    let backend = GpuBackend::<Float32>::new();
    
    // Sparse vector: [0, 1.0, 0, -1.0]
    let values = vec![Float32(1.0), Float32(-1.0)];
    let indices = vec![1, 3];
    let indptr = vec![0, 2];
    
    let csr = CsrStorage::new(values.clone(), indices, indptr, &[1, 4]).unwrap();
    
    let result = csr.tanh_sparse(&backend).unwrap();
    
    // Tanh(1.0) ≈ 0.7616
    // Tanh(-1.0) ≈ -0.7616
    let res_vals = result.data();
    assert!((res_vals[0].0 - 0.7616).abs() < 1e-3);
    assert!((res_vals[1].0 - -0.7616).abs() < 1e-3);
}

#[test]
fn test_gpu_sparse_relu() {
    let backend = GpuBackend::<Float32>::new();
    
    // Sparse vector: [-1.0, 0, 1.0, 2.0]
    let values = vec![Float32(-1.0), Float32(1.0), Float32(2.0)];
    let indices = vec![0, 2, 3];
    let indptr = vec![0, 3];
    
    let csr = CsrStorage::new(values, indices, indptr, &[1, 4]).unwrap();
    
    let result = coeus_sparse::activation::relu::SparseRelu::relu_sparse(&csr, &backend).unwrap();
    
    let res_vals = result.data();
    assert_eq!(res_vals[0].0, 0.0);
    assert_eq!(res_vals[1].0, 1.0);
    assert_eq!(res_vals[2].0, 2.0);
}

#[test]
fn test_gpu_sparse_sigmoid() {
    let backend = GpuBackend::<Float32>::new();
    
    // Sparse vector: [0, 1.0]
    let values = vec![Float32(1.0)];
    let indices = vec![1];
    let indptr = vec![0, 1];
    
    let csr = CsrStorage::new(values, indices, indptr, &[1, 2]).unwrap();
    
    let result = coeus_sparse::activation::sigmoid::SparseSigmoid::sigmoid_sparse(&csr, &backend).unwrap();
    
    // Sigmoid(1.0) ≈ 0.731
    let res_vals = result.data();
    assert!((res_vals[0].0 - 0.731).abs() < 1e-3);
}
