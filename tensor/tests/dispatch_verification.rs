use tensor::Tensor;
use backend::CpuBackend;
use dtype::float::Float32;
use storage::{DenseStorage, StridedStorage, CsrStorage, StorageFromVec, StorageToDense};
use tensor::ops::{relu, tanh, neg, abs, sum, max};

#[test]
fn test_strided_dispatch_parity() {
    let backend = CpuBackend::<Float32>::default();
    
    // Create a 2x3 dense tensor
    let data = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ].into_iter().map(Float32::new).collect::<Vec<_>>();
    
    let dense_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        data.clone(),
        &[2, 3],
    ).unwrap();
    
    // Create a strided view (transpose: 3x2)
    let strided_storage = StridedStorage::from_vec(data, &[2, 3]).unwrap();
    let transposed_storage = strided_storage.transpose(None).unwrap();
    let strided_tensor = Tensor::<CpuBackend<Float32>, StridedStorage<Float32>, Float32>::from_storage(
        transposed_storage,
        backend.clone(),
    );
    
    // 1. Test Negation
    let dense_neg = neg(&dense_tensor.transpose(0, 1).unwrap()).unwrap();
    let strided_neg = neg(&strided_tensor).unwrap();
    assert_eq!(strided_neg.to_cpu_dense().unwrap().as_slice(), dense_neg.as_slice());
    
    // 2. Test ReLU
    let dense_relu = relu(&dense_tensor.transpose(0, 1).unwrap()).unwrap();
    let strided_relu = relu(&strided_tensor).unwrap();
    assert_eq!(strided_relu.to_cpu_dense().unwrap().as_slice(), dense_relu.as_slice());
    
    // 3. Test Tanh
    let dense_tanh = tanh(&dense_tensor.transpose(0, 1).unwrap()).unwrap();
    let strided_tanh = tanh(&strided_tensor).unwrap();
    let s_data = strided_tanh.to_cpu_dense().unwrap();
    let d_data = dense_tanh.to_cpu_dense().unwrap();
    for (s, d) in s_data.as_slice().iter().zip(d_data.as_slice().iter()) {
        assert!((s.get() - d.get()).abs() < 1e-6f32);
    }
    
    // 4. Test Sum
    let dense_transposed = dense_tensor.transpose(0, 1).unwrap();
    let dense_sum = sum(&dense_transposed, None, false).unwrap();
    let strided_sum = sum(&strided_tensor, None, false).unwrap();
    assert!((strided_sum.as_slice()[0].get() - dense_sum.as_slice()[0].get()).abs() < 1e-6f32);
    
    // 5. Test Max
    let dense_max = max(&dense_transposed, 0, false).unwrap();
    let strided_max = max(&strided_tensor, 0, false).unwrap();
    assert_eq!(strided_max.as_slice()[0].get(), dense_max.as_slice()[0].get());
}

#[test]
fn test_csr_dispatch_parity() {
    let backend = CpuBackend::<Float32>::default();
    
    // Create a sparse 3x3 matrix (identity-ish)
    let data = vec![
        1.0, 0.0, 2.0,
        0.0, -3.0, 0.0,
        4.0, 0.0, 0.0,
    ].into_iter().map(Float32::new).collect::<Vec<_>>();
    
    let dense_tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        data.clone(),
        &[3, 3],
    ).unwrap();
    
    let csr_storage = CsrStorage::from_dense(&dense_tensor.storage().to_dense().unwrap()).unwrap();
    let csr_tensor = Tensor::<CpuBackend<Float32>, CsrStorage<Float32>, Float32>::from_storage(
        csr_storage,
        backend.clone(),
    );
    
    // 1. Test Negation
    let dense_neg = neg(&dense_tensor).unwrap();
    let csr_neg = neg(&csr_tensor).unwrap();
    assert_eq!(csr_neg.to_cpu_dense().unwrap().as_slice(), dense_neg.as_slice());
    
    // 2. Test ReLU
    let dense_relu = relu(&dense_tensor).unwrap();
    let csr_relu = relu(&csr_tensor).unwrap();
    assert_eq!(csr_relu.to_cpu_dense().unwrap().as_slice(), dense_relu.as_slice());
    
    // 3. Test ABS
    let dense_abs = abs(&dense_tensor).unwrap();
    let csr_abs = abs(&csr_tensor).unwrap();
    assert_eq!(csr_abs.to_cpu_dense().unwrap().as_slice(), dense_abs.as_slice());
    
    // 4. Test Tanh (sparsity preserving)
    let dense_tanh = tanh(&dense_tensor).unwrap();
    let csr_tanh = tanh(&csr_tensor).unwrap();
    let s_data = csr_tanh.to_cpu_dense().unwrap();
    let d_data = dense_tanh.to_cpu_dense().unwrap();
    for (s, d) in s_data.as_slice().iter().zip(d_data.as_slice().iter()) {
        assert!((s.get() - d.get()).abs() < 1e-6f32);
    }
}
