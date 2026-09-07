use coeus_core::SequentialBackend;
use coeus_cuda::CudaBackend;
use coeus_tensor::Tensor;

#[test]
fn test_cuda_evaluate_fused() {
    if !crate::availability::device_available() {
        return;
    }
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let a_data = vec![1.0f32, -2.0, 3.0, -4.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0];

    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &a_data);
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &b_data);

    let a_cuda = a_seq.to_backend_on(&seq, &cuda_b);
    let b_cuda = b_seq.to_backend_on(&seq, &cuda_b);

    use coeus_ops::fuse::TensorExprExt;
    let expr = (a_cuda.expr() * b_cuda.expr() + 5.0).relu();

    let out_cuda = coeus_cuda::evaluate_fused(&expr).expect("CUDA fused elementwise dispatch");
    let out_seq = out_cuda.to_backend_on(&cuda_b, &seq);

    let mut expected = vec![0.0f32; 4];
    for i in 0..4 {
        let val = a_data[i] * b_data[i] + 5.0;
        expected[i] = if val > 0.0 { val } else { 0.0 };
    }

    assert_eq!(out_seq.shape(), &[2, 2]);
    assert_eq!(out_seq.as_slice(), &expected);
}

#[test]
fn test_cuda_jit_fusion_correctness() {
    if !crate::availability::device_available() {
        return;
    }
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let a_data = vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let c_data = vec![-5.0f32, 5.0, -10.0, 10.0, -15.0, 15.0];
    let shape = vec![2, 3];

    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &a_data);
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &b_data);
    let c_seq = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &c_data);

    let a_cuda = a_seq.to_backend_on(&seq, &cuda_b);
    let b_cuda = b_seq.to_backend_on(&seq, &cuda_b);
    let c_cuda = c_seq.to_backend_on(&seq, &cuda_b);

    use coeus_ops::fuse::TensorExprExt;
    let expr = (a_cuda.expr() * b_cuda.expr() + c_cuda.expr())
        .relu()
        .sigmoid();

    let out_cuda = coeus_cuda::evaluate_fused(&expr).expect("CUDA fused elementwise dispatch");
    let out_seq = out_cuda.to_backend_on(&cuda_b, &seq);

    let mut expected = [0.0f32; 6];
    for i in 0..6 {
        let val = a_data[i] * b_data[i] + c_data[i];
        let relu_val = if val > 0.0 { val } else { 0.0 };
        expected[i] = 1.0 / (1.0 + (-relu_val).exp());
    }

    assert_eq!(out_seq.shape(), &[2, 3]);
    let out_slice = out_seq.as_slice();
    for i in 0..6 {
        let diff = (out_slice[i] - expected[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: {} vs expected {}",
            i,
            out_slice[i],
            expected[i]
        );
    }
}

#[test]
fn test_cuda_jit_reductions() {
    if !crate::availability::device_available() {
        return;
    }
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let a_data = vec![1.0f32, 2.0, 3.0, 10.0, -5.0, 6.0];
    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &a_data);
    let a_cuda = a_seq.to_backend_on(&seq, &cuda_b);

    let sum_cuda = coeus_ops::sum_axis(&a_cuda, 1, &cuda_b).expect("valid CUDA sum axis");
    let sum_seq = sum_cuda.to_backend_on(&cuda_b, &seq);

    let max_cuda = coeus_ops::max_axis(&a_cuda, 1, &cuda_b).expect("valid CUDA max axis");
    let max_seq = max_cuda.to_backend_on(&cuda_b, &seq);

    let min_cuda = coeus_ops::min_axis(&a_cuda, 1, &cuda_b).expect("valid CUDA min axis");
    let min_seq = min_cuda.to_backend_on(&cuda_b, &seq);

    assert_eq!(sum_seq.as_slice(), &[6.0, 11.0]);
    assert_eq!(max_seq.as_slice(), &[3.0, 10.0]);
    assert_eq!(min_seq.as_slice(), &[1.0, -5.0]);
}

#[test]
fn test_cuda_evaluate_fused_reduce() {
    if !crate::availability::device_available() {
        return;
    }
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let a_data = vec![1.0f32, -2.0, 3.0, 10.0, -5.0, 6.0];
    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &a_data);
    let a_cuda = a_seq.to_backend_on(&seq, &cuda_b);

    use coeus_ops::fuse::TensorExprExt;
    let expr = (a_cuda.expr() * 2.0).relu();

    let sum_cuda = coeus_cuda::evaluate_fused_reduce(&expr, coeus_ops::ReductionOp::Sum, 1)
        .expect("CUDA fused sum dispatch");
    let sum_seq = sum_cuda.to_backend_on(&cuda_b, &seq);

    let max_cuda = coeus_cuda::evaluate_fused_reduce(&expr, coeus_ops::ReductionOp::Max, 1)
        .expect("CUDA fused max dispatch");
    let max_seq = max_cuda.to_backend_on(&cuda_b, &seq);

    let min_cuda = coeus_cuda::evaluate_fused_reduce(&expr, coeus_ops::ReductionOp::Min, 1)
        .expect("CUDA fused min dispatch");
    let min_seq = min_cuda.to_backend_on(&cuda_b, &seq);

    assert_eq!(sum_seq.shape(), &[2, 1]);
    assert_eq!(max_seq.shape(), &[2, 1]);
    assert_eq!(min_seq.shape(), &[2, 1]);

    assert_eq!(sum_seq.as_slice(), &[8.0, 32.0]);
    assert_eq!(max_seq.as_slice(), &[6.0, 20.0]);
    assert_eq!(min_seq.as_slice(), &[0.0, 0.0]);
}
