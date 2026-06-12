#![cfg(not(feature = "cuda"))]

use coeus_core::SequentialBackend;
use coeus_cuda::CudaBackend;
use coeus_ops::TensorExprExt;
use coeus_tensor::Tensor;

#[test]
fn fallback_backend_ops_match_sequential_values() {
    let seq = SequentialBackend::new();
    let cuda = CudaBackend::new();

    let a_seq =
        Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b_seq =
        Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    let a_cuda = a_seq.to_backend_on(&seq, &cuda);
    let b_cuda = b_seq.to_backend_on(&seq, &cuda);

    let add_cuda = coeus_ops::add(&a_cuda, &b_cuda, &cuda).to_backend_on(&cuda, &seq);
    assert_eq!(add_cuda.as_slice(), &[7.0, 7.0, 7.0, 7.0, 7.0, 7.0]);

    let relu_input = Tensor::<f32, SequentialBackend>::from_slice([2, 2], &[-2.0, 3.0, -4.0, 5.0]);
    let relu_cuda = relu_input.to_backend_on(&seq, &cuda);
    let relu_out = coeus_ops::relu(&relu_cuda, &cuda).to_backend_on(&cuda, &seq);
    assert_eq!(relu_out.as_slice(), &[0.0, 3.0, 0.0, 5.0]);

    let lhs = Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs =
        Tensor::<f32, SequentialBackend>::from_slice([3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let lhs_cuda = lhs.to_backend_on(&seq, &cuda);
    let rhs_cuda = rhs.to_backend_on(&seq, &cuda);
    let matmul_out = coeus_ops::matmul(&lhs_cuda, &rhs_cuda, &cuda).to_backend_on(&cuda, &seq);
    assert_eq!(matmul_out.as_slice(), &[58.0, 64.0, 139.0, 154.0]);

    let reduced = coeus_ops::sum_axis(&a_cuda, 1, &cuda).to_backend_on(&cuda, &seq);
    assert_eq!(reduced.as_slice(), &[6.0, 15.0]);
}

#[test]
fn fallback_fused_ops_match_expression_values() {
    let seq = SequentialBackend::new();
    let cuda = CudaBackend::new();

    let a_seq = Tensor::<f32, SequentialBackend>::from_slice([4], &[1.0, -2.0, 3.0, -4.0]);
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice([4], &[2.0, 3.0, -4.0, -5.0]);
    let a_cuda = a_seq.to_backend_on(&seq, &cuda);
    let b_cuda = b_seq.to_backend_on(&seq, &cuda);

    let expr = (a_cuda.expr() * b_cuda.expr() + 1.0).relu();
    let out = coeus_cuda::evaluate_fused(&expr).to_backend_on(&cuda, &seq);
    assert_eq!(out.as_slice(), &[3.0, 0.0, 0.0, 21.0]);

    let reduce_expr = (a_cuda.expr() * 2.0).relu();
    let reduced = coeus_cuda::evaluate_fused_reduce(&reduce_expr, coeus_ops::ReductionOp::Sum, 0)
        .to_backend_on(&cuda, &seq);
    assert_eq!(reduced.as_slice(), &[8.0]);
}
