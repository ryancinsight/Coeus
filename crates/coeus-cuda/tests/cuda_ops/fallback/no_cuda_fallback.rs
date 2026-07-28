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
        Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("construct tensor");
    let b_seq =
        Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
            .expect("construct tensor");
    let a_cuda = a_seq.to_backend_on(&seq, &cuda).expect("transfer tensor");
    let b_cuda = b_seq.to_backend_on(&seq, &cuda).expect("transfer tensor");

    let add_cuda = coeus_ops::add(&a_cuda, &b_cuda, &cuda)
        .expect("evaluate addition")
        .to_backend_on(&cuda, &seq)
        .expect("transfer tensor");
    assert_eq!(add_cuda.as_slice(), &[7.0, 7.0, 7.0, 7.0, 7.0, 7.0]);

    let relu_input = Tensor::<f32, SequentialBackend>::from_slice([2, 2], &[-2.0, 3.0, -4.0, 5.0])
        .expect("construct tensor");
    let relu_cuda = relu_input
        .to_backend_on(&seq, &cuda)
        .expect("transfer tensor");
    let relu_out = coeus_ops::relu(&relu_cuda, &cuda)
        .expect("evaluate activation")
        .to_backend_on(&cuda, &seq)
        .expect("transfer tensor");
    assert_eq!(relu_out.as_slice(), &[0.0, 3.0, 0.0, 5.0]);

    let lhs = Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("construct tensor");
    let rhs =
        Tensor::<f32, SequentialBackend>::from_slice([3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .expect("construct tensor");
    let lhs_cuda = lhs.to_backend_on(&seq, &cuda).expect("transfer tensor");
    let rhs_cuda = rhs.to_backend_on(&seq, &cuda).expect("transfer tensor");
    let matmul_out = coeus_ops::matmul(&lhs_cuda, &rhs_cuda, &cuda)
        .expect("evaluate matmul")
        .to_backend_on(&cuda, &seq)
        .expect("transfer tensor");
    assert_eq!(matmul_out.as_slice(), &[58.0, 64.0, 139.0, 154.0]);

    let reduced = coeus_ops::sum_axis(&a_cuda, 1, &cuda)
        .expect("valid fallback sum axis")
        .to_backend_on(&cuda, &seq)
        .expect("transfer tensor");
    assert_eq!(reduced.as_slice(), &[6.0, 15.0]);

    let mean = coeus_ops::mean_axis(&a_cuda, 1, &cuda)
        .expect("valid fallback mean axis")
        .to_backend_on(&cuda, &seq)
        .expect("transfer tensor");
    assert_eq!(mean.as_slice(), &[2.0, 5.0]);
}

#[test]
fn fallback_fused_ops_match_expression_values() {
    let seq = SequentialBackend::new();
    let cuda = CudaBackend::new();

    let a_seq = Tensor::<f32, SequentialBackend>::from_slice([4], &[1.0, -2.0, 3.0, -4.0])
        .expect("construct tensor");
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice([4], &[2.0, 3.0, -4.0, -5.0])
        .expect("construct tensor");
    let a_cuda = a_seq.to_backend_on(&seq, &cuda).expect("transfer tensor");
    let b_cuda = b_seq.to_backend_on(&seq, &cuda).expect("transfer tensor");

    let expr = (a_cuda.expr() * b_cuda.expr() + 1.0).relu();
    let out = coeus_cuda::evaluate_fused(&expr)
        .expect("evaluate fused expression")
        .to_backend_on(&cuda, &seq)
        .expect("transfer tensor");
    assert_eq!(out.as_slice(), &[3.0, 0.0, 0.0, 21.0]);

    let reduce_expr = (a_cuda.expr() * 2.0).relu();
    let reduced = coeus_cuda::evaluate_fused_reduce(&reduce_expr, coeus_ops::ReductionOp::Sum, 0)
        .expect("evaluate fused reduction")
        .to_backend_on(&cuda, &seq)
        .expect("transfer tensor");
    assert_eq!(reduced.as_slice(), &[8.0]);

    let mean = coeus_cuda::evaluate_fused_reduce(&reduce_expr, coeus_ops::ReductionOp::Mean, 0)
        .expect("evaluate fused reduction")
        .to_backend_on(&cuda, &seq)
        .expect("transfer tensor");
    assert_eq!(mean.as_slice(), &[2.0]);
}

#[test]
fn cpu_backed_cuda_unfold_fold_preserves_adjoint_values() {
    let backend = CudaBackend::new();
    let input =
        Tensor::<f32, CudaBackend>::from_slice_on([1, 1, 5], &[1.0, 2.0, 3.0, 4.0, 5.0], &backend)
            .expect("construct tensor");
    let columns = coeus_ops::unfold1d(&input, 3, 1, 1, 1, &backend).expect("evaluate unfold");
    let sequential = SequentialBackend::new();
    assert_eq!(
        columns
            .to_backend_on(&backend, &sequential)
            .expect("transfer tensor")
            .as_slice(),
        &[0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 0.0,]
    );
    let reconstructed =
        coeus_ops::fold1d(&columns, 5, 3, 1, 1, 1, &backend).expect("evaluate fold");
    assert_eq!(
        reconstructed
            .to_backend_on(&backend, &sequential)
            .expect("transfer tensor")
            .as_slice(),
        &[2.0, 6.0, 9.0, 12.0, 10.0]
    );

    let image = Tensor::<f32, CudaBackend>::from_slice_on(
        [1, 1, 3, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &backend,
    )
    .expect("construct tensor");
    let patches =
        coeus_ops::unfold2d(&image, 2, 2, 1, 1, 0, 0, 1, 1, &backend).expect("evaluate unfold");
    assert_eq!(
        patches
            .to_backend_on(&backend, &sequential)
            .expect("transfer tensor")
            .as_slice(),
        &[1.0, 2.0, 4.0, 5.0, 2.0, 3.0, 5.0, 6.0, 4.0, 5.0, 7.0, 8.0, 5.0, 6.0, 8.0, 9.0,]
    );
    let image_reconstructed =
        coeus_ops::fold2d(&patches, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, &backend).expect("evaluate fold");
    assert_eq!(
        image_reconstructed
            .to_backend_on(&backend, &sequential)
            .expect("transfer tensor")
            .as_slice(),
        &[1.0, 4.0, 3.0, 8.0, 20.0, 12.0, 7.0, 16.0, 9.0]
    );
}
