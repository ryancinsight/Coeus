use coeus_core::SequentialBackend;
use coeus_cuda::CudaBackend;
use coeus_tensor::{Tensor, Transpose};

#[test]
fn test_cuda_strided_ops() {
    if !crate::availability::device_available() {
        return;
    }
    let seq = SequentialBackend::new();
    let cuda_b = CudaBackend::new();

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![10.0f32, 20.0, 30.0];

    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &a_data)
        .expect("invariant: test backend operation succeeds");
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 1], &b_data)
        .expect("invariant: test backend operation succeeds");

    let a_seq_t = a_seq.transpose();

    let a_cuda_t = a_seq_t
        .to_backend_on(&seq, &cuda_b)
        .expect("invariant: test backend transfer succeeds");
    let b_cuda = b_seq
        .to_backend_on(&seq, &cuda_b)
        .expect("invariant: test backend transfer succeeds");

    assert_eq!(a_cuda_t.shape(), &[3, 2]);
    assert_eq!(b_cuda.shape(), &[3, 1]);

    let c_cuda =
        coeus_ops::add(&a_cuda_t, &b_cuda, &cuda_b).expect("invariant: test operation succeeds");
    let c_seq = c_cuda
        .to_backend_on(&cuda_b, &seq)
        .expect("invariant: test backend transfer succeeds");

    let c_expected =
        coeus_ops::add(&a_seq_t, &b_seq, &seq).expect("invariant: test operation succeeds");

    for (i, (&res, &exp)) in c_seq
        .as_slice()
        .iter()
        .zip(c_expected.as_slice().iter())
        .enumerate()
    {
        assert!(
            (res - exp).abs() < 1e-4f32,
            "Strided add mismatch at {}: {} vs {}",
            i,
            res,
            exp
        );
    }

    let u_cuda = coeus_ops::relu(&a_cuda_t, &cuda_b).expect("invariant: test operation succeeds");
    let u_seq = u_cuda
        .to_backend_on(&cuda_b, &seq)
        .expect("invariant: test backend transfer succeeds");
    let u_expected = coeus_ops::relu(&a_seq_t, &seq).expect("invariant: test operation succeeds");

    for (i, (&res, &exp)) in u_seq
        .as_slice()
        .iter()
        .zip(u_expected.as_slice().iter())
        .enumerate()
    {
        assert!(
            (res - exp).abs() < 1e-4f32,
            "Strided relu mismatch at {}: {} vs {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_cuda_strided_activation_tail_matches_cpu() {
    if !crate::availability::device_available() {
        return;
    }

    let seq = SequentialBackend::new();
    let cuda = CudaBackend::new();
    let data = [
        -2.0_f32, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0,
    ];
    let cpu = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 4], &data)
        .expect("invariant: test backend operation succeeds")
        .transpose();
    let gpu_base = Tensor::<f32, SequentialBackend>::from_slice(vec![3, 4], &data)
        .expect("invariant: test backend operation succeeds")
        .to_backend_on(&seq, &cuda)
        .expect("invariant: test backend transfer succeeds");
    let gpu = gpu_base.transpose();

    for operation in [
        coeus_ops::UnaryOp::Mish,
        coeus_ops::UnaryOp::MishGrad,
        coeus_ops::UnaryOp::Elu,
        coeus_ops::UnaryOp::EluGrad,
    ] {
        let expected = coeus_ops::elementwise_unary(&cpu, &seq, operation)
            .expect("valid CPU strided activation-tail input");
        let actual = coeus_ops::elementwise_unary(&gpu, &cuda, operation)
            .expect("valid CUDA strided activation-tail input")
            .to_backend_on(&cuda, &seq)
            .expect("invariant: test backend transfer succeeds");

        for (index, (&reference, &candidate)) in expected
            .as_slice()
            .iter()
            .zip(actual.as_slice())
            .enumerate()
        {
            let tolerance = f32::EPSILON * 512.0 * reference.abs().max(1.0);
            assert!(
                (candidate - reference).abs() <= tolerance,
                "{operation:?} mismatch at {index}: CPU={reference}, CUDA={candidate}, tolerance={tolerance}"
            );
        }
    }
}

#[test]
fn test_cuda_f64_comparisons_match_cpu() {
    if !crate::availability::device_available() {
        return;
    }

    let sequential = SequentialBackend::new();
    let cuda = CudaBackend::new();
    let lhs_values = [-2.0_f64, -0.5, 0.0, 0.5, 2.0, 3.0];
    let rhs_values = [-1.0_f64, -0.5, 0.25, 0.5, 4.0, 3.0];
    let lhs = Tensor::<f64, SequentialBackend>::from_slice(vec![2, 3], &lhs_values)
        .expect("invariant: test backend operation succeeds")
        .transpose();
    let rhs = Tensor::<f64, SequentialBackend>::from_slice(vec![2, 3], &rhs_values)
        .expect("invariant: test backend operation succeeds")
        .transpose();
    let lhs_cuda = lhs
        .to_backend_on(&sequential, &cuda)
        .expect("invariant: test backend transfer succeeds");
    let rhs_cuda = rhs
        .to_backend_on(&sequential, &cuda)
        .expect("invariant: test backend transfer succeeds");

    for (operation, cpu, gpu) in [
        (
            "eq",
            coeus_ops::eq(&lhs, &rhs, &sequential).expect("invariant: test operation succeeds"),
            coeus_ops::eq(&lhs_cuda, &rhs_cuda, &cuda).expect("invariant: test operation succeeds"),
        ),
        (
            "ne",
            coeus_ops::ne(&lhs, &rhs, &sequential).expect("invariant: test operation succeeds"),
            coeus_ops::ne(&lhs_cuda, &rhs_cuda, &cuda).expect("invariant: test operation succeeds"),
        ),
        (
            "lt",
            coeus_ops::lt(&lhs, &rhs, &sequential).expect("invariant: test operation succeeds"),
            coeus_ops::lt(&lhs_cuda, &rhs_cuda, &cuda).expect("invariant: test operation succeeds"),
        ),
        (
            "gt",
            coeus_ops::gt(&lhs, &rhs, &sequential).expect("invariant: test operation succeeds"),
            coeus_ops::gt(&lhs_cuda, &rhs_cuda, &cuda).expect("invariant: test operation succeeds"),
        ),
        (
            "le",
            coeus_ops::le(&lhs, &rhs, &sequential).expect("invariant: test operation succeeds"),
            coeus_ops::le(&lhs_cuda, &rhs_cuda, &cuda).expect("invariant: test operation succeeds"),
        ),
        (
            "ge",
            coeus_ops::ge(&lhs, &rhs, &sequential).expect("invariant: test operation succeeds"),
            coeus_ops::ge(&lhs_cuda, &rhs_cuda, &cuda).expect("invariant: test operation succeeds"),
        ),
    ] {
        let gpu = gpu
            .to_backend_on(&cuda, &sequential)
            .expect("invariant: test backend transfer succeeds");
        assert_eq!(
            gpu.as_slice(),
            cpu.as_slice(),
            "f64 CUDA comparison {operation}"
        );
    }
}

#[test]
fn test_cuda_strided_parameterized_activations_match_cpu() {
    if !crate::availability::device_available() {
        return;
    }
    let sequential = SequentialBackend::new();
    let backend = CudaBackend::new();
    let values = [-2.0_f32, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0];
    let input = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 4], &values)
        .expect("invariant: test backend operation succeeds")
        .transpose();
    let device_input = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 4], &values)
        .expect("invariant: test backend operation succeeds")
        .to_backend_on(&sequential, &backend)
        .expect("invariant: test backend transfer succeeds")
        .transpose();
    let hardtanh = u64::from((-1.0_f32).to_bits()) | (u64::from(1.0_f32.to_bits()) << 32);
    let threshold = u64::from(0.25_f32.to_bits()) | (u64::from((-0.5_f32).to_bits()) << 32);

    for operation in [
        coeus_ops::UnaryOp::Hardtanh(hardtanh),
        coeus_ops::UnaryOp::HardtanhGrad(hardtanh),
        coeus_ops::UnaryOp::Threshold(threshold),
        coeus_ops::UnaryOp::ThresholdGrad(threshold),
    ] {
        let expected = coeus_ops::elementwise_unary(&input, &sequential, operation)
            .expect("valid CPU strided parameterized activation");
        let actual = coeus_ops::elementwise_unary(&device_input, &backend, operation)
            .expect("valid CUDA strided parameterized activation")
            .to_backend_on(&backend, &sequential)
            .expect("invariant: test backend transfer succeeds");

        assert_eq!(actual.as_slice(), expected.as_slice(), "{operation:?}");
    }
}
