use coeus_core::{ComputeBackend, SequentialBackend};
use coeus_cuda::CudaBackend;
use coeus_tensor::Tensor;

#[test]
fn test_cuda_backend_transfer_roundtrip() {
    if !crate::availability::device_available() {
        return;
    }
    let cuda_b = CudaBackend::new();
    assert_eq!(cuda_b.name(), "cuda");

    let data = vec![1.0f32, 2.0, 3.0];
    let seq = SequentialBackend::new();
    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![3], &data);

    let a_cuda = a_seq.to_backend_on(&seq, &cuda_b);
    assert_eq!(a_cuda.shape(), &[3]);

    let a_seq_back = a_cuda.to_backend_on(&cuda_b, &seq);
    assert_eq!(a_seq_back.shape(), &[3]);
}

#[expect(
    clippy::excessive_precision,
    reason = "ratchet ATLAS-COEUS-LINT-RATCHET-097"
)]
#[test]
fn test_cuda_backend_ops() {
    if !crate::availability::device_available() {
        return;
    }
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0];

    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &a_data);
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &b_data);

    let a_cuda = a_seq.to_backend_on(&seq, &cuda_b);
    let b_cuda = b_seq.to_backend_on(&seq, &cuda_b);

    let c_cuda = coeus_ops::add(&a_cuda, &b_cuda, &cuda_b);
    let c_seq = c_cuda.to_backend_on(&cuda_b, &seq);

    let m_cuda = coeus_ops::matmul(&a_cuda, &b_cuda, &cuda_b);
    let m_seq = m_cuda.to_backend_on(&cuda_b, &seq);

    let s_cuda = coeus_ops::silu(&a_cuda, &cuda_b);
    let s_seq = s_cuda.to_backend_on(&cuda_b, &seq);

    let mish_cuda = coeus_ops::mish(&a_cuda, &cuda_b);
    let mish_seq = mish_cuda.to_backend_on(&cuda_b, &seq);

    assert_eq!(c_seq.as_slice(), &[11.0, 22.0, 33.0, 44.0]);
    assert_eq!(m_seq.as_slice(), &[70.0, 100.0, 150.0, 220.0]);
    assert!((s_seq.as_slice()[0] - 0.7310586).abs() < 1e-5);
    assert!((s_seq.as_slice()[1] - 1.7615942).abs() < 1e-5);
    assert!((s_seq.as_slice()[2] - 2.8577224).abs() < 1e-5);
    assert!((s_seq.as_slice()[3] - 3.9280551).abs() < 1e-5);
    assert!((mish_seq.as_slice()[0] - 0.8656606).abs() < 1e-3);
    assert!((mish_seq.as_slice()[1] - 1.943866).abs() < 1e-3);
    assert!((mish_seq.as_slice()[2] - 2.986349).abs() < 1e-3);
    assert!((mish_seq.as_slice()[3] - 3.997317).abs() < 1e-3);
}

#[test]
fn test_cuda_norm_p_provider_dispatch() {
    if !crate::availability::device_available() {
        return;
    }

    let backend = CudaBackend::new();
    let sequential = SequentialBackend::new();
    let input =
        Tensor::<f32, SequentialBackend>::from_slice([2, 3], &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
    let device_input = input.to_backend_on(&sequential, &backend);

    let actual = coeus_ops::norm_p(&device_input, 2.0, &backend);
    let expected = 91.0_f32.sqrt();
    assert!((actual - expected).abs() <= f32::EPSILON * 1024.0 * expected);

    let actual_axis = coeus_ops::norm_p_axis(&device_input, 2.0, 1, &backend)
        .to_backend_on(&backend, &sequential);
    assert_eq!(actual_axis.shape(), &[2, 1]);
    for (&actual, expected) in actual_axis
        .as_slice()
        .iter()
        .zip([14.0_f32.sqrt(), 77.0_f32.sqrt()])
    {
        assert!((actual - expected).abs() <= f32::EPSILON * 1024.0 * expected);
    }
}
