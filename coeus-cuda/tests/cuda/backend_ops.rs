use coeus_core::{ComputeBackend, SequentialBackend};
use coeus_cuda::CudaBackend;
use coeus_tensor::Tensor;

#[test]
fn test_cuda_backend_compilation_and_fallback() {
    if hephaestus_cuda::CudaDevice::try_default().is_err() {
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

#[allow(clippy::excessive_precision)]
#[test]
fn test_cuda_backend_ops() {
    if hephaestus_cuda::CudaDevice::try_default().is_err() {
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

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
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
    } else {
        assert_eq!(c_seq.shape(), &[2, 2]);
        assert_eq!(m_seq.shape(), &[2, 2]);
        assert_eq!(s_seq.shape(), &[2, 2]);
        assert_eq!(mish_seq.shape(), &[2, 2]);
    }
}
