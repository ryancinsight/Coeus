use coeus_core::{ComputeBackend, SequentialBackend};
use coeus_cuda::CudaBackend;
use coeus_tensor::Tensor;

#[test]
fn test_cuda_adamw_step() {
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let param_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let grad_data = vec![0.1f32, 0.2, 0.3, 0.4];
    let m_data = vec![0.01f32, 0.02, 0.03, 0.04];
    let v_data = vec![0.001f32, 0.002, 0.003, 0.004];

    let mut param_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![4], &param_data);
    let grad_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![4], &grad_data);
    let mut m_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![4], &m_data);
    let mut v_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![4], &v_data);

    let mut param_cuda = param_seq.to_backend_on(&seq, &cuda_b);
    let grad_cuda = grad_seq.to_backend_on(&seq, &cuda_b);
    let mut m_cuda = m_seq.to_backend_on(&seq, &cuda_b);
    let mut v_cuda = v_seq.to_backend_on(&seq, &cuda_b);

    let p_seq_layout = param_seq.layout().clone();
    let g_seq_layout = grad_seq.layout().clone();
    let m_seq_layout = m_seq.layout().clone();
    let v_seq_layout = v_seq.layout().clone();

    let p_cuda_layout = param_cuda.layout().clone();
    let g_cuda_layout = grad_cuda.layout().clone();
    let m_cuda_layout = m_cuda.layout().clone();
    let v_cuda_layout = v_cuda.layout().clone();

    use coeus_ops::BackendOps;
    seq.adamw_step(
        param_seq.storage_mut(),
        &p_seq_layout,
        grad_seq.storage(),
        &g_seq_layout,
        m_seq.storage_mut(),
        &m_seq_layout,
        v_seq.storage_mut(),
        &v_seq_layout,
        0.001,
        0.9,
        0.999,
        1e-8,
        0.01,
        1,
    );

    cuda_b.adamw_step(
        param_cuda.storage_mut(),
        &p_cuda_layout,
        grad_cuda.storage(),
        &g_cuda_layout,
        m_cuda.storage_mut(),
        &m_cuda_layout,
        v_cuda.storage_mut(),
        &v_cuda_layout,
        0.001,
        0.9,
        0.999,
        1e-8,
        0.01,
        1,
    );

    let param_cuda_on_cpu = param_cuda.to_backend_on(&cuda_b, &seq);
    let m_cuda_on_cpu = m_cuda.to_backend_on(&cuda_b, &seq);
    let v_cuda_on_cpu = v_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        for (i, (&res, &exp)) in param_cuda_on_cpu
            .as_slice()
            .iter()
            .zip(param_seq.as_slice().iter())
            .enumerate()
        {
            assert!(
                (res - exp).abs() < 1e-5,
                "Param mismatch at {}: {} vs expected {}",
                i,
                res,
                exp
            );
        }
        for (i, (&res, &exp)) in m_cuda_on_cpu
            .as_slice()
            .iter()
            .zip(m_seq.as_slice().iter())
            .enumerate()
        {
            assert!(
                (res - exp).abs() < 1e-5,
                "M mismatch at {}: {} vs expected {}",
                i,
                res,
                exp
            );
        }
        for (i, (&res, &exp)) in v_cuda_on_cpu
            .as_slice()
            .iter()
            .zip(v_seq.as_slice().iter())
            .enumerate()
        {
            assert!(
                (res - exp).abs() < 1e-5,
                "V mismatch at {}: {} vs expected {}",
                i,
                res,
                exp
            );
        }
    }
}

#[test]
fn test_cuda_conv3d() {
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let weight_data = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &input_data);
    let weight_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &weight_data);

    let input_cuda = input_seq.to_backend_on(&seq, &cuda_b);
    let weight_cuda = weight_seq.to_backend_on(&seq, &cuda_b);

    let mut out_seq = Tensor::<f32, SequentialBackend>::zeros(vec![1, 1, 1, 1, 1]);
    let mut out_cuda = Tensor::<f32, CudaBackend>::zeros(vec![1, 1, 1, 1, 1]);

    let input_seq_layout = input_seq.layout().clone();
    let weight_seq_layout = weight_seq.layout().clone();
    let out_seq_layout = out_seq.layout().clone();

    let input_cuda_layout = input_cuda.layout().clone();
    let weight_cuda_layout = weight_cuda.layout().clone();
    let out_cuda_layout = out_cuda.layout().clone();

    use coeus_ops::BackendOps;
    seq.conv3d(
        input_seq.storage(),
        &input_seq_layout,
        weight_seq.storage(),
        &weight_seq_layout,
        None,
        1,
        0,
        1,
        out_seq.storage_mut(),
        &out_seq_layout,
    );

    cuda_b.conv3d(
        input_cuda.storage(),
        &input_cuda_layout,
        weight_cuda.storage(),
        &weight_cuda_layout,
        None,
        1,
        0,
        1,
        out_cuda.storage_mut(),
        &out_cuda_layout,
    );

    let out_cuda_on_cpu = out_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        assert_eq!(out_cuda_on_cpu.as_slice(), out_seq.as_slice());
    }
}

#[test]
fn test_cuda_pool3d() {
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &input_data);
    let input_cuda = input_seq.to_backend_on(&seq, &cuda_b);

    let mut out_seq = Tensor::<f32, SequentialBackend>::zeros(vec![1, 1, 1, 1, 1]);
    let mut out_cuda = Tensor::<f32, CudaBackend>::zeros(vec![1, 1, 1, 1, 1]);

    let input_seq_layout = input_seq.layout().clone();
    let out_seq_layout = out_seq.layout().clone();

    let input_cuda_layout = input_cuda.layout().clone();
    let out_cuda_layout = out_cuda.layout().clone();

    use coeus_ops::BackendOps;
    seq.max_pool3d(
        input_seq.storage(),
        &input_seq_layout,
        2,
        2,
        0,
        1,
        out_seq.storage_mut(),
        &out_seq_layout,
    );

    cuda_b.max_pool3d(
        input_cuda.storage(),
        &input_cuda_layout,
        2,
        2,
        0,
        1,
        out_cuda.storage_mut(),
        &out_cuda_layout,
    );

    let out_cuda_on_cpu = out_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        assert_eq!(out_cuda_on_cpu.as_slice(), out_seq.as_slice());
    }
}
