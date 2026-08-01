use coeus_core::SequentialBackend;
use coeus_cuda::CudaBackend;
use coeus_ops::{ConvOps, OptimizerOps, PoolOps};
use coeus_tensor::Tensor;

#[test]
fn test_cuda_adamw_step() {
    if hephaestus_cuda::CudaDevice::try_default().is_err() {
        return;
    }
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
    )
    .expect("CPU AdamW step");

    cuda_b
        .adamw_step(
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
        )
        .expect("CUDA AdamW step");

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
    if hephaestus_cuda::CudaDevice::try_default().is_err() {
        return;
    }
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
    )
    .expect("CPU conv3d dispatch");

    cuda_b
        .conv3d(
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
        )
        .expect("CUDA conv3d dispatch");

    let out_cuda_on_cpu = out_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        assert_eq!(out_cuda_on_cpu.as_slice(), out_seq.as_slice());
    }
}

#[test]
fn test_cuda_max_pool3d_forward_backward() {
    if hephaestus_cuda::CudaDevice::try_default().is_err() {
        return;
    }
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let input_data: Vec<f32> = (1..=27).map(|x| x as f32).collect();
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3, 3, 3], &input_data);
    let input_cuda = input_seq.to_backend_on(&seq, &cuda_b);

    let mut out_seq = Tensor::<f32, SequentialBackend>::zeros(vec![1, 1, 2, 2, 2]);
    let mut out_cuda = Tensor::<f32, CudaBackend>::zeros(vec![1, 1, 2, 2, 2]);

    let input_seq_layout = input_seq.layout().clone();
    let out_seq_layout = out_seq.layout().clone();

    let input_cuda_layout = input_cuda.layout().clone();
    let out_cuda_layout = out_cuda.layout().clone();

    seq.max_pool3d(
        input_seq.storage(),
        &input_seq_layout,
        2,
        1,
        0,
        1,
        out_seq.storage_mut(),
        &out_seq_layout,
    )
    .expect("invariant: validated CPU max_pool3d dispatch must succeed");

    cuda_b
        .max_pool3d(
            input_cuda.storage(),
            &input_cuda_layout,
            2,
            1,
            0,
            1,
            out_cuda.storage_mut(),
            &out_cuda_layout,
        )
        .expect("invariant: validated CUDA max_pool3d dispatch must succeed");

    let out_cuda_on_cpu = out_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        assert_eq!(out_cuda_on_cpu.as_slice(), out_seq.as_slice());
    }

    let grad_out_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let grad_out_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &grad_out_data);
    let grad_out_cuda = grad_out_seq.to_backend_on(&seq, &cuda_b);

    let mut grad_input_seq = Tensor::<f32, SequentialBackend>::zeros(vec![1, 1, 3, 3, 3]);
    let mut grad_input_cuda = Tensor::<f32, CudaBackend>::zeros(vec![1, 1, 3, 3, 3]);

    let grad_out_seq_layout = grad_out_seq.layout().clone();
    let grad_input_seq_layout = grad_input_seq.layout().clone();
    let grad_out_cuda_layout = grad_out_cuda.layout().clone();
    let grad_input_cuda_layout = grad_input_cuda.layout().clone();

    seq.max_pool3d_backward(
        grad_out_seq.storage(),
        &grad_out_seq_layout,
        input_seq.storage(),
        &input_seq_layout,
        2,
        1,
        0,
        1,
        grad_input_seq.storage_mut(),
        &grad_input_seq_layout,
    )
    .expect("invariant: validated CPU max_pool3d backward dispatch must succeed");

    cuda_b
        .max_pool3d_backward(
            grad_out_cuda.storage(),
            &grad_out_cuda_layout,
            input_cuda.storage(),
            &input_cuda_layout,
            2,
            1,
            0,
            1,
            grad_input_cuda.storage_mut(),
            &grad_input_cuda_layout,
        )
        .expect("invariant: validated CUDA max_pool3d backward dispatch must succeed");

    let grad_input_cuda_on_cpu = grad_input_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        assert_eq!(grad_input_cuda_on_cpu.as_slice(), grad_input_seq.as_slice());
    }
}

#[test]
fn test_cuda_avg_pool3d_forward_backward() {
    if hephaestus_cuda::CudaDevice::try_default().is_err() {
        return;
    }
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let input_data: Vec<f32> = (1..=27).map(|x| x as f32).collect();
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3, 3, 3], &input_data);
    let input_cuda = input_seq.to_backend_on(&seq, &cuda_b);

    let mut out_seq = Tensor::<f32, SequentialBackend>::zeros(vec![1, 1, 2, 2, 2]);
    let mut out_cuda = Tensor::<f32, CudaBackend>::zeros(vec![1, 1, 2, 2, 2]);

    let input_seq_layout = input_seq.layout().clone();
    let out_seq_layout = out_seq.layout().clone();

    let input_cuda_layout = input_cuda.layout().clone();
    let out_cuda_layout = out_cuda.layout().clone();

    seq.avg_pool3d(
        input_seq.storage(),
        &input_seq_layout,
        2,
        1,
        0,
        1,
        out_seq.storage_mut(),
        &out_seq_layout,
    )
    .expect("invariant: validated CPU avg_pool3d dispatch must succeed");

    cuda_b
        .avg_pool3d(
            input_cuda.storage(),
            &input_cuda_layout,
            2,
            1,
            0,
            1,
            out_cuda.storage_mut(),
            &out_cuda_layout,
        )
        .expect("invariant: validated CUDA avg_pool3d dispatch must succeed");

    let out_cuda_on_cpu = out_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        assert_close(out_cuda_on_cpu.as_slice(), out_seq.as_slice(), "avg_pool3d");
    }

    let grad_out_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let grad_out_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &grad_out_data);
    let grad_out_cuda = grad_out_seq.to_backend_on(&seq, &cuda_b);

    let mut grad_input_seq = Tensor::<f32, SequentialBackend>::zeros(vec![1, 1, 3, 3, 3]);
    let mut grad_input_cuda = Tensor::<f32, CudaBackend>::zeros(vec![1, 1, 3, 3, 3]);

    let grad_out_seq_layout = grad_out_seq.layout().clone();
    let grad_input_seq_layout = grad_input_seq.layout().clone();
    let grad_out_cuda_layout = grad_out_cuda.layout().clone();
    let grad_input_cuda_layout = grad_input_cuda.layout().clone();

    seq.avg_pool3d_backward(
        grad_out_seq.storage(),
        &grad_out_seq_layout,
        2,
        1,
        0,
        1,
        grad_input_seq.storage_mut(),
        &grad_input_seq_layout,
    )
    .expect("invariant: validated CPU avg_pool3d backward dispatch must succeed");

    cuda_b
        .avg_pool3d_backward(
            grad_out_cuda.storage(),
            &grad_out_cuda_layout,
            2,
            1,
            0,
            1,
            grad_input_cuda.storage_mut(),
            &grad_input_cuda_layout,
        )
        .expect("invariant: validated CUDA avg_pool3d backward dispatch must succeed");

    let grad_input_cuda_on_cpu = grad_input_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        assert_close(
            grad_input_cuda_on_cpu.as_slice(),
            grad_input_seq.as_slice(),
            "avg_pool3d_backward",
        );
    }
}

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    for (index, (&actual, &expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "{label} mismatch at {index}: actual={actual} expected={expected}"
        );
    }
}
