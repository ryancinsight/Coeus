use coeus_core::SequentialBackend;
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

#[test]
fn test_wgpu_adamw() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let shape = vec![2, 3];
    let p_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let g_data = vec![0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6];
    let m_data = vec![0.01f32, 0.02, 0.03, 0.04, 0.05, 0.06];
    let v_data = vec![0.11f32, 0.12, 0.13, 0.14, 0.15, 0.16];

    let mut p_seq = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &p_data);
    let g_seq = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &g_data);
    let mut m_seq = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &m_data);
    let mut v_seq = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &v_data);

    let mut p_wgpu = p_seq.to_backend_on(&seq, &wgpu_b);
    let g_wgpu = g_seq.to_backend_on(&seq, &wgpu_b);
    let mut m_wgpu = m_seq.to_backend_on(&seq, &wgpu_b);
    let mut v_wgpu = v_seq.to_backend_on(&seq, &wgpu_b);

    let lr = 0.05f32;
    let beta1 = 0.9f32;
    let beta2 = 0.99f32;
    let eps = 1e-6f32;
    let weight_decay = 0.02f32;
    let t = 3usize;

    {
        let (p_storage, p_layout) = p_seq.storage_mut_and_layout();
        let (m_storage, m_layout) = m_seq.storage_mut_and_layout();
        let (v_storage, v_layout) = v_seq.storage_mut_and_layout();
        seq.adamw_step(
            p_storage,
            p_layout,
            g_seq.storage(),
            g_seq.layout(),
            m_storage,
            m_layout,
            v_storage,
            v_layout,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t,
        );
    }

    {
        let (p_storage, p_layout) = p_wgpu.storage_mut_and_layout();
        let (m_storage, m_layout) = m_wgpu.storage_mut_and_layout();
        let (v_storage, v_layout) = v_wgpu.storage_mut_and_layout();
        wgpu_b.adamw_step(
            p_storage,
            p_layout,
            g_wgpu.storage(),
            g_wgpu.layout(),
            m_storage,
            m_layout,
            v_storage,
            v_layout,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t,
        );
    }

    let p_res = p_wgpu.to_backend_on(&wgpu_b, &seq);
    let m_res = m_wgpu.to_backend_on(&wgpu_b, &seq);
    let v_res = v_wgpu.to_backend_on(&wgpu_b, &seq);

    for i in 0..p_seq.as_slice().len() {
        assert!(
            (p_res.as_slice()[i] - p_seq.as_slice()[i]).abs() < 1e-5,
            "p mismatch at {}: GPU={:?}, CPU={:?}",
            i,
            p_res.as_slice()[i],
            p_seq.as_slice()[i]
        );
        assert!(
            (m_res.as_slice()[i] - m_seq.as_slice()[i]).abs() < 1e-5,
            "m mismatch at {}: GPU={:?}, CPU={:?}",
            i,
            m_res.as_slice()[i],
            m_seq.as_slice()[i]
        );
        assert!(
            (v_res.as_slice()[i] - v_seq.as_slice()[i]).abs() < 1e-5,
            "v mismatch at {}: GPU={:?}, CPU={:?}",
            i,
            v_res.as_slice()[i],
            v_seq.as_slice()[i]
        );
    }
}
