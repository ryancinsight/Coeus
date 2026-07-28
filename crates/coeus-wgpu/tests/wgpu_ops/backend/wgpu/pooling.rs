use coeus_core::{ComputeBackend, SequentialBackend};
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

#[test]
fn test_wgpu_max_pool2d() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 4, 4], &input_data);
    let input_wgpu = input_seq.to_backend_on(&seq, &wgpu_b);

    let mut out_wgpu_storage = wgpu_b.allocate::<f32>(4);
    let out_layout = coeus_core::Layout::new(vec![1, 1, 2, 2].into());

    coeus_ops::PoolOps::max_pool2d(
        &wgpu_b,
        input_wgpu.storage(),
        input_wgpu.layout(),
        2,
        2,
        0,
        1,
        &mut out_wgpu_storage,
        &out_layout,
    )
    .expect("WGPU max_pool2d dispatch");

    let out_wgpu_tensor: Tensor<f32, WgpuBackend> =
        Tensor::from_raw_parts(out_wgpu_storage, out_layout.clone());
    let out_wgpu_cpu = out_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    assert_eq!(out_wgpu_cpu.as_slice(), &[6.0, 8.0, 14.0, 16.0]);

    // Backward
    let grad_out_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let grad_out_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2], &grad_out_data);
    let grad_out_wgpu = grad_out_seq.to_backend_on(&seq, &wgpu_b);

    let mut grad_input_wgpu_storage = wgpu_b.allocate::<f32>(16);
    wgpu_b.fill(&mut grad_input_wgpu_storage, 0.0);
    let gi_layout = coeus_core::Layout::new(vec![1, 1, 4, 4].into());

    coeus_ops::PoolOps::max_pool2d_backward(
        &wgpu_b,
        grad_out_wgpu.storage(),
        grad_out_wgpu.layout(),
        input_wgpu.storage(),
        input_wgpu.layout(),
        2,
        2,
        0,
        1,
        &mut grad_input_wgpu_storage,
        &gi_layout,
    )
    .expect("WGPU max_pool2d backward dispatch");

    let gi_wgpu_tensor: Tensor<f32, WgpuBackend> =
        Tensor::from_raw_parts(grad_input_wgpu_storage, gi_layout);
    let gi_wgpu_cpu = gi_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    let mut expected_gi = vec![0.0f32; 16];
    expected_gi[5] = 1.0;
    expected_gi[7] = 2.0;
    expected_gi[13] = 3.0;
    expected_gi[15] = 4.0;

    assert_eq!(gi_wgpu_cpu.as_slice(), &expected_gi);
}

#[test]
fn test_wgpu_avg_pool2d() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 4, 4], &input_data);
    let input_wgpu = input_seq.to_backend_on(&seq, &wgpu_b);

    let mut out_wgpu_storage = wgpu_b.allocate::<f32>(4);
    let out_layout = coeus_core::Layout::new(vec![1, 1, 2, 2].into());

    coeus_ops::PoolOps::avg_pool2d(
        &wgpu_b,
        input_wgpu.storage(),
        input_wgpu.layout(),
        2,
        2,
        0,
        1,
        &mut out_wgpu_storage,
        &out_layout,
    )
    .expect("WGPU avg_pool2d dispatch");

    let out_wgpu_tensor: Tensor<f32, WgpuBackend> =
        Tensor::from_raw_parts(out_wgpu_storage, out_layout.clone());
    let out_wgpu_cpu = out_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    assert_eq!(out_wgpu_cpu.as_slice(), &[3.5, 5.5, 11.5, 13.5]);

    // Backward
    let grad_out_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let grad_out_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2], &grad_out_data);
    let grad_out_wgpu = grad_out_seq.to_backend_on(&seq, &wgpu_b);

    let mut grad_input_wgpu_storage = wgpu_b.allocate::<f32>(16);
    wgpu_b.fill(&mut grad_input_wgpu_storage, 0.0);
    let gi_layout = coeus_core::Layout::new(vec![1, 1, 4, 4].into());

    coeus_ops::PoolOps::avg_pool2d_backward(
        &wgpu_b,
        grad_out_wgpu.storage(),
        grad_out_wgpu.layout(),
        2,
        2,
        0,
        1,
        &mut grad_input_wgpu_storage,
        &gi_layout,
    )
    .expect("WGPU avg_pool2d backward dispatch");

    let gi_wgpu_tensor: Tensor<f32, WgpuBackend> =
        Tensor::from_raw_parts(grad_input_wgpu_storage, gi_layout);
    let gi_wgpu_cpu = gi_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    let expected_gi = vec![
        0.25f32, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 0.75, 0.75, 1.0, 1.0,
    ];
    assert_eq!(gi_wgpu_cpu.as_slice(), &expected_gi);
}

#[test]
fn test_wgpu_max_pool3d() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let input_data: Vec<f32> = (1..=27).map(|x| x as f32).collect();
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3, 3, 3], &input_data);
    let input_wgpu = input_seq.to_backend_on(&seq, &wgpu_b);

    let mut out_wgpu_storage = wgpu_b.allocate::<f32>(8);
    let out_layout = coeus_core::Layout::new(vec![1, 1, 2, 2, 2].into());

    coeus_ops::PoolOps::max_pool3d(
        &wgpu_b,
        input_wgpu.storage(),
        input_wgpu.layout(),
        2,
        1,
        0,
        1,
        &mut out_wgpu_storage,
        &out_layout,
    )
    .expect("WGPU max_pool3d dispatch");

    let out_wgpu_tensor =
        Tensor::<f32, WgpuBackend>::from_raw_parts(out_wgpu_storage, out_layout.clone());
    let out_wgpu_cpu = out_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    let mut out_expected_storage = seq.allocate::<f32>(8);
    coeus_ops::PoolOps::max_pool3d(
        &seq,
        input_seq.storage(),
        input_seq.layout(),
        2,
        1,
        0,
        1,
        &mut out_expected_storage,
        &out_layout,
    )
    .expect("sequential max_pool3d dispatch");
    let out_expected =
        Tensor::<f32, SequentialBackend>::from_raw_parts(out_expected_storage, out_layout);

    assert_eq!(out_wgpu_cpu.as_slice(), out_expected.as_slice());

    // Backward
    let grad_out_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let grad_out_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &grad_out_data);
    let grad_out_wgpu = grad_out_seq.to_backend_on(&seq, &wgpu_b);

    let mut grad_input_wgpu_storage = wgpu_b.allocate::<f32>(27);
    wgpu_b.fill(&mut grad_input_wgpu_storage, 0.0);
    let gi_layout = coeus_core::Layout::new(vec![1, 1, 3, 3, 3].into());

    coeus_ops::PoolOps::max_pool3d_backward(
        &wgpu_b,
        grad_out_wgpu.storage(),
        grad_out_wgpu.layout(),
        input_wgpu.storage(),
        input_wgpu.layout(),
        2,
        1,
        0,
        1,
        &mut grad_input_wgpu_storage,
        &gi_layout,
    )
    .expect("WGPU max_pool3d backward dispatch");

    let gi_wgpu_tensor =
        Tensor::<f32, WgpuBackend>::from_raw_parts(grad_input_wgpu_storage, gi_layout.clone());
    let gi_wgpu_cpu = gi_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    let mut grad_input_expected_storage = seq.allocate::<f32>(27);
    seq.fill(&mut grad_input_expected_storage, 0.0);
    coeus_ops::PoolOps::max_pool3d_backward(
        &seq,
        grad_out_seq.storage(),
        grad_out_seq.layout(),
        input_seq.storage(),
        input_seq.layout(),
        2,
        1,
        0,
        1,
        &mut grad_input_expected_storage,
        &gi_layout,
    )
    .expect("sequential max_pool3d backward dispatch");
    let gi_expected =
        Tensor::<f32, SequentialBackend>::from_raw_parts(grad_input_expected_storage, gi_layout);

    assert_eq!(gi_wgpu_cpu.as_slice(), gi_expected.as_slice());
}

#[test]
fn test_wgpu_avg_pool3d() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let input_data: Vec<f32> = (1..=27).map(|x| x as f32).collect();
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3, 3, 3], &input_data);
    let input_wgpu = input_seq.to_backend_on(&seq, &wgpu_b);

    let mut out_wgpu_storage = wgpu_b.allocate::<f32>(8);
    let out_layout = coeus_core::Layout::new(vec![1, 1, 2, 2, 2].into());

    coeus_ops::PoolOps::avg_pool3d(
        &wgpu_b,
        input_wgpu.storage(),
        input_wgpu.layout(),
        2,
        1,
        0,
        1,
        &mut out_wgpu_storage,
        &out_layout,
    )
    .expect("WGPU avg_pool3d dispatch");

    let out_wgpu_tensor =
        Tensor::<f32, WgpuBackend>::from_raw_parts(out_wgpu_storage, out_layout.clone());
    let out_wgpu_cpu = out_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    let mut out_expected_storage = seq.allocate::<f32>(8);
    coeus_ops::PoolOps::avg_pool3d(
        &seq,
        input_seq.storage(),
        input_seq.layout(),
        2,
        1,
        0,
        1,
        &mut out_expected_storage,
        &out_layout,
    )
    .expect("sequential avg_pool3d dispatch");
    let out_expected =
        Tensor::<f32, SequentialBackend>::from_raw_parts(out_expected_storage, out_layout);

    for (i, (&res, &exp)) in out_wgpu_cpu
        .as_slice()
        .iter()
        .zip(out_expected.as_slice().iter())
        .enumerate()
    {
        assert!(
            (res - exp).abs() < 1e-4f32,
            "AvgPool3D mismatch at {}: {} vs {}",
            i,
            res,
            exp
        );
    }

    // Backward
    let grad_out_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let grad_out_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &grad_out_data);
    let grad_out_wgpu = grad_out_seq.to_backend_on(&seq, &wgpu_b);

    let mut grad_input_wgpu_storage = wgpu_b.allocate::<f32>(27);
    wgpu_b.fill(&mut grad_input_wgpu_storage, 0.0);
    let gi_layout = coeus_core::Layout::new(vec![1, 1, 3, 3, 3].into());

    coeus_ops::PoolOps::avg_pool3d_backward(
        &wgpu_b,
        grad_out_wgpu.storage(),
        grad_out_wgpu.layout(),
        2,
        1,
        0,
        1,
        &mut grad_input_wgpu_storage,
        &gi_layout,
    )
    .expect("WGPU avg_pool3d backward dispatch");

    let gi_wgpu_tensor =
        Tensor::<f32, WgpuBackend>::from_raw_parts(grad_input_wgpu_storage, gi_layout.clone());
    let gi_wgpu_cpu = gi_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    let mut grad_input_expected_storage = seq.allocate::<f32>(27);
    seq.fill(&mut grad_input_expected_storage, 0.0);
    coeus_ops::PoolOps::avg_pool3d_backward(
        &seq,
        grad_out_seq.storage(),
        grad_out_seq.layout(),
        2,
        1,
        0,
        1,
        &mut grad_input_expected_storage,
        &gi_layout,
    )
    .expect("sequential avg_pool3d backward dispatch");
    let gi_expected =
        Tensor::<f32, SequentialBackend>::from_raw_parts(grad_input_expected_storage, gi_layout);

    for (i, (&res, &exp)) in gi_wgpu_cpu
        .as_slice()
        .iter()
        .zip(gi_expected.as_slice().iter())
        .enumerate()
    {
        assert!(
            (res - exp).abs() < 1e-4f32,
            "AvgPool3D backward mismatch at {}: {} vs {}",
            i,
            res,
            exp
        );
    }
}
