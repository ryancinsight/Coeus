use coeus_core::{ComputeBackend, CpuAddressableStorage, SequentialBackend};
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

#[test]
fn test_wgpu_conv3d() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let input_data: Vec<f32> = (1..=27).map(|x| x as f32).collect();
    let weight_data: Vec<f32> = vec![1.0, 0.0, -1.0, 1.0, 2.0, -2.0, 0.5, -0.5];
    let bias_data: Vec<f32> = vec![0.5];

    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3, 3, 3], &input_data);
    let weight_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &weight_data);
    let bias_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1], &bias_data);

    let input_wgpu = input_seq.to_backend_on(&seq, &wgpu_b);
    let weight_wgpu = weight_seq.to_backend_on(&seq, &wgpu_b);
    let bias_wgpu = bias_seq.to_backend_on(&seq, &wgpu_b);

    let out_layout = coeus_core::Layout::new(vec![1, 1, 2, 2, 2].into());
    let mut out_wgpu_storage = wgpu_b.allocate::<f32>(8);

    coeus_ops::BackendOps::conv3d(
        &wgpu_b,
        input_wgpu.storage(),
        input_wgpu.layout(),
        weight_wgpu.storage(),
        weight_wgpu.layout(),
        Some(bias_wgpu.storage()),
        1,
        0,
        1,
        &mut out_wgpu_storage,
        &out_layout,
    );

    let out_tensor_wgpu =
        Tensor::<f32, WgpuBackend>::from_raw_parts(out_wgpu_storage, out_layout.clone());
    let out_wgpu_cpu = out_tensor_wgpu.to_backend_on(&wgpu_b, &seq);

    let mut out_expected_storage = seq.allocate::<f32>(8);
    coeus_ops::BackendOps::conv3d(
        &seq,
        input_seq.storage(),
        input_seq.layout(),
        weight_seq.storage(),
        weight_seq.layout(),
        Some(bias_seq.storage()),
        1,
        0,
        1,
        &mut out_expected_storage,
        &out_layout,
    );
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
            "Conv3D mismatch at {}: {} vs {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_wgpu_conv3d_backward() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let grad_out_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let input_data: Vec<f32> = (1..=27).map(|x| x as f32).collect();
    let weight_data: Vec<f32> = vec![1.0, 0.0, -1.0, 1.0, 2.0, -2.0, 0.5, -0.5];

    let grad_out_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &grad_out_data);
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3, 3, 3], &input_data);
    let weight_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &weight_data);

    let grad_out_wgpu = grad_out_seq.to_backend_on(&seq, &wgpu_b);
    let input_wgpu = input_seq.to_backend_on(&seq, &wgpu_b);
    let weight_wgpu = weight_seq.to_backend_on(&seq, &wgpu_b);

    let mut gi_wgpu = wgpu_b.allocate::<f32>(27);
    wgpu_b.fill(&mut gi_wgpu, 0.0);
    let mut gw_wgpu = wgpu_b.allocate::<f32>(8);
    wgpu_b.fill(&mut gw_wgpu, 0.0);
    let mut gb_wgpu = wgpu_b.allocate::<f32>(1);
    wgpu_b.fill(&mut gb_wgpu, 0.0);

    let gi_layout = coeus_core::Layout::new(vec![1, 1, 3, 3, 3].into());
    let gw_layout = coeus_core::Layout::new(vec![1, 1, 2, 2, 2].into());

    coeus_ops::BackendOps::conv3d_backward(
        &wgpu_b,
        grad_out_wgpu.storage(),
        grad_out_wgpu.layout(),
        input_wgpu.storage(),
        input_wgpu.layout(),
        weight_wgpu.storage(),
        weight_wgpu.layout(),
        Some(&mut gi_wgpu),
        &gi_layout,
        Some(&mut gw_wgpu),
        &gw_layout,
        Some(&mut gb_wgpu),
        1,
        0,
        1,
    );

    let mut gi_expected = seq.allocate::<f32>(27);
    seq.fill(&mut gi_expected, 0.0);
    let mut gw_expected = seq.allocate::<f32>(8);
    seq.fill(&mut gw_expected, 0.0);
    let mut gb_expected = seq.allocate::<f32>(1);
    seq.fill(&mut gb_expected, 0.0);

    coeus_ops::BackendOps::conv3d_backward(
        &seq,
        grad_out_seq.storage(),
        grad_out_seq.layout(),
        input_seq.storage(),
        input_seq.layout(),
        weight_seq.storage(),
        weight_seq.layout(),
        Some(&mut gi_expected),
        &gi_layout,
        Some(&mut gw_expected),
        &gw_layout,
        Some(&mut gb_expected),
        1,
        0,
        1,
    );

    let gi_wgpu_tensor = Tensor::<f32, WgpuBackend>::from_raw_parts(gi_wgpu, gi_layout);
    let gi_wgpu_cpu = gi_wgpu_tensor.to_backend_on(&wgpu_b, &seq);
    for (i, (&res, &exp)) in gi_wgpu_cpu
        .as_slice()
        .iter()
        .zip(gi_expected.as_slice().iter())
        .enumerate()
    {
        assert!(
            (res - exp).abs() < 1e-4f32,
            "Conv3D grad_input mismatch at {}: {} vs {}",
            i,
            res,
            exp
        );
    }

    let gw_wgpu_tensor = Tensor::<f32, WgpuBackend>::from_raw_parts(gw_wgpu, gw_layout);
    let gw_wgpu_cpu = gw_wgpu_tensor.to_backend_on(&wgpu_b, &seq);
    for (i, (&res, &exp)) in gw_wgpu_cpu
        .as_slice()
        .iter()
        .zip(gw_expected.as_slice().iter())
        .enumerate()
    {
        assert!(
            (res - exp).abs() < 1e-4f32,
            "Conv3D grad_weight mismatch at {}: {} vs {}",
            i,
            res,
            exp
        );
    }

    let gb_wgpu_tensor = Tensor::<f32, WgpuBackend>::from_raw_parts(
        gb_wgpu,
        coeus_core::Layout::new(vec![1].into()),
    );
    let gb_wgpu_cpu = gb_wgpu_tensor.to_backend_on(&wgpu_b, &seq);
    for (i, (&res, &exp)) in gb_wgpu_cpu
        .as_slice()
        .iter()
        .zip(gb_expected.as_slice().iter())
        .enumerate()
    {
        assert!(
            (res - exp).abs() < 1e-4f32,
            "Conv3D grad_bias mismatch at {}: {} vs {}",
            i,
            res,
            exp
        );
    }
}
