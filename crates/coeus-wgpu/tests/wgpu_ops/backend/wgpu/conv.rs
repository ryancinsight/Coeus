use coeus_core::{ComputeBackend, SequentialBackend};
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

#[test]
fn test_wgpu_conv() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    // 1D Convolution
    let input_data = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
    let weight_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 2, 3], &input_data);
    let weight_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 2, 1], &weight_data);

    let input_wgpu = input_seq.to_backend_on(&seq, &wgpu_b);
    let weight_wgpu = weight_seq.to_backend_on(&seq, &wgpu_b);

    let mut out_wgpu_storage = wgpu_b.allocate::<f32>(6);
    let out_layout = coeus_core::Layout::new(vec![1, 2, 3].into());

    coeus_ops::ConvOps::conv1d(
        &wgpu_b,
        input_wgpu.storage(),
        input_wgpu.layout(),
        weight_wgpu.storage(),
        weight_wgpu.layout(),
        None,
        1,
        0,
        1,
        &mut out_wgpu_storage,
        &out_layout,
    )
    .expect("WGPU conv1d dispatch");

    let out_tensor_wgpu: Tensor<f32, WgpuBackend> =
        Tensor::from_raw_parts(out_wgpu_storage, out_layout.clone());
    let out_seq = out_tensor_wgpu.to_backend_on(&wgpu_b, &seq);

    let mut out_expected_storage = seq.allocate::<f32>(6);
    coeus_ops::ConvOps::conv1d(
        &seq,
        input_seq.storage(),
        input_seq.layout(),
        weight_seq.storage(),
        weight_seq.layout(),
        None,
        1,
        0,
        1,
        &mut out_expected_storage,
        &out_layout,
    )
    .expect("CPU conv1d dispatch");
    let out_expected: Tensor<f32, SequentialBackend> =
        Tensor::from_raw_parts(out_expected_storage, out_layout);

    for (i, (&res, &exp)) in out_seq
        .as_slice()
        .iter()
        .zip(out_expected.as_slice().iter())
        .enumerate()
    {
        assert!(
            (res - exp).abs() < 1e-4f32,
            "Conv1D mismatch at {}: {} vs {}",
            i,
            res,
            exp
        );
    }

    // 2D Convolution
    let input_2d_data = vec![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let weight_2d_data = vec![1.0f32, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0];
    let input_2d_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 4, 4], &input_2d_data);
    let weight_2d_seq =
        Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3, 3], &weight_2d_data);

    let input_2d_wgpu = input_2d_seq.to_backend_on(&seq, &wgpu_b);
    let weight_2d_wgpu = weight_2d_seq.to_backend_on(&seq, &wgpu_b);

    let mut out_2d_wgpu_storage = wgpu_b.allocate::<f32>(4);
    let out_2d_layout = coeus_core::Layout::new(vec![1, 1, 2, 2].into());

    coeus_ops::ConvOps::conv2d(
        &wgpu_b,
        input_2d_wgpu.storage(),
        input_2d_wgpu.layout(),
        weight_2d_wgpu.storage(),
        weight_2d_wgpu.layout(),
        None,
        1,
        0,
        1,
        &mut out_2d_wgpu_storage,
        &out_2d_layout,
    )
    .expect("WGPU conv2d dispatch");

    let out_2d_tensor_wgpu: Tensor<f32, WgpuBackend> =
        Tensor::from_raw_parts(out_2d_wgpu_storage, out_2d_layout.clone());
    let out_2d_seq = out_2d_tensor_wgpu.to_backend_on(&wgpu_b, &seq);

    let mut out_2d_expected_storage = seq.allocate::<f32>(4);
    coeus_ops::ConvOps::conv2d(
        &seq,
        input_2d_seq.storage(),
        input_2d_seq.layout(),
        weight_2d_seq.storage(),
        weight_2d_seq.layout(),
        None,
        1,
        0,
        1,
        &mut out_2d_expected_storage,
        &out_2d_layout,
    )
    .expect("CPU conv2d dispatch");
    let out_2d_expected: Tensor<f32, SequentialBackend> =
        Tensor::from_raw_parts(out_2d_expected_storage, out_2d_layout);

    for (i, (&res, &exp)) in out_2d_seq
        .as_slice()
        .iter()
        .zip(out_2d_expected.as_slice().iter())
        .enumerate()
    {
        assert!(
            (res - exp).abs() < 1e-4f32,
            "Conv2D mismatch at {}: {} vs {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_wgpu_conv_backward() {
    use coeus_core::CpuAddressableStorage;
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let grad_out_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_data = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
    let weight_data = vec![1.0f32, 2.0, 3.0, 4.0];

    let grad_out_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 2, 3], &grad_out_data);
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 2, 3], &input_data);
    let weight_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 2, 1], &weight_data);

    let grad_out_wgpu = grad_out_seq.to_backend_on(&seq, &wgpu_b);
    let input_wgpu = input_seq.to_backend_on(&seq, &wgpu_b);
    let weight_wgpu = weight_seq.to_backend_on(&seq, &wgpu_b);

    let mut gi_wgpu = wgpu_b.allocate::<f32>(6);
    wgpu_b.fill(&mut gi_wgpu, 0.0);
    let mut gw_wgpu = wgpu_b.allocate::<f32>(4);
    wgpu_b.fill(&mut gw_wgpu, 0.0);
    let mut gb_wgpu = wgpu_b.allocate::<f32>(2);
    wgpu_b.fill(&mut gb_wgpu, 0.0);

    let gi_layout = coeus_core::Layout::new(vec![1, 2, 3].into());
    let gw_layout = coeus_core::Layout::new(vec![2, 2, 1].into());

    coeus_ops::ConvOps::conv1d_backward(
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
    )
    .expect("WGPU conv1d backward dispatch");

    let mut gi_expected = seq.allocate::<f32>(6);
    seq.fill(&mut gi_expected, 0.0);
    let mut gw_expected = seq.allocate::<f32>(4);
    seq.fill(&mut gw_expected, 0.0);
    let mut gb_expected = seq.allocate::<f32>(2);
    seq.fill(&mut gb_expected, 0.0);

    coeus_ops::ConvOps::conv1d_backward(
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
    )
    .expect("CPU conv1d backward dispatch");

    let gi_wgpu_tensor: Tensor<f32, WgpuBackend> = Tensor::from_raw_parts(gi_wgpu, gi_layout);
    let gi_wgpu_cpu = gi_wgpu_tensor.to_backend_on(&wgpu_b, &seq);
    for (i, (&res, &exp)) in gi_wgpu_cpu
        .as_slice()
        .iter()
        .zip(gi_expected.as_slice().iter())
        .enumerate()
    {
        assert!(
            (res - exp).abs() < 1e-4f32,
            "Conv1D grad_input mismatch at {}: {} vs {}",
            i,
            res,
            exp
        );
    }

    let gw_wgpu_tensor: Tensor<f32, WgpuBackend> = Tensor::from_raw_parts(gw_wgpu, gw_layout);
    let gw_wgpu_cpu = gw_wgpu_tensor.to_backend_on(&wgpu_b, &seq);
    for (i, (&res, &exp)) in gw_wgpu_cpu
        .as_slice()
        .iter()
        .zip(gw_expected.as_slice().iter())
        .enumerate()
    {
        assert!(
            (res - exp).abs() < 1e-4f32,
            "Conv1D grad_weight mismatch at {}: {} vs {}",
            i,
            res,
            exp
        );
    }

    let gb_wgpu_tensor: Tensor<f32, WgpuBackend> =
        Tensor::from_raw_parts(gb_wgpu, coeus_core::Layout::new(vec![2].into()));
    let gb_wgpu_cpu = gb_wgpu_tensor.to_backend_on(&wgpu_b, &seq);
    for (i, (&res, &exp)) in gb_wgpu_cpu
        .as_slice()
        .iter()
        .zip(gb_expected.as_slice().iter())
        .enumerate()
    {
        assert!(
            (res - exp).abs() < 1e-4f32,
            "Conv1D grad_bias mismatch at {}: {} vs {}",
            i,
            res,
            exp
        );
    }
}
