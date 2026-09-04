use coeus_core::{ComputeBackend, SequentialBackend};
use coeus_cuda::CudaBackend;
use coeus_tensor::Tensor;

#[test]
fn test_cuda_backend_conv_and_reduce() {
    if hephaestus_cuda::CudaDevice::try_default().is_err() {
        return;
    }
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 4], &a_data);
    let a_cuda = a_seq.to_backend_on(&seq, &cuda_b);

    let r_cuda = coeus_ops::sum_axis(&a_cuda, 1, &cuda_b).expect("valid CUDA sum axis");
    let r_seq = r_cuda.to_backend_on(&cuda_b, &seq);

    assert_eq!(r_seq.as_slice(), &[10.0, 26.0]);

    let input_data = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
    let weight_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 2, 3], &input_data);
    let weight_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 2, 1], &weight_data);

    let input_cuda = input_seq.to_backend_on(&seq, &cuda_b);
    let weight_cuda = weight_seq.to_backend_on(&seq, &cuda_b);

    let mut out_cuda_storage = cuda_b.allocate::<f32>(6);
    let out_layout = coeus_core::Layout::new(vec![1, 2, 3].into());

    coeus_ops::ConvOps::conv1d(
        &cuda_b,
        input_cuda.storage(),
        input_cuda.layout(),
        weight_cuda.storage(),
        weight_cuda.layout(),
        None,
        1,
        0,
        1,
        &mut out_cuda_storage,
        &out_layout,
    )
    .expect("CUDA conv1d dispatch");

    let out_tensor_cuda: Tensor<f32, CudaBackend> =
        Tensor::from_raw_parts(out_cuda_storage, out_layout.clone());
    let out_seq = out_tensor_cuda.to_backend_on(&cuda_b, &seq);

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

    let out_seq_slice: &[f32] = out_seq.as_slice();
    let out_expected_slice: &[f32] = out_expected.as_slice();
    for (i, (&res, &exp)) in out_seq_slice
        .iter()
        .zip(out_expected_slice.iter())
        .enumerate()
    {
        assert!(
            (res - exp).abs() < 1e-4f32,
            "Mismatch at {}: {} vs {}",
            i,
            res,
            exp
        );
    }
}

#[test]
fn test_cuda_conv_backward() {
    if hephaestus_cuda::CudaDevice::try_default().is_err() {
        return;
    }
    use coeus_core::CpuAddressableStorage;
    let seq = SequentialBackend::new();
    let cuda_b = CudaBackend::new();

    let grad_out_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_data = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0];
    let weight_data = vec![1.0f32, 2.0, 3.0, 4.0];

    let grad_out_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 2, 3], &grad_out_data);
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 2, 3], &input_data);
    let weight_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 2, 1], &weight_data);

    let grad_out_cuda = grad_out_seq.to_backend_on(&seq, &cuda_b);
    let input_cuda = input_seq.to_backend_on(&seq, &cuda_b);
    let weight_cuda = weight_seq.to_backend_on(&seq, &cuda_b);

    let mut gi_cuda = cuda_b.allocate::<f32>(6);
    cuda_b.fill(&mut gi_cuda, 0.0);
    let mut gw_cuda = cuda_b.allocate::<f32>(4);
    cuda_b.fill(&mut gw_cuda, 0.0);
    let mut gb_cuda = cuda_b.allocate::<f32>(2);
    cuda_b.fill(&mut gb_cuda, 0.0);

    let gi_layout = coeus_core::Layout::new(vec![1, 2, 3].into());
    let gw_layout = coeus_core::Layout::new(vec![2, 2, 1].into());

    coeus_ops::ConvOps::conv1d_backward(
        &cuda_b,
        grad_out_cuda.storage(),
        grad_out_cuda.layout(),
        input_cuda.storage(),
        input_cuda.layout(),
        weight_cuda.storage(),
        weight_cuda.layout(),
        Some(&mut gi_cuda),
        &gi_layout,
        Some(&mut gw_cuda),
        &gw_layout,
        Some(&mut gb_cuda),
        1,
        0,
        1,
    )
    .expect("CUDA conv1d backward dispatch");

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

    let gi_cuda_tensor: Tensor<f32, CudaBackend> = Tensor::from_raw_parts(gi_cuda, gi_layout);
    let gi_cuda_cpu = gi_cuda_tensor.to_backend_on(&cuda_b, &seq);
    let gi_expected_slice = gi_expected.as_slice();
    for (i, (&res, &exp)) in gi_cuda_cpu
        .as_slice()
        .iter()
        .zip(gi_expected_slice.iter())
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

    let gw_cuda_tensor: Tensor<f32, CudaBackend> = Tensor::from_raw_parts(gw_cuda, gw_layout);
    let gw_cuda_cpu = gw_cuda_tensor.to_backend_on(&cuda_b, &seq);
    let gw_expected_slice = gw_expected.as_slice();
    for (i, (&res, &exp)) in gw_cuda_cpu
        .as_slice()
        .iter()
        .zip(gw_expected_slice.iter())
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

    let gb_cuda_tensor: Tensor<f32, CudaBackend> =
        Tensor::from_raw_parts(gb_cuda, coeus_core::Layout::new(vec![2].into()));
    let gb_cuda_cpu = gb_cuda_tensor.to_backend_on(&cuda_b, &seq);
    let gb_expected_slice = gb_expected.as_slice();
    for (i, (&res, &exp)) in gb_cuda_cpu
        .as_slice()
        .iter()
        .zip(gb_expected_slice.iter())
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
