use coeus_core::{SequentialBackend, ComputeBackend, CpuAddressableStorage};
use coeus_tensor::Tensor;
use coeus_wgpu::{WgpuBackend, add};

#[test]
fn test_wgpu_transfers_and_addition() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    // 1. Create source tensors on host CPU
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    
    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &a_data);
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &b_data);

    // 2. Transfer to GPU
    let a_wgpu = a_seq.to_backend_on(&seq, &wgpu_b);
    let b_wgpu = b_seq.to_backend_on(&seq, &wgpu_b);

    // Verify shapes on GPU
    assert_eq!(a_wgpu.shape(), &[2, 3]);
    assert_eq!(b_wgpu.shape(), &[2, 3]);

    // 3. Execute element-wise addition on GPU
    let c_wgpu = add(&a_wgpu, &b_wgpu);

    // 4. Transfer result back to CPU
    let c_seq = c_wgpu.to_backend_on(&wgpu_b, &seq);

    // 5. Validate result correctness
    let expected = vec![11.0f32, 22.0, 33.0, 44.0, 55.0, 66.0];
    assert_eq!(c_seq.as_slice(), &expected);
}

#[test]
fn test_wgpu_backend_ops_unified() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    // Test addition and ReLU
    let a = Tensor::<f32, WgpuBackend>::from_slice_on(vec![2, 3], &[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &wgpu_b);
    let b = Tensor::<f32, WgpuBackend>::from_slice_on(vec![2, 3], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &wgpu_b);

    // coeus_ops::add
    let c = coeus_ops::add(&a, &b, &wgpu_b);
    let c_cpu = c.to_backend_on(&wgpu_b, &seq);
    assert_eq!(c_cpu.as_slice(), &[11.0, 18.0, 33.0, 36.0, 55.0, 54.0]);

    // coeus_ops::relu
    let d = coeus_ops::relu(&a, &wgpu_b);
    let d_cpu = d.to_backend_on(&wgpu_b, &seq);
    assert_eq!(d_cpu.as_slice(), &[1.0, 0.0, 3.0, 0.0, 5.0, 0.0]);

    // coeus_ops::matmul
    let m1 = Tensor::<f32, WgpuBackend>::from_slice_on(vec![2, 2], &[1.0, 2.0, 3.0, 4.0], &wgpu_b);
    let m2 = Tensor::<f32, WgpuBackend>::from_slice_on(vec![2, 2], &[5.0, 6.0, 7.0, 8.0], &wgpu_b);
    let mr = coeus_ops::matmul(&m1, &m2, &wgpu_b);
    let mr_cpu = mr.to_backend_on(&wgpu_b, &seq);
    // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
    // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
    assert_eq!(mr_cpu.as_slice(), &[19.0, 22.0, 43.0, 50.0]);

    // coeus_ops::sum_axis along axis 0
    let s0 = coeus_ops::sum_axis(&a, 0, &wgpu_b);
    let s0_cpu = s0.to_backend_on(&wgpu_b, &seq);
    assert_eq!(s0_cpu.as_slice(), &[-3.0, 3.0, -3.0]);
}

#[test]
fn test_wgpu_tiled_matmul() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    // 1. Create a non-tile-multiple shape, e.g. 20 x 24 and 24 x 18.
    let m = 20;
    let k = 24;
    let n = 18;

    let a_data: Vec<f32> = (0..m * k).map(|x| (x as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..k * n).map(|x| (x as f32) * 0.02).collect();

    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![m, k], &a_data);
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![k, n], &b_data);

    let a_wgpu = a_seq.to_backend_on(&seq, &wgpu_b);
    let b_wgpu = b_seq.to_backend_on(&seq, &wgpu_b);

    // Run custom matmul
    let c_wgpu = coeus_wgpu::matmul(&a_wgpu, &b_wgpu);
    let c_seq_res = c_wgpu.to_backend_on(&wgpu_b, &seq);

    // Run seq matmul as reference
    let c_seq_expected = coeus_ops::matmul(&a_seq, &b_seq, &seq);

    // Check sizes and values
    assert_eq!(c_seq_res.shape(), c_seq_expected.shape());
    let slice_res = c_seq_res.as_slice();
    let slice_expected = c_seq_expected.as_slice();
    for i in 0..slice_res.len() {
        let diff = (slice_res[i] - slice_expected[i]).abs();
        assert!(diff < 1e-4, "Mismatch at {}: {} vs {} (diff {})", i, slice_res[i], slice_expected[i], diff);
    }
}

#[test]
fn test_wgpu_conv() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    // 1D Convolution test
    let input_data = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0]; // shape [1, 2, 3]
    let weight_data = vec![1.0f32, 2.0, 3.0, 4.0]; // shape [2, 2, 1]
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 2, 3], &input_data);
    let weight_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 2, 1], &weight_data);

    let input_wgpu = input_seq.to_backend_on(&seq, &wgpu_b);
    let weight_wgpu = weight_seq.to_backend_on(&seq, &wgpu_b);

    // output shape: [1, 2, 3]
    let mut out_wgpu_storage = wgpu_b.allocate::<f32>(6);
    let out_layout = coeus_core::Layout::new(vec![1, 2, 3].into());

    coeus_ops::BackendOps::conv1d(
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
    );

    let out_tensor_wgpu: Tensor<f32, WgpuBackend> = Tensor::from_raw_parts(out_wgpu_storage, out_layout.clone());
    let out_seq = out_tensor_wgpu.to_backend_on(&wgpu_b, &seq);

    let mut out_expected_storage = seq.allocate::<f32>(6);
    coeus_ops::BackendOps::conv1d(
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
    );
    let out_expected: Tensor<f32, SequentialBackend> = Tensor::from_raw_parts(out_expected_storage, out_layout);

    let out_seq_slice: &[f32] = out_seq.as_slice();
    let out_expected_slice: &[f32] = out_expected.as_slice();
    for (i, (&res, &exp)) in out_seq_slice.iter().zip(out_expected_slice.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-4f32, "Conv1D mismatch at {}: {} vs {}", i, res, exp);
    }

    // 2D Convolution test
    // input shape: [1, 1, 4, 4]
    let input_2d_data = vec![
        1.0f32, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];
    // weight shape: [1, 1, 3, 3]
    let weight_2d_data = vec![
        1.0f32, 0.0, -1.0,
        1.0, 0.0, -1.0,
        1.0, 0.0, -1.0,
    ];
    let input_2d_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 4, 4], &input_2d_data);
    let weight_2d_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3, 3], &weight_2d_data);

    let input_2d_wgpu = input_2d_seq.to_backend_on(&seq, &wgpu_b);
    let weight_2d_wgpu = weight_2d_seq.to_backend_on(&seq, &wgpu_b);

    // output shape: [1, 1, 2, 2]
    let mut out_2d_wgpu_storage = wgpu_b.allocate::<f32>(4);
    let out_2d_layout = coeus_core::Layout::new(vec![1, 1, 2, 2].into());

    coeus_ops::BackendOps::conv2d(
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
    );

    let out_2d_tensor_wgpu: Tensor<f32, WgpuBackend> = Tensor::from_raw_parts(out_2d_wgpu_storage, out_2d_layout.clone());
    let out_2d_seq = out_2d_tensor_wgpu.to_backend_on(&wgpu_b, &seq);

    let mut out_2d_expected_storage = seq.allocate::<f32>(4);
    coeus_ops::BackendOps::conv2d(
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
    );
    let out_2d_expected: Tensor<f32, SequentialBackend> = Tensor::from_raw_parts(out_2d_expected_storage, out_2d_layout);

    let out_2d_seq_slice: &[f32] = out_2d_seq.as_slice();
    let out_2d_expected_slice: &[f32] = out_2d_expected.as_slice();
    for (i, (&res, &exp)) in out_2d_seq_slice.iter().zip(out_2d_expected_slice.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-4f32, "Conv2D mismatch at {}: {} vs {}", i, res, exp);
    }
}

#[test]
fn test_wgpu_conv_backward() {
    use coeus_core::CpuAddressableStorage;
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    // 1D Conv backward test
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

    coeus_ops::BackendOps::conv1d_backward(
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

    let mut gi_expected = seq.allocate::<f32>(6);
    seq.fill(&mut gi_expected, 0.0);
    let mut gw_expected = seq.allocate::<f32>(4);
    seq.fill(&mut gw_expected, 0.0);
    let mut gb_expected = seq.allocate::<f32>(2);
    seq.fill(&mut gb_expected, 0.0);

    coeus_ops::BackendOps::conv1d_backward(
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

    let gi_wgpu_tensor: Tensor<f32, WgpuBackend> = Tensor::from_raw_parts(gi_wgpu, gi_layout);
    let gi_wgpu_cpu = gi_wgpu_tensor.to_backend_on(&wgpu_b, &seq);
    let gi_expected_slice = gi_expected.as_slice();
    for (i, (&res, &exp)) in gi_wgpu_cpu.as_slice().iter().zip(gi_expected_slice.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-4f32, "Conv1D grad_input mismatch at {}: {} vs {}", i, res, exp);
    }

    let gw_wgpu_tensor: Tensor<f32, WgpuBackend> = Tensor::from_raw_parts(gw_wgpu, gw_layout);
    let gw_wgpu_cpu = gw_wgpu_tensor.to_backend_on(&wgpu_b, &seq);
    let gw_expected_slice = gw_expected.as_slice();
    for (i, (&res, &exp)) in gw_wgpu_cpu.as_slice().iter().zip(gw_expected_slice.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-4f32, "Conv1D grad_weight mismatch at {}: {} vs {}", i, res, exp);
    }

    let gb_wgpu_tensor: Tensor<f32, WgpuBackend> = Tensor::from_raw_parts(gb_wgpu, coeus_core::Layout::new(vec![2].into()));
    let gb_wgpu_cpu = gb_wgpu_tensor.to_backend_on(&wgpu_b, &seq);
    let gb_expected_slice = gb_expected.as_slice();
    for (i, (&res, &exp)) in gb_wgpu_cpu.as_slice().iter().zip(gb_expected_slice.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-4f32, "Conv1D grad_bias mismatch at {}: {} vs {}", i, res, exp);
    }
}

#[test]
fn test_wgpu_cow_semantics() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    // 1. Create source tensor on GPU
    let a = Tensor::<f32, WgpuBackend>::from_slice_on(vec![3], &[1.0, 2.0, 3.0], &wgpu_b);

    // 2. Clone tensor (shares underlying wgpu buffer through Arc)
    let mut b = a.clone();

    // 3. Mutate the clone in-place (triggers COW via make_unique)
    wgpu_b.fill(b.storage_mut(), 10.0);

    // 4. Transfer back to host CPU to verify isolation
    let a_cpu = a.to_backend_on(&wgpu_b, &seq);
    let b_cpu = b.to_backend_on(&wgpu_b, &seq);

    // 5. Assert value-semantic correctness
    assert_eq!(a_cpu.as_slice(), &[1.0, 2.0, 3.0]);
    assert_eq!(b_cpu.as_slice(), &[10.0, 10.0, 10.0]);
}

#[test]
fn test_wgpu_max_pool2d() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 4, 4], &input_data);
    let input_wgpu = input_seq.to_backend_on(&seq, &wgpu_b);

    let mut out_wgpu_storage = wgpu_b.allocate::<f32>(4);
    let out_layout = coeus_core::Layout::new(vec![1, 1, 2, 2].into());

    coeus_ops::BackendOps::max_pool2d(
        &wgpu_b,
        input_wgpu.storage(),
        input_wgpu.layout(),
        2, // kernel_size
        2, // stride
        0, // padding
        1, // dilation
        &mut out_wgpu_storage,
        &out_layout,
    );

    let out_wgpu_tensor: Tensor<f32, WgpuBackend> = Tensor::from_raw_parts(out_wgpu_storage, out_layout.clone());
    let out_wgpu_cpu = out_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    assert_eq!(out_wgpu_cpu.as_slice(), &[6.0, 8.0, 14.0, 16.0]);

    // Backward pass
    let grad_out_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let grad_out_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2], &grad_out_data);
    let grad_out_wgpu = grad_out_seq.to_backend_on(&seq, &wgpu_b);

    let mut grad_input_wgpu_storage = wgpu_b.allocate::<f32>(16);
    wgpu_b.fill(&mut grad_input_wgpu_storage, 0.0);
    let gi_layout = coeus_core::Layout::new(vec![1, 1, 4, 4].into());

    coeus_ops::BackendOps::max_pool2d_backward(
        &wgpu_b,
        grad_out_wgpu.storage(),
        grad_out_wgpu.layout(),
        input_wgpu.storage(),
        input_wgpu.layout(),
        2, // kernel_size
        2, // stride
        0, // padding
        1, // dilation
        &mut grad_input_wgpu_storage,
        &gi_layout,
    );

    let gi_wgpu_tensor: Tensor<f32, WgpuBackend> = Tensor::from_raw_parts(grad_input_wgpu_storage, gi_layout);
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

    coeus_ops::BackendOps::avg_pool2d(
        &wgpu_b,
        input_wgpu.storage(),
        input_wgpu.layout(),
        2, // kernel_size
        2, // stride
        0, // padding
        1, // dilation
        &mut out_wgpu_storage,
        &out_layout,
    );

    let out_wgpu_tensor: Tensor<f32, WgpuBackend> = Tensor::from_raw_parts(out_wgpu_storage, out_layout.clone());
    let out_wgpu_cpu = out_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    assert_eq!(out_wgpu_cpu.as_slice(), &[3.5, 5.5, 11.5, 13.5]);

    // Backward pass
    let grad_out_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let grad_out_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2], &grad_out_data);
    let grad_out_wgpu = grad_out_seq.to_backend_on(&seq, &wgpu_b);

    let mut grad_input_wgpu_storage = wgpu_b.allocate::<f32>(16);
    wgpu_b.fill(&mut grad_input_wgpu_storage, 0.0);
    let gi_layout = coeus_core::Layout::new(vec![1, 1, 4, 4].into());

    coeus_ops::BackendOps::avg_pool2d_backward(
        &wgpu_b,
        grad_out_wgpu.storage(),
        grad_out_wgpu.layout(),
        2, // kernel_size
        2, // stride
        0, // padding
        1, // dilation
        &mut grad_input_wgpu_storage,
        &gi_layout,
    );

    let gi_wgpu_tensor: Tensor<f32, WgpuBackend> = Tensor::from_raw_parts(grad_input_wgpu_storage, gi_layout);
    let gi_wgpu_cpu = gi_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    let expected_gi = vec![
        0.25f32, 0.25, 0.5, 0.5,
        0.25, 0.25, 0.5, 0.5,
        0.75, 0.75, 1.0, 1.0,
        0.75, 0.75, 1.0, 1.0,
    ];
    assert_eq!(gi_wgpu_cpu.as_slice(), &expected_gi);
}

#[test]
fn test_wgpu_silu_parity() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let input_data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let input_cpu = Tensor::<f32, SequentialBackend>::from_slice([5], &input_data);
    let input_gpu = input_cpu.to_backend_on(&seq, &wgpu_b);

    let var_cpu = coeus_autograd::Var::new(input_cpu, true);
    let var_gpu = coeus_autograd::Var::new(input_gpu, true);

    let out_cpu = coeus_nn::silu(&var_cpu);
    let out_gpu = coeus_nn::silu(&var_gpu);

    // Parity of forward output
    let out_gpu_cpu = out_gpu.tensor.to_backend_on(&wgpu_b, &seq);
    let out_cpu_slice = out_cpu.tensor.as_slice();
    let out_gpu_slice = out_gpu_cpu.as_slice();

    for i in 0..5 {
        assert!((out_cpu_slice[i] - out_gpu_slice[i]).abs() < 1e-5);
    }

    // Parity of backward gradients
    out_cpu.backward();
    out_gpu.backward();

    let grad_cpu = var_cpu.grad().unwrap();
    let grad_gpu = var_gpu.grad().unwrap();
    let grad_gpu_cpu = grad_gpu.to_backend_on(&wgpu_b, &seq);

    let grad_cpu_slice = grad_cpu.as_slice();
    let grad_gpu_slice = grad_gpu_cpu.as_slice();

    for i in 0..5 {
        assert!((grad_cpu_slice[i] - grad_gpu_slice[i]).abs() < 1e-5);
    }
}

#[test]
fn test_wgpu_mish_parity() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let input_data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let input_cpu = Tensor::<f32, SequentialBackend>::from_slice([5], &input_data);
    let input_gpu = input_cpu.to_backend_on(&seq, &wgpu_b);

    let var_cpu = coeus_autograd::Var::new(input_cpu, true);
    let var_gpu = coeus_autograd::Var::new(input_gpu, true);

    let out_cpu = coeus_nn::mish(&var_cpu);
    let out_gpu = coeus_nn::mish(&var_gpu);

    // Parity of forward output
    let out_gpu_cpu = out_gpu.tensor.to_backend_on(&wgpu_b, &seq);
    let out_cpu_slice = out_cpu.tensor.as_slice();
    let out_gpu_slice = out_gpu_cpu.as_slice();

    for i in 0..5 {
        assert!((out_cpu_slice[i] - out_gpu_slice[i]).abs() < 1e-5);
    }

    // Parity of backward gradients
    out_cpu.backward();
    out_gpu.backward();

    let grad_cpu = var_cpu.grad().unwrap();
    let grad_gpu = var_gpu.grad().unwrap();
    let grad_gpu_cpu = grad_gpu.to_backend_on(&wgpu_b, &seq);

    let grad_cpu_slice = grad_cpu.as_slice();
    let grad_gpu_slice = grad_gpu_cpu.as_slice();

    for i in 0..5 {
        assert!((grad_cpu_slice[i] - grad_gpu_slice[i]).abs() < 1e-5);
    }
}

#[test]
fn test_wgpu_conv3d() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    // Input shape [1, 1, 3, 3, 3]
    let input_data: Vec<f32> = (1..=27).map(|x| x as f32).collect();
    // Weight shape [1, 1, 2, 2, 2]
    let weight_data: Vec<f32> = vec![1.0, 0.0, -1.0, 1.0, 2.0, -2.0, 0.5, -0.5];
    // Bias shape [1]
    let bias_data: Vec<f32> = vec![0.5];

    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3, 3, 3], &input_data);
    let weight_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &weight_data);
    let bias_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1], &bias_data);

    let input_wgpu = input_seq.to_backend_on(&seq, &wgpu_b);
    let weight_wgpu = weight_seq.to_backend_on(&seq, &wgpu_b);
    let bias_wgpu = bias_seq.to_backend_on(&seq, &wgpu_b);

    // Output shape: [1, 1, 2, 2, 2]
    let out_layout = coeus_core::Layout::new(vec![1, 1, 2, 2, 2].into());
    let mut out_wgpu_storage = wgpu_b.allocate::<f32>(8);

    coeus_ops::BackendOps::conv3d(
        &wgpu_b,
        input_wgpu.storage(),
        input_wgpu.layout(),
        weight_wgpu.storage(),
        weight_wgpu.layout(),
        Some(bias_wgpu.storage()),
        1, // stride
        0, // padding
        1, // dilation
        &mut out_wgpu_storage,
        &out_layout,
    );

    let out_tensor_wgpu = Tensor::<f32, WgpuBackend>::from_raw_parts(out_wgpu_storage, out_layout.clone());
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
    let out_expected = Tensor::<f32, SequentialBackend>::from_raw_parts(out_expected_storage, out_layout);

    let out_seq_slice = out_wgpu_cpu.as_slice();
    let out_expected_slice = out_expected.as_slice();
    for (i, (&res, &exp)) in out_seq_slice.iter().zip(out_expected_slice.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-4f32, "Conv3D mismatch at {}: {} vs {}", i, res, exp);
    }
}

#[test]
fn test_wgpu_conv3d_backward() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    // shapes:
    // grad_out: [1, 1, 2, 2, 2]
    // input: [1, 1, 3, 3, 3]
    // weight: [1, 1, 2, 2, 2]
    let grad_out_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let input_data: Vec<f32> = (1..=27).map(|x| x as f32).collect();
    let weight_data: Vec<f32> = vec![1.0, 0.0, -1.0, 1.0, 2.0, -2.0, 0.5, -0.5];

    let grad_out_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &grad_out_data);
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 3, 3, 3], &input_data);
    let weight_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &weight_data);

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
        1, // stride
        0, // padding
        1, // dilation
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

    // Verify grad_input
    let gi_wgpu_tensor = Tensor::<f32, WgpuBackend>::from_raw_parts(gi_wgpu, gi_layout);
    let gi_wgpu_cpu = gi_wgpu_tensor.to_backend_on(&wgpu_b, &seq);
    let gi_expected_slice = gi_expected.as_slice();
    for (i, (&res, &exp)) in gi_wgpu_cpu.as_slice().iter().zip(gi_expected_slice.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-4f32, "Conv3D grad_input mismatch at {}: {} vs {}", i, res, exp);
    }

    // Verify grad_weight
    let gw_wgpu_tensor = Tensor::<f32, WgpuBackend>::from_raw_parts(gw_wgpu, gw_layout);
    let gw_wgpu_cpu = gw_wgpu_tensor.to_backend_on(&wgpu_b, &seq);
    let gw_expected_slice = gw_expected.as_slice();
    for (i, (&res, &exp)) in gw_wgpu_cpu.as_slice().iter().zip(gw_expected_slice.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-4f32, "Conv3D grad_weight mismatch at {}: {} vs {}", i, res, exp);
    }

    // Verify grad_bias
    let gb_wgpu_tensor = Tensor::<f32, WgpuBackend>::from_raw_parts(gb_wgpu, coeus_core::Layout::new(vec![1].into()));
    let gb_wgpu_cpu = gb_wgpu_tensor.to_backend_on(&wgpu_b, &seq);
    let gb_expected_slice = gb_expected.as_slice();
    for (i, (&res, &exp)) in gb_wgpu_cpu.as_slice().iter().zip(gb_expected_slice.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-4f32, "Conv3D grad_bias mismatch at {}: {} vs {}", i, res, exp);
    }
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

    coeus_ops::BackendOps::max_pool3d(
        &wgpu_b,
        input_wgpu.storage(),
        input_wgpu.layout(),
        2, // kernel_size
        1, // stride
        0, // padding
        1, // dilation
        &mut out_wgpu_storage,
        &out_layout,
    );

    let out_wgpu_tensor = Tensor::<f32, WgpuBackend>::from_raw_parts(out_wgpu_storage, out_layout.clone());
    let out_wgpu_cpu = out_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    let mut out_expected_storage = seq.allocate::<f32>(8);
    coeus_ops::BackendOps::max_pool3d(
        &seq,
        input_seq.storage(),
        input_seq.layout(),
        2,
        1,
        0,
        1,
        &mut out_expected_storage,
        &out_layout,
    );
    let out_expected = Tensor::<f32, SequentialBackend>::from_raw_parts(out_expected_storage, out_layout);

    assert_eq!(out_wgpu_cpu.as_slice(), out_expected.as_slice());

    // Backward pass
    let grad_out_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let grad_out_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &grad_out_data);
    let grad_out_wgpu = grad_out_seq.to_backend_on(&seq, &wgpu_b);

    let mut grad_input_wgpu_storage = wgpu_b.allocate::<f32>(27);
    wgpu_b.fill(&mut grad_input_wgpu_storage, 0.0);
    let gi_layout = coeus_core::Layout::new(vec![1, 1, 3, 3, 3].into());

    coeus_ops::BackendOps::max_pool3d_backward(
        &wgpu_b,
        grad_out_wgpu.storage(),
        grad_out_wgpu.layout(),
        input_wgpu.storage(),
        input_wgpu.layout(),
        2, // kernel_size
        1, // stride
        0, // padding
        1, // dilation
        &mut grad_input_wgpu_storage,
        &gi_layout,
    );

    let gi_wgpu_tensor = Tensor::<f32, WgpuBackend>::from_raw_parts(grad_input_wgpu_storage, gi_layout.clone());
    let gi_wgpu_cpu = gi_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    let mut grad_input_expected_storage = seq.allocate::<f32>(27);
    seq.fill(&mut grad_input_expected_storage, 0.0);
    coeus_ops::BackendOps::max_pool3d_backward(
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
    );
    let gi_expected = Tensor::<f32, SequentialBackend>::from_raw_parts(grad_input_expected_storage, gi_layout);

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

    coeus_ops::BackendOps::avg_pool3d(
        &wgpu_b,
        input_wgpu.storage(),
        input_wgpu.layout(),
        2, // kernel_size
        1, // stride
        0, // padding
        1, // dilation
        &mut out_wgpu_storage,
        &out_layout,
    );

    let out_wgpu_tensor = Tensor::<f32, WgpuBackend>::from_raw_parts(out_wgpu_storage, out_layout.clone());
    let out_wgpu_cpu = out_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    let mut out_expected_storage = seq.allocate::<f32>(8);
    coeus_ops::BackendOps::avg_pool3d(
        &seq,
        input_seq.storage(),
        input_seq.layout(),
        2,
        1,
        0,
        1,
        &mut out_expected_storage,
        &out_layout,
    );
    let out_expected = Tensor::<f32, SequentialBackend>::from_raw_parts(out_expected_storage, out_layout);

    let res_slice = out_wgpu_cpu.as_slice();
    let exp_slice = out_expected.as_slice();
    for (i, (&res, &exp)) in res_slice.iter().zip(exp_slice.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-4f32, "AvgPool3D mismatch at {}: {} vs {}", i, res, exp);
    }

    // Backward pass
    let grad_out_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let grad_out_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 1, 2, 2, 2], &grad_out_data);
    let grad_out_wgpu = grad_out_seq.to_backend_on(&seq, &wgpu_b);

    let mut grad_input_wgpu_storage = wgpu_b.allocate::<f32>(27);
    wgpu_b.fill(&mut grad_input_wgpu_storage, 0.0);
    let gi_layout = coeus_core::Layout::new(vec![1, 1, 3, 3, 3].into());

    coeus_ops::BackendOps::avg_pool3d_backward(
        &wgpu_b,
        grad_out_wgpu.storage(),
        grad_out_wgpu.layout(),
        2, // kernel_size
        1, // stride
        0, // padding
        1, // dilation
        &mut grad_input_wgpu_storage,
        &gi_layout,
    );

    let gi_wgpu_tensor = Tensor::<f32, WgpuBackend>::from_raw_parts(grad_input_wgpu_storage, gi_layout.clone());
    let gi_wgpu_cpu = gi_wgpu_tensor.to_backend_on(&wgpu_b, &seq);

    let mut grad_input_expected_storage = seq.allocate::<f32>(27);
    seq.fill(&mut grad_input_expected_storage, 0.0);
    coeus_ops::BackendOps::avg_pool3d_backward(
        &seq,
        grad_out_seq.storage(),
        grad_out_seq.layout(),
        2,
        1,
        0,
        1,
        &mut grad_input_expected_storage,
        &gi_layout,
    );
    let gi_expected = Tensor::<f32, SequentialBackend>::from_raw_parts(grad_input_expected_storage, gi_layout);

    let res_gi_slice = gi_wgpu_cpu.as_slice();
    let exp_gi_slice = gi_expected.as_slice();
    for (i, (&res, &exp)) in res_gi_slice.iter().zip(exp_gi_slice.iter()).enumerate() {
        assert!((res - exp).abs() < 1e-4f32, "AvgPool3D backward mismatch at {}: {} vs {}", i, res, exp);
    }
}

#[test]
fn test_wgpu_adamw() {
    use coeus_ops::BackendOps;
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let shape = vec![2, 3];
    let p_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let g_data = vec![0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6];
    let m_data = vec![0.01f32, 0.02, 0.03, 0.04, 0.05, 0.06];
    let v_data = vec![0.11f32, 0.12, 0.13, 0.14, 0.15, 0.16];

    // CPU tensors
    let mut p_seq = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &p_data);
    let g_seq = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &g_data);
    let mut m_seq = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &m_data);
    let mut v_seq = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &v_data);

    // GPU tensors
    let mut p_wgpu = p_seq.to_backend_on(&seq, &wgpu_b);
    let g_wgpu = g_seq.to_backend_on(&seq, &wgpu_b);
    let mut m_wgpu = m_seq.to_backend_on(&seq, &wgpu_b);
    let mut v_wgpu = v_seq.to_backend_on(&seq, &wgpu_b);

    // Hyperparameters
    let lr = 0.05f32;
    let beta1 = 0.9f32;
    let beta2 = 0.99f32;
    let eps = 1e-6f32;
    let weight_decay = 0.02f32;
    let t = 3usize;

    // CPU step
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

    // GPU step
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

    // Copy GPU tensors back to CPU for comparison
    let p_res = p_wgpu.to_backend_on(&wgpu_b, &seq);
    let m_res = m_wgpu.to_backend_on(&wgpu_b, &seq);
    let v_res = v_wgpu.to_backend_on(&wgpu_b, &seq);

    // Assert parity
    for i in 0..p_seq.as_slice().len() {
        assert!((p_res.as_slice()[i] - p_seq.as_slice()[i]).abs() < 1e-5, "Mismatch in p at index {}: GPU={:?}, CPU={:?}", i, p_res.as_slice()[i], p_seq.as_slice()[i]);
        assert!((m_res.as_slice()[i] - m_seq.as_slice()[i]).abs() < 1e-5, "Mismatch in m at index {}: GPU={:?}, CPU={:?}", i, m_res.as_slice()[i], m_seq.as_slice()[i]);
        assert!((v_res.as_slice()[i] - v_seq.as_slice()[i]).abs() < 1e-5, "Mismatch in v at index {}: GPU={:?}, CPU={:?}", i, v_res.as_slice()[i], v_seq.as_slice()[i]);
    }
}

