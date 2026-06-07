use coeus_core::{SequentialBackend, ComputeBackend};
use coeus_tensor::{Tensor, Transpose};
use coeus_cuda::CudaBackend;

#[test]
fn test_cuda_backend_compilation_and_fallback() {
    let cuda_b = CudaBackend::new();
    assert_eq!(cuda_b.name(), "cuda");
    
    // Verify that constructing tensors and utilizing dynamic fallback compilations works cleanly.
    let data = vec![1.0f32, 2.0, 3.0];
    let seq = SequentialBackend::new();
    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![3], &data);
    
    // Transfer to CUDA (allocates dummy/empty wrapper if no physical CUDA driver is available at runtime)
    let a_cuda = a_seq.to_backend_on(&seq, &cuda_b);
    assert_eq!(a_cuda.shape(), &[3]);
    
    // Transfer back from CUDA
    let a_seq_back = a_cuda.to_backend_on(&cuda_b, &seq);
    assert_eq!(a_seq_back.shape(), &[3]);
}

#[test]
fn test_cuda_backend_ops() {
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0];

    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &a_data);
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &b_data);

    let a_cuda = a_seq.to_backend_on(&seq, &cuda_b);
    let b_cuda = b_seq.to_backend_on(&seq, &cuda_b);

    // Verify addition
    let c_cuda = coeus_ops::add(&a_cuda, &b_cuda, &cuda_b);
    let c_seq = c_cuda.to_backend_on(&cuda_b, &seq);

    // Verify matrix multiplication
    let m_cuda = coeus_ops::matmul(&a_cuda, &b_cuda, &cuda_b);
    let m_seq = m_cuda.to_backend_on(&cuda_b, &seq);

    // Verify SiLU
    let s_cuda = coeus_ops::silu(&a_cuda, &cuda_b);
    let s_seq = s_cuda.to_backend_on(&cuda_b, &seq);

    // Verify Mish
    let mish_cuda = coeus_ops::mish(&a_cuda, &cuda_b);
    let mish_seq = mish_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        assert_eq!(c_seq.as_slice(), &[11.0, 22.0, 33.0, 44.0]);
        // MatMul of [1, 2; 3, 4] and [10, 20; 30, 40]
        // [1*10+2*30, 1*20+2*40] = [70, 100]
        // [3*10+4*30, 3*20+4*40] = [150, 220]
        assert_eq!(m_seq.as_slice(), &[70.0, 100.0, 150.0, 220.0]);
        
        // SiLU verification
        assert!((s_seq.as_slice()[0] - 0.7310586).abs() < 1e-5);
        assert!((s_seq.as_slice()[1] - 1.7615942).abs() < 1e-5);
        assert!((s_seq.as_slice()[2] - 2.8577224).abs() < 1e-5);
        assert!((s_seq.as_slice()[3] - 3.9280551).abs() < 1e-5);

        // Mish verification
        assert!((mish_seq.as_slice()[0] - 0.8656606).abs() < 1e-5);
        assert!((mish_seq.as_slice()[1] - 1.943866).abs() < 1e-5);
        assert!((mish_seq.as_slice()[2] - 2.986349).abs() < 1e-5);
        assert!((mish_seq.as_slice()[3] - 3.997317).abs() < 1e-5);
    } else {
        assert_eq!(c_seq.shape(), &[2, 2]);
        assert_eq!(m_seq.shape(), &[2, 2]);
        assert_eq!(s_seq.shape(), &[2, 2]);
        assert_eq!(mish_seq.shape(), &[2, 2]);
    }
}

#[test]
fn test_cuda_backend_conv_and_reduce() {
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 4], &a_data);
    let a_cuda = a_seq.to_backend_on(&seq, &cuda_b);

    // Sum reduction along axis 1
    let r_cuda = coeus_ops::sum_axis(&a_cuda, 1, &cuda_b);
    let r_seq = r_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        // [1+2+3+4, 5+6+7+8] = [10.0, 26.0]
        assert_eq!(r_seq.as_slice(), &[10.0, 26.0]);
    }

    // Conv1D forward test
    let input_data = vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0]; // shape [1, 2, 3]
    let weight_data = vec![1.0f32, 2.0, 3.0, 4.0]; // shape [2, 2, 1]
    let input_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![1, 2, 3], &input_data);
    let weight_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 2, 1], &weight_data);

    let input_cuda = input_seq.to_backend_on(&seq, &cuda_b);
    let weight_cuda = weight_seq.to_backend_on(&seq, &cuda_b);

    // output shape: [1, 2, 3]
    let mut out_cuda_storage = cuda_b.allocate::<f32>(6);
    let out_layout = coeus_core::Layout::new(vec![1, 2, 3].into());

    coeus_ops::BackendOps::conv1d(
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
    );

    let out_tensor_cuda: Tensor<f32, CudaBackend> = Tensor::from_raw_parts(out_cuda_storage, out_layout.clone());
    let out_seq = out_tensor_cuda.to_backend_on(&cuda_b, &seq);

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

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        let out_seq_slice: &[f32] = out_seq.as_slice();
        let out_expected_slice: &[f32] = out_expected.as_slice();
        for (i, (&res, &exp)) in out_seq_slice.iter().zip(out_expected_slice.iter()).enumerate() {
            assert!((res - exp).abs() < 1e-4f32, "Mismatch at {}: {} vs {}", i, res, exp);
        }
    }
}

#[test]
fn test_cuda_conv_backward() {
    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        use coeus_core::CpuAddressableStorage;
        let seq = SequentialBackend::new();
        let cuda_b = CudaBackend::new();

        // 1D Conv backward test
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

        coeus_ops::BackendOps::conv1d_backward(
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

        let gi_cuda_tensor: Tensor<f32, CudaBackend> = Tensor::from_raw_parts(gi_cuda, gi_layout);
        let gi_cuda_cpu = gi_cuda_tensor.to_backend_on(&cuda_b, &seq);
        let gi_expected_slice = gi_expected.as_slice();
        for (i, (&res, &exp)) in gi_cuda_cpu.as_slice().iter().zip(gi_expected_slice.iter()).enumerate() {
            assert!((res - exp).abs() < 1e-4f32, "Conv1D grad_input mismatch at {}: {} vs {}", i, res, exp);
        }

        let gw_cuda_tensor: Tensor<f32, CudaBackend> = Tensor::from_raw_parts(gw_cuda, gw_layout);
        let gw_cuda_cpu = gw_cuda_tensor.to_backend_on(&cuda_b, &seq);
        let gw_expected_slice = gw_expected.as_slice();
        for (i, (&res, &exp)) in gw_cuda_cpu.as_slice().iter().zip(gw_expected_slice.iter()).enumerate() {
            assert!((res - exp).abs() < 1e-4f32, "Conv1D grad_weight mismatch at {}: {} vs {}", i, res, exp);
        }

        let gb_cuda_tensor: Tensor<f32, CudaBackend> = Tensor::from_raw_parts(gb_cuda, coeus_core::Layout::new(vec![2].into()));
        let gb_cuda_cpu = gb_cuda_tensor.to_backend_on(&cuda_b, &seq);
        let gb_expected_slice = gb_expected.as_slice();
        for (i, (&res, &exp)) in gb_cuda_cpu.as_slice().iter().zip(gb_expected_slice.iter()).enumerate() {
            assert!((res - exp).abs() < 1e-4f32, "Conv1D grad_bias mismatch at {}: {} vs {}", i, res, exp);
        }
    }
}

#[test]
fn test_cuda_strided_ops() {
    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        let seq = SequentialBackend::new();
        let cuda_b = CudaBackend::new();

        // Binary strided addition with transpose/broadcast
        // Create 2x3 input
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![10.0f32, 20.0, 30.0];

        let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &a_data);
        let b_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![3], &b_data);

        // Transpose A to shape 3x2 (non-contiguous)
        let a_seq_t = a_seq.transpose();

        let a_cuda_t = a_seq_t.to_backend_on(&seq, &cuda_b);
        let b_cuda = b_seq.to_backend_on(&seq, &cuda_b);

        // Verify shape and strides
        assert_eq!(a_cuda_t.shape(), &[3, 2]);
        assert_eq!(b_cuda.shape(), &[3]);

        // Broadcasted/strided elementwise addition (3x2 + 3 -> 3x2)
        let c_cuda = coeus_ops::add(&a_cuda_t, &b_cuda, &cuda_b);
        let c_seq = c_cuda.to_backend_on(&cuda_b, &seq);

        let c_expected = coeus_ops::add(&a_seq_t, &b_seq, &seq);

        for (i, (&res, &exp)) in c_seq.as_slice().iter().zip(c_expected.as_slice().iter()).enumerate() {
            assert!((res - exp).abs() < 1e-4f32, "Strided add mismatch at {}: {} vs {}", i, res, exp);
        }

        // Unary strided operation
        let u_cuda = coeus_ops::relu(&a_cuda_t, &cuda_b);
        let u_seq = u_cuda.to_backend_on(&cuda_b, &seq);
        let u_expected = coeus_ops::relu(&a_seq_t, &seq);

        for (i, (&res, &exp)) in u_seq.as_slice().iter().zip(u_expected.as_slice().iter()).enumerate() {
            assert!((res - exp).abs() < 1e-4f32, "Strided relu mismatch at {}: {} vs {}", i, res, exp);
        }
    }
}

#[test]
fn test_cuda_evaluate_fused() {
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let a_data = vec![1.0f32, -2.0, 3.0, -4.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0];

    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &a_data);
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 2], &b_data);

    let a_cuda = a_seq.to_backend_on(&seq, &cuda_b);
    let b_cuda = b_seq.to_backend_on(&seq, &cuda_b);

    // Build fused expression
    use coeus_ops::fuse::TensorExprExt;
    let expr = (a_cuda.expr() * b_cuda.expr() + 5.0).relu();

    // Evaluate on CUDA (runs via CPU fallback)
    let out_cuda = coeus_cuda::evaluate_fused(&expr);
    let out_seq = out_cuda.to_backend_on(&cuda_b, &seq);

    // Compute expected CPU values manually
    let mut expected = vec![0.0f32; 4];
    for i in 0..4 {
        let val = a_data[i] * b_data[i] + 5.0;
        expected[i] = if val > 0.0 { val } else { 0.0 };
    }

    assert_eq!(out_seq.shape(), &[2, 2]);
    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        assert_eq!(out_seq.as_slice(), &expected);
    }
}

#[test]
fn test_cuda_jit_fusion_correctness() {
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let a_data = vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let c_data = vec![-5.0f32, 5.0, -10.0, 10.0, -15.0, 15.0];
    let shape = vec![2, 3];

    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &a_data);
    let b_seq = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &b_data);
    let c_seq = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &c_data);

    let a_cuda = a_seq.to_backend_on(&seq, &cuda_b);
    let b_cuda = b_seq.to_backend_on(&seq, &cuda_b);
    let c_cuda = c_seq.to_backend_on(&seq, &cuda_b);

    // Dynamic fused expression: (a * b + c).relu().sigmoid()
    use coeus_ops::fuse::TensorExprExt;
    let expr = (a_cuda.expr() * b_cuda.expr() + c_cuda.expr()).relu().sigmoid();

    let out_cuda = coeus_cuda::evaluate_fused(&expr);
    let out_seq = out_cuda.to_backend_on(&cuda_b, &seq);

    // Expected CPU calculation
    let mut expected = [0.0f32; 6];
    for i in 0..6 {
        let val = a_data[i] * b_data[i] + c_data[i];
        let relu_val = if val > 0.0 { val } else { 0.0 };
        expected[i] = 1.0 / (1.0 + (-relu_val).exp());
    }

    assert_eq!(out_seq.shape(), &[2, 3]);
    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        let out_slice = out_seq.as_slice();
        for i in 0..6 {
            let diff = (out_slice[i] - expected[i]).abs();
            assert!(diff < 1e-5, "Mismatch at index {}: {} vs expected {}", i, out_slice[i], expected[i]);
        }
    }
}

#[test]
fn test_cuda_jit_reductions() {
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let a_data = vec![
        1.0f32,  2.0,  3.0,
        10.0,  -5.0,  6.0,
    ];
    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &a_data);
    let a_cuda = a_seq.to_backend_on(&seq, &cuda_b);

    // Sum reduction along axis 1 (horizontal)
    let sum_cuda = coeus_ops::sum_axis(&a_cuda, 1, &cuda_b);
    let sum_seq = sum_cuda.to_backend_on(&cuda_b, &seq);

    // Max reduction along axis 1
    let max_cuda = coeus_ops::max_axis(&a_cuda, 1, &cuda_b);
    let max_seq = max_cuda.to_backend_on(&cuda_b, &seq);

    // Min reduction along axis 1
    let min_cuda = coeus_ops::min_axis(&a_cuda, 1, &cuda_b);
    let min_seq = min_cuda.to_backend_on(&cuda_b, &seq);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        assert_eq!(sum_seq.as_slice(), &[6.0, 11.0]);
        assert_eq!(max_seq.as_slice(), &[3.0, 10.0]);
        assert_eq!(min_seq.as_slice(), &[1.0, -5.0]);
    }
}

#[test]
fn test_cuda_evaluate_fused_reduce() {
    let cuda_b = CudaBackend::new();
    let seq = SequentialBackend::new();

    let a_data = vec![
        1.0f32, -2.0, 3.0,
        10.0, -5.0, 6.0,
    ];
    let a_seq = Tensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &a_data);
    let a_cuda = a_seq.to_backend_on(&seq, &cuda_b);

    // Dynamic fused expression: (a * 2.0).relu()
    use coeus_ops::fuse::TensorExprExt;
    let expr = (a_cuda.expr() * 2.0).relu();

    // Perform fused sum reduction along axis 1
    let sum_cuda = coeus_cuda::evaluate_fused_reduce(&expr, coeus_ops::ReductionOp::Sum, 1);
    let sum_seq = sum_cuda.to_backend_on(&cuda_b, &seq);

    // Perform fused max reduction along axis 1
    let max_cuda = coeus_cuda::evaluate_fused_reduce(&expr, coeus_ops::ReductionOp::Max, 1);
    let max_seq = max_cuda.to_backend_on(&cuda_b, &seq);

    // Perform fused min reduction along axis 1
    let min_cuda = coeus_cuda::evaluate_fused_reduce(&expr, coeus_ops::ReductionOp::Min, 1);
    let min_seq = min_cuda.to_backend_on(&cuda_b, &seq);

    assert_eq!(sum_seq.shape(), &[2, 1]);
    assert_eq!(max_seq.shape(), &[2, 1]);
    assert_eq!(min_seq.shape(), &[2, 1]);

    if coeus_cuda::CudaDriver::get().is_some() && coeus_cuda::get_cuda_context().is_some() {
        // Expected values:
        // (a * 2.0).relu() on CPU:
        // [2.0, 0.0, 6.0]
        // [20.0, 0.0, 12.0]
        // Sums: [8.0, 32.0]
        // Maxs: [6.0, 20.0]
        // Mins: [0.0, 0.0]
        assert_eq!(sum_seq.as_slice(), &[8.0, 32.0]);
        assert_eq!(max_seq.as_slice(), &[6.0, 20.0]);
        assert_eq!(min_seq.as_slice(), &[0.0, 0.0]);
    }
}





