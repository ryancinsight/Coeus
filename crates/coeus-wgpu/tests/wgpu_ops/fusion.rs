use coeus_core::SequentialBackend;
use coeus_ops::fuse::{evaluate_fused_cpu, TensorExprExt};
use coeus_tensor::Tensor;
use coeus_wgpu::{evaluate_fused, WgpuBackend};

#[test]
fn test_wgpu_fusion_parity() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let shape = vec![2, 3];
    let a_data = vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0];
    let c_data = vec![-5.0f32, 5.0, -10.0, 10.0, -15.0, 15.0];

    // Create tensors on CPU
    let a_cpu = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &a_data);
    let b_cpu = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &b_data);
    let c_cpu = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &c_data);

    // Transfer to GPU
    let a_gpu = a_cpu.to_backend_on(&seq, &wgpu_b);
    let b_gpu = b_cpu.to_backend_on(&seq, &wgpu_b);
    let c_gpu = c_cpu.to_backend_on(&seq, &wgpu_b);

    // Fused expression on GPU: (a_gpu * b_gpu + c_gpu).relu().sigmoid()
    let expr = (a_gpu.expr() * b_gpu.expr() + c_gpu.expr())
        .relu()
        .sigmoid();
    let out_gpu = evaluate_fused(&expr);

    // Transfer back to CPU
    let out_cpu = out_gpu.to_backend_on(&wgpu_b, &seq);

    // Fused expression on CPU
    let expr_cpu = (a_cpu.expr() * b_cpu.expr() + c_cpu.expr())
        .relu()
        .sigmoid();
    let expected_cpu = evaluate_fused_cpu(&expr_cpu, &seq);

    // Compare
    let out_slice = out_cpu.as_slice();
    let exp_slice = expected_cpu.as_slice();
    for i in 0..out_slice.len() {
        let diff = (out_slice[i] - exp_slice[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: {} vs expected {}",
            i,
            out_slice[i],
            exp_slice[i]
        );
    }
}

#[test]
fn test_wgpu_evaluate_fused_reduce() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let shape = vec![2, 3];
    let a_data = vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0];

    let a_cpu = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &a_data);
    let a_gpu = a_cpu.to_backend_on(&seq, &wgpu_b);

    // Fused expression: (a * 2.0).relu()
    let expr_gpu = (a_gpu.expr() * 2.0).relu();
    let expr_cpu = (a_cpu.expr() * 2.0).relu();
    let evaluated_cpu = evaluate_fused_cpu(&expr_cpu, &seq);

    // Fused sum reduction along axis 1
    let out_sum_gpu = coeus_wgpu::evaluate_fused_reduce(&expr_gpu, coeus_ops::ReductionOp::Sum, 1);
    let out_sum_cpu = out_sum_gpu.to_backend_on(&wgpu_b, &seq);
    let expected_sum =
        coeus_ops::fuse::evaluate_fused_reduce_cpu(&expr_cpu, coeus_ops::ReductionOp::Sum, 1, &seq);

    assert_eq!(out_sum_cpu.as_slice(), expected_sum.as_slice());

    // Fused mean reduction along axis 1
    let out_mean_gpu =
        coeus_wgpu::evaluate_fused_reduce(&expr_gpu, coeus_ops::ReductionOp::Mean, 1);
    let out_mean_cpu = out_mean_gpu.to_backend_on(&wgpu_b, &seq);
    let expected_mean = coeus_ops::fuse::evaluate_fused_reduce_cpu(
        &expr_cpu,
        coeus_ops::ReductionOp::Mean,
        1,
        &seq,
    );

    let axis_len = shape[1] as f32;
    let eps = f32::EPSILON;
    let gamma = (axis_len * eps) / (1.0 - axis_len * eps);
    for (index, (&actual, &expected)) in out_mean_cpu
        .as_slice()
        .iter()
        .zip(expected_mean.as_slice())
        .enumerate()
    {
        let row_start = index * shape[1];
        let row_magnitude: f32 = evaluated_cpu.as_slice()[row_start..row_start + shape[1]]
            .iter()
            .map(|value| value.abs())
            .sum();
        // Forward error for summing n terms, gamma_n = n*eps/(1-n*eps),
        // then one rounded division by n.
        let tolerance = (gamma * row_magnitude / axis_len) + eps * expected.abs().max(1.0);
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "mean mismatch at index {index}: got {actual}, expected {expected}, diff {diff}, tolerance {tolerance}",
        );
    }

    // Fused max reduction along axis 1
    let out_max_gpu = coeus_wgpu::evaluate_fused_reduce(&expr_gpu, coeus_ops::ReductionOp::Max, 1);
    let out_max_cpu = out_max_gpu.to_backend_on(&wgpu_b, &seq);
    let expected_max =
        coeus_ops::fuse::evaluate_fused_reduce_cpu(&expr_cpu, coeus_ops::ReductionOp::Max, 1, &seq);

    assert_eq!(out_max_cpu.as_slice(), expected_max.as_slice());

    // Fused min reduction along axis 1
    let out_min_gpu = coeus_wgpu::evaluate_fused_reduce(&expr_gpu, coeus_ops::ReductionOp::Min, 1);
    let out_min_cpu = out_min_gpu.to_backend_on(&wgpu_b, &seq);
    let expected_min =
        coeus_ops::fuse::evaluate_fused_reduce_cpu(&expr_cpu, coeus_ops::ReductionOp::Min, 1, &seq);

    assert_eq!(out_min_cpu.as_slice(), expected_min.as_slice());
}

#[test]
fn test_wgpu_fusion_silu() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let shape = vec![5];
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];

    let a_cpu = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &data);
    let a_gpu = a_cpu.to_backend_on(&seq, &wgpu_b);

    // Fused expression: a.silu()
    let expr_gpu = a_gpu.expr().silu();
    let out_gpu = evaluate_fused(&expr_gpu);

    // Transfer back to CPU
    let out_cpu = out_gpu.to_backend_on(&wgpu_b, &seq);

    // Compute expected using CPU fusion
    let expr_cpu = a_cpu.expr().silu();
    let expected = evaluate_fused_cpu(&expr_cpu, &seq);

    // Compare
    let out_slice = out_cpu.as_slice();
    let exp_slice = expected.as_slice();
    for i in 0..out_slice.len() {
        let diff = (out_slice[i] - exp_slice[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: {} vs {}",
            i,
            out_slice[i],
            exp_slice[i]
        );
    }
}

#[test]
fn test_wgpu_fusion_mish() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let shape = vec![5];
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];

    let a_cpu = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &data);
    let a_gpu = a_cpu.to_backend_on(&seq, &wgpu_b);

    // Fused expression: a.mish()
    let expr_gpu = a_gpu.expr().mish();
    let out_gpu = evaluate_fused(&expr_gpu);

    // Transfer back to CPU
    let out_cpu = out_gpu.to_backend_on(&wgpu_b, &seq);

    // Compute expected using CPU fusion
    let expr_cpu = a_cpu.expr().mish();
    let expected = evaluate_fused_cpu(&expr_cpu, &seq);

    // Compare
    let out_slice = out_cpu.as_slice();
    let exp_slice = expected.as_slice();
    for i in 0..out_slice.len() {
        let diff = (out_slice[i] - exp_slice[i]).abs();
        assert!(
            diff < 1e-5,
            "Mismatch at index {}: {} vs {}",
            i,
            out_slice[i],
            exp_slice[i]
        );
    }
}

#[test]
fn test_wgpu_fusion_gelu() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let shape = vec![5];
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];

    let a_cpu = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &data);
    let a_gpu = a_cpu.to_backend_on(&seq, &wgpu_b);

    // Fused expression: a.gelu()
    let expr_gpu = a_gpu.expr().gelu();
    let out_gpu = evaluate_fused(&expr_gpu);

    // Transfer back to CPU
    let out_cpu = out_gpu.to_backend_on(&wgpu_b, &seq);

    // Compute expected using CPU fusion
    let expr_cpu = a_cpu.expr().gelu();
    let expected = evaluate_fused_cpu(&expr_cpu, &seq);

    // Compare — WGPU WGSL uses the tanh approximation while CPU fused uses the
    // exact erf formula; allow up to 5e-3 difference (max observed ~0.003 at x=±1).
    let out_slice = out_cpu.as_slice();
    let exp_slice = expected.as_slice();
    for i in 0..out_slice.len() {
        let diff = (out_slice[i] - exp_slice[i]).abs();
        assert!(
            diff < 5e-3,
            "Mismatch at index {}: gpu_tanh={} cpu_erf={} diff={}",
            i,
            out_slice[i],
            exp_slice[i],
            diff,
        );
    }
}

#[test]
fn test_wgpu_fusion_gelu_grad() {
    let seq = SequentialBackend::new();
    let wgpu_b = WgpuBackend::new();

    let shape = vec![5];
    let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];

    let a_cpu = Tensor::<f32, SequentialBackend>::from_slice(shape.clone(), &data);
    let a_gpu = a_cpu.to_backend_on(&seq, &wgpu_b);

    // Fused expression: a.gelu_grad()
    let expr_gpu = a_gpu.expr().gelu_grad();
    let out_gpu = evaluate_fused(&expr_gpu);

    // Transfer back to CPU
    let out_cpu = out_gpu.to_backend_on(&wgpu_b, &seq);

    // Compute expected using CPU fusion
    let expr_cpu = a_cpu.expr().gelu_grad();
    let expected = evaluate_fused_cpu(&expr_cpu, &seq);

    // Compare
    let out_slice = out_cpu.as_slice();
    let exp_slice = expected.as_slice();
    for i in 0..out_slice.len() {
        let diff = (out_slice[i] - exp_slice[i]).abs();
        assert!(
            diff < 1e-4,
            "Mismatch at index {}: {} vs {}",
            i,
            out_slice[i],
            exp_slice[i]
        );
    }
}
