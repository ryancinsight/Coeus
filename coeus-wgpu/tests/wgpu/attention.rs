use coeus_core::SequentialBackend;
use coeus_ops::{scaled_dot_product_attention, scaled_dot_product_attention_backward};
use coeus_tensor::Tensor;
use coeus_wgpu::WgpuBackend;

const BATCH: usize = 1;
const SEQ_Q: usize = 3;
const SEQ_K: usize = 4;
const D_K: usize = 5;
const D_V: usize = 3;

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        // WGPU attention currently routes through the CPU reference path and
        // performs device transfers around it. The tolerance covers f32
        // roundoff in the public attention equations and backend transfer.
        let tol = 512.0 * f32::EPSILON * (1.0 + want.abs());
        assert!(
            (got - want).abs() <= tol,
            "{label}[{index}]: got {got}, expected {want}, tol {tol}",
        );
    }
}

fn attention_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let query = (0..BATCH * SEQ_Q * D_K)
        .map(|i| ((i as f32 + 1.0) * 0.125).sin())
        .collect();
    let key = (0..BATCH * SEQ_K * D_K)
        .map(|i| ((i as f32 + 3.0) * 0.09375).cos())
        .collect();
    let value = (0..BATCH * SEQ_K * D_V)
        .map(|i| (i as f32 % 7.0 - 3.0) * 0.25)
        .collect();
    let grad_out = (0..BATCH * SEQ_Q * D_V)
        .map(|i| (i as f32 % 5.0 - 2.0) * 0.375)
        .collect();

    (query, key, value, grad_out)
}

/// On-device forward tolerance. Unmasked/causal forward runs the WGSL softmax
/// (device `exp` vs libm `expf`, ~1e-6 relative) and accumulates over `seq_k`;
/// the bound covers that plus f32 roundoff. Looser than the CPU-path transfer
/// tolerance because `exp` is now evaluated on-device, not copied from the CPU.
fn assert_close_device(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let tol = 1e-4 * (1.0 + want.abs());
        assert!(
            (got - want).abs() <= tol,
            "{label}[{index}]: got {got}, expected {want}, tol {tol}",
        );
    }
}

fn run_device_forward(is_causal: bool, label: &str) {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let (query_data, key_data, value_data, _) = attention_inputs();
    let scale = 0.5f32;

    let query_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_Q, D_K], &query_data);
    let key_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_K, D_K], &key_data);
    let value_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_K, D_V], &value_data);

    let query_gpu = query_cpu.to_backend_on(&seq, &wgpu);
    let key_gpu = key_cpu.to_backend_on(&seq, &wgpu);
    let value_gpu = value_cpu.to_backend_on(&seq, &wgpu);

    // No mask -> on-device WGSL kernel.
    let (expected_out, expected_weights) = scaled_dot_product_attention(
        &query_cpu, &key_cpu, &value_cpu, None, is_causal, scale, &seq,
    );
    let (actual_out, actual_weights) = scaled_dot_product_attention(
        &query_gpu, &key_gpu, &value_gpu, None, is_causal, scale, &wgpu,
    );

    let actual_out = actual_out.to_backend_on(&wgpu, &seq);
    let actual_weights = actual_weights.to_backend_on(&wgpu, &seq);

    assert_eq!(actual_out.shape(), expected_out.shape());
    assert_eq!(actual_weights.shape(), expected_weights.shape());
    assert_close_device(
        &format!("{label}_out"),
        actual_out.as_slice(),
        expected_out.as_slice(),
    );
    assert_close_device(
        &format!("{label}_weights"),
        actual_weights.as_slice(),
        expected_weights.as_slice(),
    );
}

#[test]
fn wgpu_attention_forward_unmasked_matches_cpu_on_device() {
    run_device_forward(false, "attn_fwd_unmasked");
}

#[test]
fn wgpu_attention_forward_causal_matches_cpu_on_device() {
    run_device_forward(true, "attn_fwd_causal");
}

#[test]
fn wgpu_attention_forward_matches_cpu_with_mask_and_causal() {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let (query_data, key_data, value_data, _) = attention_inputs();
    let mask_data = vec![1.0f32, 1.0, 0.0, 1.0];
    let scale = 0.5f32;

    let query_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_Q, D_K], &query_data);
    let key_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_K, D_K], &key_data);
    let value_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_K, D_V], &value_data);
    let mask_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_K], &mask_data);

    let query_gpu = query_cpu.to_backend_on(&seq, &wgpu);
    let key_gpu = key_cpu.to_backend_on(&seq, &wgpu);
    let value_gpu = value_cpu.to_backend_on(&seq, &wgpu);
    let mask_gpu = mask_cpu.to_backend_on(&seq, &wgpu);

    let (expected_out, expected_weights) = scaled_dot_product_attention(
        &query_cpu,
        &key_cpu,
        &value_cpu,
        Some(&mask_cpu),
        true,
        scale,
        &seq,
    );
    let (actual_out, actual_weights) = scaled_dot_product_attention(
        &query_gpu,
        &key_gpu,
        &value_gpu,
        Some(&mask_gpu),
        true,
        scale,
        &wgpu,
    );

    let actual_out = actual_out.to_backend_on(&wgpu, &seq);
    let actual_weights = actual_weights.to_backend_on(&wgpu, &seq);

    assert_eq!(actual_out.shape(), expected_out.shape());
    assert_eq!(actual_weights.shape(), expected_weights.shape());
    assert_close(
        "attention output",
        actual_out.as_slice(),
        expected_out.as_slice(),
    );
    assert_close(
        "attention weights",
        actual_weights.as_slice(),
        expected_weights.as_slice(),
    );
}

#[test]
fn wgpu_attention_backward_matches_cpu() {
    let seq = SequentialBackend::new();
    let wgpu = WgpuBackend::new();
    let (query_data, key_data, value_data, grad_out_data) = attention_inputs();
    let scale = 0.25f32;

    let query_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_Q, D_K], &query_data);
    let key_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_K, D_K], &key_data);
    let value_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_K, D_V], &value_data);
    let grad_out_cpu =
        Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_Q, D_V], &grad_out_data);

    let query_gpu = query_cpu.to_backend_on(&seq, &wgpu);
    let key_gpu = key_cpu.to_backend_on(&seq, &wgpu);
    let value_gpu = value_cpu.to_backend_on(&seq, &wgpu);
    let grad_out_gpu = grad_out_cpu.to_backend_on(&seq, &wgpu);

    let (_, weights_cpu) =
        scaled_dot_product_attention(&query_cpu, &key_cpu, &value_cpu, None, false, scale, &seq);
    let (_, weights_gpu) =
        scaled_dot_product_attention(&query_gpu, &key_gpu, &value_gpu, None, false, scale, &wgpu);

    let mut expected_q = Tensor::<f32, SequentialBackend>::zeros_on([BATCH, SEQ_Q, D_K], &seq);
    let mut expected_k = Tensor::<f32, SequentialBackend>::zeros_on([BATCH, SEQ_K, D_K], &seq);
    let mut expected_v = Tensor::<f32, SequentialBackend>::zeros_on([BATCH, SEQ_K, D_V], &seq);
    scaled_dot_product_attention_backward(
        &grad_out_cpu,
        &query_cpu,
        &key_cpu,
        &value_cpu,
        &weights_cpu,
        scale,
        Some(&mut expected_q),
        Some(&mut expected_k),
        Some(&mut expected_v),
        &seq,
    );

    let mut actual_q = Tensor::<f32, WgpuBackend>::zeros_on([BATCH, SEQ_Q, D_K], &wgpu);
    let mut actual_k = Tensor::<f32, WgpuBackend>::zeros_on([BATCH, SEQ_K, D_K], &wgpu);
    let mut actual_v = Tensor::<f32, WgpuBackend>::zeros_on([BATCH, SEQ_K, D_V], &wgpu);
    scaled_dot_product_attention_backward(
        &grad_out_gpu,
        &query_gpu,
        &key_gpu,
        &value_gpu,
        &weights_gpu,
        scale,
        Some(&mut actual_q),
        Some(&mut actual_k),
        Some(&mut actual_v),
        &wgpu,
    );

    let actual_q = actual_q.to_backend_on(&wgpu, &seq);
    let actual_k = actual_k.to_backend_on(&wgpu, &seq);
    let actual_v = actual_v.to_backend_on(&wgpu, &seq);

    assert_close("grad_q", actual_q.as_slice(), expected_q.as_slice());
    assert_close("grad_k", actual_k.as_slice(), expected_k.as_slice());
    assert_close("grad_v", actual_v.as_slice(), expected_v.as_slice());
}
