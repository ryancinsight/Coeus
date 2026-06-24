// CudaBackend SDP attention parity differential tests.
//
// The unmasked/causal cases exercise the on-device NVRTC kernels
// (`kernels::attention`); the masked case exercises the CPU-reference capability
// boundary. Every case asserts element-wise agreement with `SequentialBackend`
// within a derived tolerance. Skipped unless a live CUDA device/context exists.

use coeus_core::SequentialBackend;
use coeus_cuda::CudaBackend;
use coeus_ops::{scaled_dot_product_attention, scaled_dot_product_attention_backward};
use coeus_tensor::Tensor;

const BATCH: usize = 2;
const SEQ_Q: usize = 3;
const SEQ_K: usize = 4;
const D_K: usize = 5;
const D_V: usize = 3;

/// Accumulating-op tolerance: softmax (libm `expf` vs device `expf`) plus f32
/// dot-product roundoff over `d_k`/`seq_k`. The fma-based device dots match the
/// CPU `dot_slice` order, so the dominant term is the ~1e-6 relative `expf`
/// difference; 1e-3 is the reduction-order-sensitive bound shared with conv.
const TOL: f32 = 1e-3;

fn backends() -> Option<(SequentialBackend, CudaBackend)> {
    if hephaestus_cuda::CudaDevice::try_default().is_err() {
        return None;
    }
    if coeus_cuda::CudaDriver::get().is_none() || coeus_cuda::get_cuda_context().is_none() {
        return None;
    }
    Some((SequentialBackend::new(), CudaBackend::new()))
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let tol = TOL * (1.0 + want.abs());
        assert!(
            (got - want).abs() <= tol,
            "{label}[{index}]: got {got}, expected {want}, tol {tol}",
        );
    }
}

fn inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
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

fn run_forward_case(is_causal: bool, label: &str) {
    let Some((seq, cuda)) = backends() else {
        return;
    };
    let (q, k, v, _) = inputs();
    let scale = 0.5f32;

    let q_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_Q, D_K], &q);
    let k_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_K, D_K], &k);
    let v_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_K, D_V], &v);

    let q_g = q_cpu.to_backend_on(&seq, &cuda);
    let k_g = k_cpu.to_backend_on(&seq, &cuda);
    let v_g = v_cpu.to_backend_on(&seq, &cuda);

    let (out_cpu, aw_cpu) =
        scaled_dot_product_attention(&q_cpu, &k_cpu, &v_cpu, None, is_causal, scale, &seq);
    let (out_g, aw_g) =
        scaled_dot_product_attention(&q_g, &k_g, &v_g, None, is_causal, scale, &cuda);

    let out_g = out_g.to_backend_on(&cuda, &seq);
    let aw_g = aw_g.to_backend_on(&cuda, &seq);

    assert_eq!(out_g.shape(), out_cpu.shape());
    assert_eq!(aw_g.shape(), aw_cpu.shape());
    assert_close(
        &format!("{label}_out"),
        out_g.as_slice(),
        out_cpu.as_slice(),
    );
    assert_close(
        &format!("{label}_weights"),
        aw_g.as_slice(),
        aw_cpu.as_slice(),
    );
}

#[test]
fn test_cuda_attention_forward_nomask() {
    run_forward_case(false, "attn_fwd_nomask");
}

#[test]
fn test_cuda_attention_forward_causal() {
    run_forward_case(true, "attn_fwd_causal");
}

#[test]
fn test_cuda_attention_forward_masked_uses_cpu_path() {
    // key_padding_mask present routes to the CPU reference path; must still match.
    let Some((seq, cuda)) = backends() else {
        return;
    };
    let (q, k, v, _) = inputs();
    let mask = vec![1.0f32, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let scale = 0.5f32;

    let q_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_Q, D_K], &q);
    let k_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_K, D_K], &k);
    let v_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_K, D_V], &v);
    let m_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_K], &mask);

    let q_g = q_cpu.to_backend_on(&seq, &cuda);
    let k_g = k_cpu.to_backend_on(&seq, &cuda);
    let v_g = v_cpu.to_backend_on(&seq, &cuda);
    let m_g = m_cpu.to_backend_on(&seq, &cuda);

    let (out_cpu, aw_cpu) =
        scaled_dot_product_attention(&q_cpu, &k_cpu, &v_cpu, Some(&m_cpu), false, scale, &seq);
    let (out_g, aw_g) =
        scaled_dot_product_attention(&q_g, &k_g, &v_g, Some(&m_g), false, scale, &cuda);

    let out_g = out_g.to_backend_on(&cuda, &seq);
    let aw_g = aw_g.to_backend_on(&cuda, &seq);
    assert_close("attn_masked_out", out_g.as_slice(), out_cpu.as_slice());
    assert_close("attn_masked_weights", aw_g.as_slice(), aw_cpu.as_slice());
}

#[test]
fn test_cuda_attention_backward() {
    let Some((seq, cuda)) = backends() else {
        return;
    };
    let (q, k, v, go) = inputs();
    let scale = 0.25f32;

    let q_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_Q, D_K], &q);
    let k_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_K, D_K], &k);
    let v_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_K, D_V], &v);
    let go_cpu = Tensor::<f32, SequentialBackend>::from_slice([BATCH, SEQ_Q, D_V], &go);

    let q_g = q_cpu.to_backend_on(&seq, &cuda);
    let k_g = k_cpu.to_backend_on(&seq, &cuda);
    let v_g = v_cpu.to_backend_on(&seq, &cuda);
    let go_g = go_cpu.to_backend_on(&seq, &cuda);

    // Stored attention weights from the forward pass feed the backward.
    let (_, aw_cpu) =
        scaled_dot_product_attention(&q_cpu, &k_cpu, &v_cpu, None, false, scale, &seq);
    let (_, aw_g) = scaled_dot_product_attention(&q_g, &k_g, &v_g, None, false, scale, &cuda);

    let mut gq_cpu = Tensor::<f32, SequentialBackend>::zeros_on([BATCH, SEQ_Q, D_K], &seq);
    let mut gk_cpu = Tensor::<f32, SequentialBackend>::zeros_on([BATCH, SEQ_K, D_K], &seq);
    let mut gv_cpu = Tensor::<f32, SequentialBackend>::zeros_on([BATCH, SEQ_K, D_V], &seq);
    scaled_dot_product_attention_backward(
        &go_cpu,
        &q_cpu,
        &k_cpu,
        &v_cpu,
        &aw_cpu,
        scale,
        Some(&mut gq_cpu),
        Some(&mut gk_cpu),
        Some(&mut gv_cpu),
        &seq,
    );

    let mut gq_g = Tensor::<f32, CudaBackend>::zeros_on([BATCH, SEQ_Q, D_K], &cuda);
    let mut gk_g = Tensor::<f32, CudaBackend>::zeros_on([BATCH, SEQ_K, D_K], &cuda);
    let mut gv_g = Tensor::<f32, CudaBackend>::zeros_on([BATCH, SEQ_K, D_V], &cuda);
    scaled_dot_product_attention_backward(
        &go_g,
        &q_g,
        &k_g,
        &v_g,
        &aw_g,
        scale,
        Some(&mut gq_g),
        Some(&mut gk_g),
        Some(&mut gv_g),
        &cuda,
    );

    assert_close(
        "attn_bwd_grad_q",
        gq_g.to_backend_on(&cuda, &seq).as_slice(),
        gq_cpu.as_slice(),
    );
    assert_close(
        "attn_bwd_grad_k",
        gk_g.to_backend_on(&cuda, &seq).as_slice(),
        gk_cpu.as_slice(),
    );
    assert_close(
        "attn_bwd_grad_v",
        gv_g.to_backend_on(&cuda, &seq).as_slice(),
        gv_cpu.as_slice(),
    );
}
