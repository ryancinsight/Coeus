//! Scaled dot-product attention, multi-head attention, and the
//! sinusoidal encoding, plus the fixtures the encoder cases share.

pub(super) use coeus_autograd::Var;
pub(super) use coeus_core::{MoiraiBackend, Storage};
pub(super) use coeus_nn::{
    feed_forward, transformer_encoder_layer, FeedForward, MhaProjectionParams, Module,
    MultiHeadAttention, NullMask, SinusoidalEncoding, TransformerEncoderLayer,
    TransformerEncoderLayerParams,
};
pub(super) use coeus_tensor::Tensor;

pub(super) type B = MoiraiBackend;

/// Every parameter carries a finite, correctly shaped gradient, and the
/// set as a whole received signal.
///
/// `grad.is_some()` only proves the accumulator was allocated. Backward
/// allocates before it writes, so a pass that never reached these
/// parameters leaves every buffer at its zero initialization and still
/// satisfies `is_some` — the failure this assertion exists to catch. A
/// single parameter may legitimately end at zero, so the non-zero claim
/// is made over the set.
pub(super) fn assert_parameters_received_signal(params: &[Var<f32, B>], module: &str) {
    let mut any_nonzero = false;
    for (i, p) in params.iter().enumerate() {
        let grad = p
            .grad()
            .unwrap_or_else(|| panic!("{module} parameter {i} has no gradient after backward"));
        assert_eq!(
            grad.shape(),
            p.tensor.shape(),
            "{module} parameter {i}: gradient shape must match the parameter"
        );
        for (j, g) in grad.as_slice().iter().enumerate() {
            assert!(
                g.is_finite(),
                "{module} parameter {i} element {j}: gradient is {g}"
            );
            any_nonzero |= *g != 0.0;
        }
    }
    assert!(
        any_nonzero,
        "{module}: backward left every parameter gradient at zero"
    );
}
pub(super) const EPS: f32 = 1e-5;

/// Rounding bound for the uniform-attention output with `eps = f32::EPSILON`:
/// the four equal exponentials sum with at most `eps` relative error, the
/// division adds `eps / 2` and the four-term weighted sum of rows adds
/// `3 * eps / 2`, so the relative error is at most `4 * eps` on a result of
/// 2.5, an absolute `10 * eps`; `16 * eps` covers it.
const SDPA_UNIFORM_TOLERANCE: f32 = 16.0 * f32::EPSILON;

/// Rounding bound for an `f32` sinusoidal table entry: the angle
/// `pos / 10000^(2i/d)` is at most 15 and carries at most `2 * eps` relative
/// error from `powf` and the division; entries with `i >= 1` have angle
/// at most 1.5, and `i = 0` is exact, so the absolute error of `sin`/`cos`
/// stays under `1.5 * 2 * eps + eps = 4 * eps`; `8 * eps` covers it.
const SINUSOIDAL_TOLERANCE: f32 = 8.0 * f32::EPSILON;

// ── SDPA: forward shape ──────────────────────────────────────────────────

#[test]
fn sdpa_forward_shape_null_mask() {
    let backend = B::default();
    let batch = 2;
    let seq_q = 4;
    let seq_k = 4;
    let d_k = 8;
    let d_v = 8;

    let q = Tensor::<f32, B>::ones_on([batch, seq_q, d_k], &backend)
        .expect("invariant: test backend operation succeeds");
    let k = Tensor::<f32, B>::ones_on([batch, seq_k, d_k], &backend)
        .expect("invariant: test backend operation succeeds");
    // Key position `j` carries the constant row `j + 1`, so the output exposes
    // which keys the attention weights select and how they are weighted.
    let v_rows = [1.0_f32, 2.0, 3.0, 4.0];
    let v_data: Vec<f32> = (0..batch)
        .flat_map(|_| v_rows.iter().flat_map(|&row| [row; 8]))
        .collect();
    let v = Tensor::<f32, B>::from_slice_on([batch, seq_k, d_v], &v_data, &backend)
        .expect("invariant: test backend operation succeeds");

    let q_var = Var::new(q, true).expect("invariant: test backend operation succeeds");
    let k_var = Var::new(k, true).expect("invariant: test backend operation succeeds");
    let v_var = Var::new(v, true).expect("invariant: test backend operation succeeds");

    let scale = 1.0_f32 / (d_k as f32).sqrt();
    let (out, _aw) =
        coeus_autograd::sdp_attention::<f32, B, NullMask>(&q_var, &k_var, &v_var, None, scale)
            .expect("valid attention fixture");

    assert_eq!(out.tensor.shape(), &[batch, seq_q, d_v]);

    // All scores are equal, so every query attends to the four keys with
    // weight 1/4 and each output element is the mean of the rows: (1+2+3+4)/4.
    let out_data = out
        .tensor
        .storage()
        .try_as_slice()
        .expect("test: attention output must be CPU-addressable");
    assert_eq!(out_data.len(), batch * seq_q * d_v);
    for (index, &value) in out_data.iter().enumerate() {
        assert!(
            (value - 2.5).abs() <= SDPA_UNIFORM_TOLERANCE,
            "output[{index}] = {value}, expected 2.5"
        );
    }
}

// ── SDPA: causal mask ────────────────────────────────────────────────────

#[test]
fn sdpa_causal_mask_upper_triangle_zero() {
    // With all-ones Q and K, causal masking must zero-out attention to future positions.
    // Specifically, for row i, position j > i should have attn_weight = 0.
    let backend = B::default();
    let batch = 1;
    let seq = 4;
    let d = 4;

    let q = Tensor::<f32, B>::ones_on([batch, seq, d], &backend)
        .expect("invariant: test backend operation succeeds");
    let k = Tensor::<f32, B>::ones_on([batch, seq, d], &backend)
        .expect("invariant: test backend operation succeeds");
    let v = Tensor::<f32, B>::ones_on([batch, seq, d], &backend)
        .expect("invariant: test backend operation succeeds");

    let (attn_out, attn_weights) = coeus_ops::scaled_dot_product_attention(
        &q,
        &k,
        &v,
        None,
        true,
        1.0_f32 / (d as f32).sqrt(),
        &backend,
    )
    .expect("valid attention fixture");

    // attn_weights: [1, seq, seq]
    let aw_data = attn_weights
        .storage()
        .try_as_slice()
        .expect("test: attn_weights must be CPU-addressable");

    // For row i, all j > i must be (approximately) 0.
    for i in 0..seq {
        for j in (i + 1)..seq {
            let val = aw_data[i * seq + j]; // batch=0 so no offset
            assert!(
                val.abs() < EPS,
                "causal mask violation: attn_weights[{i},{j}] = {val} (expected ~0)"
            );
        }
    }

    // Output shape check
    assert_eq!(attn_out.shape(), &[batch, seq, d]);
}

// ── SDPA: gradient flow ──────────────────────────────────────────────────

#[test]
fn sdpa_gradient_flow_qkv() {
    let backend = B::default();
    let batch = 1;
    let seq = 3;
    let d = 4;

    // Non-uniform inputs: avoid uniform softmax cancellation.
    // With uniform Q/K/V, dQ = dK = 0 exactly (softmax row-sum == uniform).
    // Use distinct values so dQ and dK are non-zero.
    let q_data: Vec<f32> = (1..=(batch * seq * d)).map(|x| x as f32 * 0.1).collect();
    let k_data: Vec<f32> = (2..=(batch * seq * d + 1))
        .map(|x| x as f32 * 0.07)
        .collect();
    let v_data: Vec<f32> = (3..=(batch * seq * d + 2))
        .map(|x| x as f32 * 0.05)
        .collect();

    let q_t = Tensor::<f32, B>::from_slice_on([batch, seq, d], &q_data, &backend)
        .expect("invariant: test backend operation succeeds");
    let k_t = Tensor::<f32, B>::from_slice_on([batch, seq, d], &k_data, &backend)
        .expect("invariant: test backend operation succeeds");
    let v_t = Tensor::<f32, B>::from_slice_on([batch, seq, d], &v_data, &backend)
        .expect("invariant: test backend operation succeeds");

    let q = Var::new(q_t, true).expect("invariant: test backend operation succeeds");
    let k = Var::new(k_t, true).expect("invariant: test backend operation succeeds");
    let v = Var::new(v_t, true).expect("invariant: test backend operation succeeds");

    let scale = 1.0_f32 / (d as f32).sqrt();
    let (out, _) = coeus_autograd::sdp_attention::<f32, B, NullMask>(&q, &k, &v, None, scale)
        .expect("valid attention fixture");

    // Sum-reduce to scalar loss and backprop
    let loss = coeus_autograd::sum(&out).expect("invariant: test operation succeeds");
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");

    // All three inputs must have non-None gradients
    for (label, var) in [("q", &q), ("k", &k), ("v", &v)] {
        let grad = var
            .grad
            .as_ref()
            .unwrap_or_else(|| panic!("SDPA: {label}.grad is None"));
        let gdata = grad.read();
        let slice = gdata
            .storage()
            .try_as_slice()
            .expect("grad must be CPU-addressable");
        let any_nonzero = slice.iter().any(|&x: &f32| x.abs() > EPS);
        assert!(any_nonzero, "SDPA: {label}.grad is all-zero");
        let any_nan = slice.iter().any(|x: &f32| x.is_nan() || x.is_infinite());
        assert!(!any_nan, "SDPA: {label}.grad contains NaN/Inf");
    }
}

// ── MHA: output shape ────────────────────────────────────────────────────

#[test]
fn mha_output_shape() {
    const H: usize = 4;
    let d_model = 16;
    let batch = 2;
    let seq = 5;

    let mha = MultiHeadAttention::<f32, B, H, NullMask>::new(d_model, true)
        .expect("valid attention shape fixture");
    let backend = B::default();
    let x = Tensor::<f32, B>::ones_on([batch, seq, d_model], &backend)
        .expect("invariant: test backend operation succeeds");
    let x_var = Var::new(x, false).expect("invariant: test backend operation succeeds");

    let out = mha.forward(&x_var).expect("valid MultiHeadAttention input");
    assert_eq!(
        out.tensor.shape(),
        &[batch, seq, d_model],
        "MHA output shape mismatch"
    );
}

// ── MHA: gradient flow through parameters ───────────────────────────────

#[test]
fn mha_gradient_flow_params() {
    const H: usize = 2;
    let d_model = 8;
    let batch = 1;
    let seq = 3;

    let mha = MultiHeadAttention::<f32, B, H, NullMask>::new(d_model, true)
        .expect("valid attention gradient fixture");
    let backend = B::default();
    let x = Tensor::<f32, B>::ones_on([batch, seq, d_model], &backend)
        .expect("invariant: test backend operation succeeds");
    let x_var = Var::new(x, false).expect("invariant: test backend operation succeeds");

    let out = mha.forward(&x_var).expect("valid MultiHeadAttention input");
    let loss = coeus_autograd::sum(&out).expect("invariant: test operation succeeds");
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");

    let params = mha.parameters();
    assert!(!params.is_empty(), "MHA must have parameters");
    assert_parameters_received_signal(&params, "MHA");
}

// ── SinusoidalEncoding: shape and content ────────────────────────────────

#[test]
fn sinusoidal_encoding_shape_and_values() {
    let max_len = 16;
    let d_model = 8;
    let pe = SinusoidalEncoding::<f32, B>::new(max_len, d_model)
        .expect("invariant: test operation succeeds");

    assert_eq!(pe.table.shape(), &[max_len, d_model]);

    let data = pe
        .table
        .storage()
        .try_as_slice()
        .expect("PE table must be CPU-addressable");

    // Position 0 has angle 0 in every pair: sin = 0 and cos = 1, exactly.
    let row_zero: Vec<f32> = (0..d_model / 2).flat_map(|_| [0.0_f32, 1.0]).collect();
    assert_eq!(&data[..d_model], row_zero.as_slice());

    // PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(same angle)
    // (Vaswani et al. 2017, section 3.5).
    let as_f64 = |n: usize| f64::from(u16::try_from(n).expect("invariant: index fits u16"));
    for pos in 0..max_len {
        for i in 0..d_model / 2 {
            let angle = as_f64(pos) / 10_000_f64.powf(as_f64(2 * i) / as_f64(d_model));
            for (offset, want) in [(0, angle.sin()), (1, angle.cos())] {
                let got = f64::from(data[pos * d_model + 2 * i + offset]);
                assert!(
                    (got - want).abs() <= f64::from(SINUSOIDAL_TOLERANCE),
                    "table[{pos}][{}] = {got}, expected {want}",
                    2 * i + offset
                );
            }
        }
    }
}

#[test]
fn sinusoidal_encoding_forward_shape() {
    let max_len = 16;
    let d_model = 8;
    let pe = SinusoidalEncoding::<f32, B>::new(max_len, d_model)
        .expect("invariant: test operation succeeds");

    let backend = B::default();
    let batch = 2;
    let seq = 6;
    let x = Tensor::<f32, B>::zeros_on([batch, seq, d_model], &backend)
        .expect("invariant: test backend operation succeeds");
    let x_var = Var::new(x, false).expect("invariant: test backend operation succeeds");

    let out = pe.forward(&x_var).expect("valid SinusoidalEncoding input");
    assert_eq!(out.tensor.shape(), &[batch, seq, d_model]);
}
