// ── Attention and Transformer architecture tests ──
//
// Validates:
//   1. SDPA forward shape
//   2. Causal mask correctness (upper-triangle → zero attention)
//   3. Gradient flow: q.grad, k.grad, v.grad non-None and finite
//   4. MHA output shape
//   5. TransformerEncoderLayer output shape and non-zero gradients
//   6. SinusoidalEncoding shape and non-zero values

#[cfg(test)]
mod tests {
    use coeus_autograd::Var;
    use coeus_core::{MoiraiBackend, Storage};
    use coeus_nn::{
        FeedForward, Module, MultiHeadAttention, NullMask, SinusoidalEncoding,
        TransformerEncoderLayer,
    };
    use coeus_tensor::Tensor;

    type B = MoiraiBackend;
    const EPS: f32 = 1e-5;

    // ── SDPA: forward shape ──────────────────────────────────────────────────

    #[test]
    fn sdpa_forward_shape_null_mask() {
        let backend = B::default();
        let batch = 2;
        let seq_q = 4;
        let seq_k = 4;
        let d_k = 8;
        let d_v = 8;

        let q = Tensor::<f32, B>::ones_on([batch, seq_q, d_k], &backend);
        let k = Tensor::<f32, B>::ones_on([batch, seq_k, d_k], &backend);
        let v = Tensor::<f32, B>::ones_on([batch, seq_k, d_v], &backend);

        let q_var = Var::new(q, true);
        let k_var = Var::new(k, true);
        let v_var = Var::new(v, true);

        let scale = 1.0_f32 / (d_k as f32).sqrt();
        let (out, _aw) =
            coeus_autograd::sdp_attention::<f32, B, NullMask>(&q_var, &k_var, &v_var, None, scale);

        assert_eq!(out.tensor.shape(), &[batch, seq_q, d_v]);
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

        let q = Tensor::<f32, B>::ones_on([batch, seq, d], &backend);
        let k = Tensor::<f32, B>::ones_on([batch, seq, d], &backend);
        let v = Tensor::<f32, B>::ones_on([batch, seq, d], &backend);

        let (attn_out, attn_weights) = coeus_ops::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            None,
            true,
            1.0_f32 / (d as f32).sqrt(),
            &backend,
        );

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

        let q_t = Tensor::<f32, B>::from_slice_on([batch, seq, d], &q_data, &backend);
        let k_t = Tensor::<f32, B>::from_slice_on([batch, seq, d], &k_data, &backend);
        let v_t = Tensor::<f32, B>::from_slice_on([batch, seq, d], &v_data, &backend);

        let q = Var::new(q_t, true);
        let k = Var::new(k_t, true);
        let v = Var::new(v_t, true);

        let scale = 1.0_f32 / (d as f32).sqrt();
        let (out, _) = coeus_autograd::sdp_attention::<f32, B, NullMask>(&q, &k, &v, None, scale);

        // Sum-reduce to scalar loss and backprop
        let loss = coeus_autograd::sum(&out);
        loss.backward();

        // All three inputs must have non-None gradients
        for (label, var) in [("q", &q), ("k", &k), ("v", &v)] {
            let grad = var
                .grad
                .as_ref()
                .unwrap_or_else(|| panic!("SDPA: {label}.grad is None"));
            let gdata = grad.lock().unwrap();
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

        let mha = MultiHeadAttention::<f32, B, H, NullMask>::new(d_model, true);
        let backend = B::default();
        let x = Tensor::<f32, B>::ones_on([batch, seq, d_model], &backend);
        let x_var = Var::new(x, false);

        let out = mha.forward(&x_var);
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

        let mha = MultiHeadAttention::<f32, B, H, NullMask>::new(d_model, true);
        let backend = B::default();
        let x = Tensor::<f32, B>::ones_on([batch, seq, d_model], &backend);
        let x_var = Var::new(x, false);

        let out = mha.forward(&x_var);
        let loss = coeus_autograd::sum(&out);
        loss.backward();

        let params = mha.parameters();
        assert!(!params.is_empty(), "MHA must have parameters");
        for (i, p) in params.iter().enumerate() {
            assert!(
                p.grad.is_some(),
                "MHA parameter {i} has no gradient after backward"
            );
        }
    }

    // ── SinusoidalEncoding: shape and content ────────────────────────────────

    #[test]
    fn sinusoidal_encoding_shape_and_values() {
        let max_len = 16;
        let d_model = 8;
        let pe = SinusoidalEncoding::<f32, B>::new(max_len, d_model);

        assert_eq!(pe.table.shape(), &[max_len, d_model]);

        // PE table must contain non-zero values (sin/cos are not all zero)
        let data = pe
            .table
            .storage()
            .try_as_slice()
            .expect("PE table must be CPU-addressable");
        let any_nonzero = data.iter().any(|&x: &f32| x.abs() > EPS);
        assert!(any_nonzero, "SinusoidalEncoding: table is all-zero");
    }

    #[test]
    fn sinusoidal_encoding_forward_shape() {
        let max_len = 16;
        let d_model = 8;
        let pe = SinusoidalEncoding::<f32, B>::new(max_len, d_model);

        let backend = B::default();
        let batch = 2;
        let seq = 6;
        let x = Tensor::<f32, B>::zeros_on([batch, seq, d_model], &backend);
        let x_var = Var::new(x, false);

        let out = pe.forward(&x_var);
        assert_eq!(out.tensor.shape(), &[batch, seq, d_model]);
    }

    // ── FeedForward: shape ───────────────────────────────────────────────────

    #[test]
    fn ffn_forward_shape() {
        let d_model = 16;
        let d_ff = 64;
        let ffn = FeedForward::<f32, B>::new(d_model, d_ff, 0.0);

        let backend = B::default();
        let batch = 2;
        let seq = 5;
        let x = Tensor::<f32, B>::ones_on([batch, seq, d_model], &backend);
        let x_var = Var::new(x, false);

        let out = ffn.forward(&x_var);
        assert_eq!(out.tensor.shape(), &[batch, seq, d_model]);
    }

    // ── TransformerEncoderLayer: shape and gradient ──────────────────────────

    #[test]
    fn encoder_layer_forward_shape() {
        const H: usize = 2;
        let d_model = 8;
        let d_ff = 32;

        let layer = TransformerEncoderLayer::<f32, B, H, NullMask>::new(d_model, d_ff, 0.0);
        let backend = B::default();
        let batch = 1;
        let seq = 4;
        let x = Tensor::<f32, B>::ones_on([batch, seq, d_model], &backend);
        let x_var = Var::new(x, false);

        let out = layer.forward(&x_var);
        assert_eq!(
            out.tensor.shape(),
            &[batch, seq, d_model],
            "EncoderLayer output shape mismatch"
        );
    }

    #[test]
    fn encoder_layer_gradient_through_all_params() {
        const H: usize = 2;
        let d_model = 8;
        let d_ff = 32;

        let layer = TransformerEncoderLayer::<f32, B, H, NullMask>::new(d_model, d_ff, 0.0);
        let backend = B::default();
        let batch = 1;
        let seq = 4;
        let x = Tensor::<f32, B>::ones_on([batch, seq, d_model], &backend);
        let x_var = Var::new(x, true);

        let out = layer.forward(&x_var);
        let loss = coeus_autograd::sum(&out);
        loss.backward();

        let params = layer.parameters();
        for (i, p) in params.iter().enumerate() {
            assert!(
                p.grad.is_some(),
                "EncoderLayer parameter {i} has no gradient"
            );
        }
    }

    #[test]
    fn test_mha_key_padding_mask() {
        const H: usize = 2;
        let d_model = 8;
        let batch = 1;
        let seq_q = 3;
        let seq_k = 4;
        let backend = B::default();

        let q = Tensor::<f32, B>::ones_on([batch * H, seq_q, d_model / H], &backend);
        let k = Tensor::<f32, B>::ones_on([batch * H, seq_k, d_model / H], &backend);
        let v = Tensor::<f32, B>::ones_on([batch * H, seq_k, d_model / H], &backend);

        let q_var = Var::new(q, true);
        let k_var = Var::new(k, true);
        let v_var = Var::new(v, true);

        // Mask out the last two key/value elements (indices 2 and 3)
        // 1.0 means keep, 0.0 means pad
        let mask_data = vec![1.0_f32, 1.0_f32, 0.0_f32, 0.0_f32];
        let mask = Tensor::<f32, B>::from_slice_on([batch, seq_k], &mask_data, &backend);
        let mask_var = Var::new(mask, false);

        let scale = 1.0_f32;
        let (out, aw) = coeus_autograd::sdp_attention::<f32, B, NullMask>(
            &q_var,
            &k_var,
            &v_var,
            Some(&mask_var),
            scale,
        );

        let aw_data = aw.storage().try_as_slice().unwrap();
        println!("aw_data: {:?}", aw_data);

        let loss = coeus_autograd::sum(&out);
        loss.backward();

        let k_grad = k_var.grad.as_ref().unwrap().lock().unwrap();
        let k_grad_slice = k_grad.storage().try_as_slice().unwrap();
        println!("k_grad_slice: {:?}", k_grad_slice);

        // Check that gradients for key and value at indices 2 and 3 (the padded elements) are zero
        // k has shape [2, 4, 4]. Padded indices are seq_idx = 2, 3 for both batch heads.
        for head in 0..2 {
            for seq_idx in 2..4 {
                for d_idx in 0..4 {
                    let idx = head * 16 + seq_idx * 4 + d_idx;
                    assert!(
                        k_grad_slice[idx].abs() < EPS,
                        "k_grad at index {} should be 0, but got {}",
                        idx,
                        k_grad_slice[idx]
                    );
                }
            }
        }
    }
}
