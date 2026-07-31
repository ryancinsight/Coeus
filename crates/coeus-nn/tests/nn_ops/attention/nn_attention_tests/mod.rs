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
mod mha_mask;

#[cfg(test)]
mod tests {
    use coeus_autograd::Var;
    use coeus_core::{MoiraiBackend, Storage};
    use coeus_nn::{
        feed_forward, transformer_encoder_layer, FeedForward, MhaProjectionParams, Module,
        MultiHeadAttention, NullMask, SinusoidalEncoding, TransformerEncoderLayer,
        TransformerEncoderLayerParams,
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
            coeus_autograd::sdp_attention::<f32, B, NullMask>(&q_var, &k_var, &v_var, None, scale)
                .expect("valid attention fixture");

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

        let q_t = Tensor::<f32, B>::from_slice_on([batch, seq, d], &q_data, &backend);
        let k_t = Tensor::<f32, B>::from_slice_on([batch, seq, d], &k_data, &backend);
        let v_t = Tensor::<f32, B>::from_slice_on([batch, seq, d], &v_data, &backend);

        let q = Var::new(q_t, true);
        let k = Var::new(k_t, true);
        let v = Var::new(v_t, true);

        let scale = 1.0_f32 / (d as f32).sqrt();
        let (out, _) = coeus_autograd::sdp_attention::<f32, B, NullMask>(&q, &k, &v, None, scale)
            .expect("valid attention fixture");

        // Sum-reduce to scalar loss and backprop
        let loss = coeus_autograd::sum(&out);
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

        let mha = MultiHeadAttention::<f32, B, H, NullMask>::new(d_model, true);
        let backend = B::default();
        let x = Tensor::<f32, B>::ones_on([batch, seq, d_model], &backend);
        let x_var = Var::new(x, false);

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

        let mha = MultiHeadAttention::<f32, B, H, NullMask>::new(d_model, true);
        let backend = B::default();
        let x = Tensor::<f32, B>::ones_on([batch, seq, d_model], &backend);
        let x_var = Var::new(x, false);

        let out = mha.forward(&x_var).expect("valid MultiHeadAttention input");
        let loss = coeus_autograd::sum(&out);
        loss.backward()
            .expect("invariant: valid autograd fixture completes backward");

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

        let out = pe.forward(&x_var).expect("valid SinusoidalEncoding input");
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

        let out = ffn.forward(&x_var).expect("valid FeedForward input");
        assert_eq!(out.tensor.shape(), &[batch, seq, d_model]);

        let out_fn = feed_forward(
            &x_var,
            &ffn.linear1.weight,
            ffn.linear1.bias.as_ref(),
            &ffn.linear2.weight,
            ffn.linear2.bias.as_ref(),
            0.0,
        )
        .expect("valid FeedForward functional input");
        assert_eq!(out_fn.tensor.shape(), &[batch, seq, d_model]);
        for (a, b) in out.tensor.as_slice().iter().zip(out_fn.tensor.as_slice()) {
            assert!(
                (a - b).abs() < 1e-6,
                "feed_forward parity mismatch: {a} vs {b}"
            );
        }
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

        let out = layer
            .forward(&x_var)
            .expect("valid TransformerEncoderLayer input");
        assert_eq!(
            out.tensor.shape(),
            &[batch, seq, d_model],
            "EncoderLayer output shape mismatch"
        );

        let out_fn = transformer_encoder_layer::<f32, B, H, NullMask>(
            &x_var,
            None,
            TransformerEncoderLayerParams {
                norm1_weight: &layer.norm1.weight,
                norm1_bias: &layer.norm1.bias,
                self_attn: MhaProjectionParams {
                    w_q: &layer.self_attn.w_q,
                    b_q: layer.self_attn.b_q.as_ref(),
                    w_k: &layer.self_attn.w_k,
                    b_k: layer.self_attn.b_k.as_ref(),
                    w_v: &layer.self_attn.w_v,
                    b_v: layer.self_attn.b_v.as_ref(),
                    w_o: &layer.self_attn.w_o,
                    b_o: layer.self_attn.b_o.as_ref(),
                },
                norm2_weight: &layer.norm2.weight,
                norm2_bias: &layer.norm2.bias,
                ffn_w1: &layer.ffn.linear1.weight,
                ffn_b1: layer.ffn.linear1.bias.as_ref(),
                ffn_w2: &layer.ffn.linear2.weight,
                ffn_b2: layer.ffn.linear2.bias.as_ref(),
                attn_residual_dropout_p: 0.0,
                attn_residual_training: false,
                ffn_hidden_dropout_p: 0.0,
                ffn_hidden_training: false,
                ffn_residual_dropout_p: 0.0,
                ffn_residual_training: false,
            },
        )
        .expect("valid TransformerEncoderLayer functional input");
        assert_eq!(out_fn.tensor.shape(), &[batch, seq, d_model]);
        for (a, b) in out.tensor.as_slice().iter().zip(out_fn.tensor.as_slice()) {
            assert!(
                (a - b).abs() < 1e-6,
                "encoder_layer functional parity mismatch: {a} vs {b}"
            );
        }
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

        let out = layer
            .forward(&x_var)
            .expect("valid TransformerEncoderLayer input");
        let loss = coeus_autograd::sum(&out);
        loss.backward()
            .expect("invariant: valid autograd fixture completes backward");

        let params = layer.parameters();
        for (i, p) in params.iter().enumerate() {
            assert!(
                p.grad.is_some(),
                "EncoderLayer parameter {i} has no gradient"
            );
        }
    }

    #[test]
    fn encoder_layer_forward_with_key_padding_mask_shape_and_grad() {
        const H: usize = 2;
        let d_model = 8;
        let d_ff = 32;

        let layer = TransformerEncoderLayer::<f32, B, H, NullMask>::new(d_model, d_ff, 0.0);
        let backend = B::default();
        let batch = 1;
        let seq = 4;

        let x = Tensor::<f32, B>::ones_on([batch, seq, d_model], &backend);
        let x_var = Var::new(x, true);

        // Keep first two tokens, mask the last two.
        let mask = Tensor::<f32, B>::from_slice_on([batch, seq], &[1.0, 1.0, 0.0, 0.0], &backend);
        let mask_var = Var::new(mask, false);

        let out = layer
            .forward_with_mask(&x_var, Some(&mask_var))
            .expect("valid masked TransformerEncoderLayer input");
        assert_eq!(
            out.tensor.shape(),
            &[batch, seq, d_model],
            "EncoderLayer(masked) output shape mismatch"
        );

        let loss = coeus_autograd::sum(&out);
        loss.backward()
            .expect("invariant: valid autograd fixture completes backward");

        let params = layer.parameters();
        for (i, p) in params.iter().enumerate() {
            assert!(
                p.grad.is_some(),
                "EncoderLayer(masked) parameter {i} has no gradient"
            );
        }
    }

    #[test]
    fn encoder_layer_all_ones_mask_matches_unmasked_forward() {
        const H: usize = 2;
        let d_model = 8;
        let d_ff = 32;

        let layer = TransformerEncoderLayer::<f32, B, H, NullMask>::new(d_model, d_ff, 0.0);
        let backend = B::default();
        let batch = 1;
        let seq = 4;

        let data: Vec<f32> = (1..=(batch * seq * d_model))
            .map(|x| x as f32 * 0.01)
            .collect();
        let x = Tensor::<f32, B>::from_slice_on([batch, seq, d_model], &data, &backend);
        let x_var = Var::new(x, false);

        let mask = Tensor::<f32, B>::ones_on([batch, seq], &backend);
        let mask_var = Var::new(mask, false);

        let unmasked = layer
            .forward(&x_var)
            .expect("valid TransformerEncoderLayer input");
        let masked = layer
            .forward_with_mask(&x_var, Some(&mask_var))
            .expect("valid masked TransformerEncoderLayer input");
        let unmasked_data = unmasked
            .tensor
            .storage()
            .try_as_slice()
            .expect("test: unmasked encoder output must be CPU-addressable");
        let masked_data = masked
            .tensor
            .storage()
            .try_as_slice()
            .expect("test: masked encoder output must be CPU-addressable");

        assert_eq!(unmasked_data.len(), masked_data.len());
        for (i, (a, b)) in unmasked_data.iter().zip(masked_data.iter()).enumerate() {
            assert!(
                (a - b).abs() < EPS,
                "all-ones mask mismatch at idx {i}: unmasked={a}, masked={b}"
            );
        }
    }

    #[test]
    fn encoder_layer_rejects_rank_before_normalization() {
        use coeus_nn::ModuleError;

        const H: usize = 2;
        let layer = TransformerEncoderLayer::<f32, B, H, NullMask>::new(8, 16, 0.0);
        let input = Var::new(Tensor::<f32, B>::ones([2, 8]), false);
        let error = layer
            .forward(&input)
            .err()
            .expect("rank-two encoder input must be rejected");

        assert!(matches!(
            error,
            ModuleError::InvalidRank {
                module: "TransformerEncoderLayer",
                expected: "3",
                actual: 2,
            }
        ));
    }
}
