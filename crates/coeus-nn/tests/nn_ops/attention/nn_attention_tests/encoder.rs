//! The feed-forward block and the transformer encoder layer.

use super::tests::*;

// ── FeedForward: shape ──────────────────────────────────────────────────

#[test]
fn ffn_forward_shape() {
    let d_model = 16;
    let d_ff = 64;
    let ffn = FeedForward::<f32, B>::new(d_model, d_ff, 0.0)
        .expect("invariant: the fixture's layer dimensions are non-zero");

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

    let layer = TransformerEncoderLayer::<f32, B, H, NullMask>::new(d_model, d_ff, 0.0)
        .expect("valid encoder layer shape fixture");
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

    let layer = TransformerEncoderLayer::<f32, B, H, NullMask>::new(d_model, d_ff, 0.0)
        .expect("valid encoder layer gradient fixture");
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
    assert!(!params.is_empty(), "EncoderLayer must have parameters");
    assert_parameters_received_signal(&params, "EncoderLayer");
}

#[test]
fn encoder_layer_forward_with_key_padding_mask_shape_and_grad() {
    const H: usize = 2;
    let d_model = 8;
    let d_ff = 32;

    let layer = TransformerEncoderLayer::<f32, B, H, NullMask>::new(d_model, d_ff, 0.0)
        .expect("valid masked encoder layer fixture");
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

    let layer = TransformerEncoderLayer::<f32, B, H, NullMask>::new(d_model, d_ff, 0.0)
        .expect("valid encoder mask parity fixture");
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
    let layer = TransformerEncoderLayer::<f32, B, H, NullMask>::new(8, 16, 0.0)
        .expect("valid encoder rank-validation fixture");
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
