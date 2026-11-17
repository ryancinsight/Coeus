use coeus_autograd::{CausalMask, NullMask, Var};
use coeus_core::MoiraiBackend;
use coeus_nn::{
    transformer_decoder_layer, MhaProjectionParams, Transformer, TransformerDecoder,
    TransformerDecoderLayer, TransformerDecoderLayerParams,
};
use coeus_tensor::Tensor;

#[test]
fn test_transformer_decoder_layer() {
    const H: usize = 2;
    let d_model = 8;
    let d_ff = 16;
    let backend = MoiraiBackend;

    let layer = TransformerDecoderLayer::<f64, MoiraiBackend, H, CausalMask, NullMask>::new(
        d_model, d_ff, 0.0,
    )
    .expect("valid decoder layer fixture");

    // parameters check
    let params = layer.parameters();
    // norm1 (2), self_attn (8), norm2 (2), cross_attn (8), norm3 (2), ffn (4) = 26 parameters
    assert_eq!(params.len(), 26);
    let named = layer.named_parameters();
    let expected_names = [
        "norm1.weight",
        "norm1.bias",
        "self_attention.query.weight",
        "self_attention.key.weight",
        "self_attention.value.weight",
        "self_attention.output.weight",
        "self_attention.query.bias",
        "self_attention.key.bias",
        "self_attention.value.bias",
        "self_attention.output.bias",
        "norm2.weight",
        "norm2.bias",
        "cross_attention.query.weight",
        "cross_attention.key.weight",
        "cross_attention.value.weight",
        "cross_attention.output.weight",
        "cross_attention.query.bias",
        "cross_attention.key.bias",
        "cross_attention.value.bias",
        "cross_attention.output.bias",
        "norm3.weight",
        "norm3.bias",
        "feed_forward.input.weight",
        "feed_forward.input.bias",
        "feed_forward.output.weight",
        "feed_forward.output.bias",
    ];
    assert_eq!(
        named
            .iter()
            .map(|parameter| parameter.name.as_str())
            .collect::<Vec<_>>(),
        expected_names
    );
    for (plain, named) in params.iter().zip(&named) {
        assert!(std::sync::Arc::ptr_eq(
            plain
                .grad
                .as_ref()
                .expect("trainable parameter gradient buffer"),
            named
                .var
                .grad
                .as_ref()
                .expect("named parameter gradient buffer")
        ));
    }
    let batch = 2;
    let seq_tgt = 4;
    let seq_src = 5;

    let tgt = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_tgt, d_model], &backend)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let memory = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_src, d_model], &backend)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = layer
        .forward_decoder(&tgt, &memory)
        .expect("valid TransformerDecoderLayer input");
    assert_eq!(output.tensor.shape(), &[batch, seq_tgt, d_model]);
    let output_fn = transformer_decoder_layer::<f64, MoiraiBackend, H, CausalMask, NullMask>(
        &tgt,
        &memory,
        TransformerDecoderLayerParams {
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
            cross_attn: MhaProjectionParams {
                w_q: &layer.cross_attn.w_q,
                b_q: layer.cross_attn.b_q.as_ref(),
                w_k: &layer.cross_attn.w_k,
                b_k: layer.cross_attn.b_k.as_ref(),
                w_v: &layer.cross_attn.w_v,
                b_v: layer.cross_attn.b_v.as_ref(),
                w_o: &layer.cross_attn.w_o,
                b_o: layer.cross_attn.b_o.as_ref(),
            },
            norm3_weight: &layer.norm3.weight,
            norm3_bias: &layer.norm3.bias,
            ffn_w1: &layer.ffn.linear1.weight,
            ffn_b1: layer.ffn.linear1.bias.as_ref(),
            ffn_w2: &layer.ffn.linear2.weight,
            ffn_b2: layer.ffn.linear2.bias.as_ref(),
            self_attn_residual_dropout_p: 0.0,
            self_attn_residual_training: false,
            cross_attn_residual_dropout_p: 0.0,
            cross_attn_residual_training: false,
            ffn_hidden_dropout_p: 0.0,
            ffn_hidden_training: false,
            ffn_residual_dropout_p: 0.0,
            ffn_residual_training: false,
        },
    )
    .expect("valid TransformerDecoderLayer functional input");
    for (a, b) in output
        .tensor
        .as_slice()
        .iter()
        .zip(output_fn.tensor.as_slice())
    {
        assert!(
            (a - b).abs() < 1e-10,
            "decoder layer functional parity mismatch: {a} vs {b}"
        );
    }

    // Backward pass
    let loss = coeus_autograd::sum(&output).expect("invariant: test operation succeeds");
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");

    let tgt_grad = tgt
        .grad()
        .expect("invariant: decoder layer gives its target a gradient");
    let memory_grad = memory
        .grad()
        .expect("invariant: decoder layer gives its memory a gradient");
    assert_eq!(tgt_grad.shape(), &[batch, seq_tgt, d_model]);
    assert_eq!(memory_grad.shape(), &[batch, seq_src, d_model]);
    for (i, p) in params.iter().enumerate() {
        let grad = p
            .grad()
            .unwrap_or_else(|| panic!("DecoderLayer parameter {i} has no gradient"));
        assert!(
            grad.as_slice().iter().all(|value| value.is_finite()),
            "DecoderLayer parameter {i} gradient must be finite"
        );
    }

    // Central differences on sum(output), reusing the same layer weights and
    // perturbing each input independently.
    let evaluate = |tgt_data: &[f64], mem_data: &[f64]| -> f64 {
        let probe_tgt = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice_on(
                [batch, seq_tgt, d_model],
                tgt_data,
                &backend,
            )
            .expect("invariant: test backend operation succeeds"),
            false,
        )
        .expect("invariant: test backend operation succeeds");
        let probe_mem = Var::new(
            Tensor::<f64, MoiraiBackend>::from_slice_on(
                [batch, seq_src, d_model],
                mem_data,
                &backend,
            )
            .expect("invariant: test backend operation succeeds"),
            false,
        )
        .expect("invariant: test backend operation succeeds");
        layer
            .forward_decoder(&probe_tgt, &probe_mem)
            .expect("invariant: probe forward succeeds")
            .tensor
            .as_slice()
            .iter()
            .sum::<f64>()
    };
    let tgt_data = vec![1.0_f64; batch * seq_tgt * d_model];
    let mem_data = vec![1.0_f64; batch * seq_src * d_model];
    let step = 1e-6_f64;
    for index in [0, 7, 31] {
        let mut plus = tgt_data.clone();
        plus[index] += step;
        let mut minus = tgt_data.clone();
        minus[index] -= step;
        let numeric = (evaluate(&plus, &mem_data) - evaluate(&minus, &mem_data)) / (2.0 * step);
        let analytic = tgt_grad.as_slice()[index];
        assert!(
            (analytic - numeric).abs() <= 1e-5,
            "decoder layer tgt grad[{index}]: analytic {analytic} vs numeric {numeric}"
        );
    }
    for index in [0, 13, 63] {
        let mut plus = mem_data.clone();
        plus[index] += step;
        let mut minus = mem_data.clone();
        minus[index] -= step;
        let numeric = (evaluate(&tgt_data, &plus) - evaluate(&tgt_data, &minus)) / (2.0 * step);
        let analytic = memory_grad.as_slice()[index];
        assert!(
            (analytic - numeric).abs() <= 1e-5,
            "decoder layer memory grad[{index}]: analytic {analytic} vs numeric {numeric}"
        );
    }
}

#[test]
fn test_transformer_decoder() {
    const H: usize = 2;
    const N: usize = 3;
    let d_model = 8;
    let d_ff = 16;
    let backend = MoiraiBackend;

    let decoder = TransformerDecoder::<f64, MoiraiBackend, H, N, CausalMask, NullMask>::new(
        d_model, d_ff, 0.0,
    )
    .expect("valid decoder stack fixture");

    let params = decoder.parameters();
    assert_eq!(params.len(), 26 * N);

    let batch = 2;
    let seq_tgt = 4;
    let seq_src = 5;

    let tgt = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_tgt, d_model], &backend)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let memory = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_src, d_model], &backend)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = decoder
        .forward_decoder(&tgt, &memory)
        .expect("valid TransformerDecoder input");
    assert_eq!(output.tensor.shape(), &[batch, seq_tgt, d_model]);

    let loss = coeus_autograd::sum(&output).expect("invariant: test operation succeeds");
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");

    let tgt_grad = tgt
        .grad()
        .expect("invariant: decoder stack gives its target a gradient");
    let memory_grad = memory
        .grad()
        .expect("invariant: decoder stack gives its memory a gradient");
    assert_eq!(tgt_grad.shape(), &[batch, seq_tgt, d_model]);
    assert_eq!(memory_grad.shape(), &[batch, seq_src, d_model]);
    for (i, p) in params.iter().enumerate() {
        let grad = p
            .grad()
            .unwrap_or_else(|| panic!("Decoder stack parameter {i} has no gradient"));
        assert!(
            grad.as_slice().iter().all(|value| value.is_finite()),
            "Decoder stack parameter {i} gradient must be finite"
        );
    }

    let tgt_data = vec![1.0_f64; batch * seq_tgt * d_model];
    let mem_data = vec![1.0_f64; batch * seq_src * d_model];
    assert_gradient_matches_central_differences(
        "decoder stack tgt",
        &tgt_data,
        tgt_grad.as_slice(),
        &[0, 7, 31],
        &|data| {
            let probe = Var::new(
                Tensor::<f64, MoiraiBackend>::from_slice_on(
                    [batch, seq_tgt, d_model],
                    data,
                    &backend,
                )
                .expect("invariant: test backend operation succeeds"),
                false,
            )
            .expect("invariant: test backend operation succeeds");
            decoder
                .forward_decoder(&probe, &memory)
                .expect("invariant: probe forward succeeds")
                .tensor
                .as_slice()
                .iter()
                .sum::<f64>()
        },
    );
    assert_gradient_matches_central_differences(
        "decoder stack memory",
        &mem_data,
        memory_grad.as_slice(),
        &[0, 13, 63],
        &|data| {
            let probe = Var::new(
                Tensor::<f64, MoiraiBackend>::from_slice_on(
                    [batch, seq_src, d_model],
                    data,
                    &backend,
                )
                .expect("invariant: test backend operation succeeds"),
                false,
            )
            .expect("invariant: test backend operation succeeds");
            decoder
                .forward_decoder(&tgt, &probe)
                .expect("invariant: probe forward succeeds")
                .tensor
                .as_slice()
                .iter()
                .sum::<f64>()
        },
    );
}

fn assert_gradient_matches_central_differences(
    label: &str,
    baseline: &[f64],
    analytic_grad: &[f64],
    indices: &[usize],
    evaluate: &impl Fn(&[f64]) -> f64,
) {
    let step = 1e-6_f64;
    for &index in indices {
        let mut plus = baseline.to_vec();
        plus[index] += step;
        let mut minus = baseline.to_vec();
        minus[index] -= step;
        let numeric = (evaluate(&plus) - evaluate(&minus)) / (2.0 * step);
        let analytic = analytic_grad[index];
        assert!(
            (analytic - numeric).abs() <= 1e-5,
            "{label} grad[{index}]: analytic {analytic} vs numeric {numeric}"
        );
    }
}

#[path = "nn_transformer_tests/extra.rs"]
mod extra;
