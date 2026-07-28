use coeus_autograd::{CausalMask, NullMask, Var};
use coeus_core::MoiraiBackend;
use coeus_nn::{
    transformer_decoder_layer, MhaProjectionParams, Module, Transformer, TransformerDecoder,
    TransformerDecoderLayer, TransformerDecoderLayerParams,
};
use coeus_tensor::Tensor;

#[test]
fn test_transformer_decoder_layer() {
    const H: usize = 2;
    let d_model = 8;
    let d_ff = 16;
    let backend = MoiraiBackend;

    let mut layer = TransformerDecoderLayer::<f64, MoiraiBackend, H, CausalMask, NullMask>::new(
        d_model, d_ff, 0.0,
    ).expect("construct module");

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
    let mut reordered = named.clone();
    reordered.swap(0, 1);
    assert!(matches!(
        layer.load_named_parameters(&reordered),
        Err(coeus_nn::ParameterLoadError::Name { index: 0, .. })
    ));

    let batch = 2;
    let seq_tgt = 4;
    let seq_src = 5;

    let tgt = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_tgt, d_model], &backend).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let memory = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_src, d_model], &backend).expect("construct tensor"),
        true,
    ).expect("construct variable");

    let output = layer.forward_decoder(&tgt, &memory).expect("run forward");
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
    ).expect("run operation");
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
    let loss = coeus_autograd::sum(&output).expect("run operation");
    loss.backward().expect("run backward");

    assert!(tgt.grad().is_some());
    assert!(memory.grad().is_some());
    for (i, p) in params.iter().enumerate() {
        assert!(
            p.grad().is_some(),
            "DecoderLayer parameter {i} has no gradient"
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
    ).expect("construct module");

    let params = decoder.parameters();
    assert_eq!(params.len(), 26 * N);

    let batch = 2;
    let seq_tgt = 4;
    let seq_src = 5;

    let tgt = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_tgt, d_model], &backend).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let memory = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_src, d_model], &backend).expect("construct tensor"),
        true,
    ).expect("construct variable");

    let output = decoder.forward_decoder(&tgt, &memory).expect("run forward");
    assert_eq!(output.tensor.shape(), &[batch, seq_tgt, d_model]);

    let loss = coeus_autograd::sum(&output).expect("run operation");
    loss.backward().expect("run backward");

    assert!(tgt.grad().is_some());
    assert!(memory.grad().is_some());
    for (i, p) in params.iter().enumerate() {
        assert!(
            p.grad().is_some(),
            "Decoder stack parameter {i} has no gradient"
        );
    }
}

#[test]
fn test_transformer_seq2seq() {
    const H: usize = 2;
    const NUM_ENC: usize = 2;
    const NUM_DEC: usize = 2;
    let d_model = 8;
    let d_ff = 16;
    let backend = MoiraiBackend;

    let transformer = Transformer::<
        f64,
        MoiraiBackend,
        H,
        NUM_ENC,
        NUM_DEC,
        NullMask,
        CausalMask,
        NullMask,
    >::new(d_model, d_ff, 0.0).expect("construct module");

    let params = transformer.parameters();
    // Encoder layer: norm1 (2), self_attn (8), norm2 (2), ffn (4) = 16 parameters
    // Encoder: 16 * NUM_ENC = 32 parameters
    // Decoder: 26 * NUM_DEC = 52 parameters
    // Total = 84 parameters
    assert_eq!(params.len(), 84);
    let named = transformer.named_parameters();
    assert_eq!(named.len(), params.len());
    let unique = named
        .iter()
        .map(|parameter| parameter.name.as_str())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), named.len());
    assert_eq!(named[0].name, "encoder.layers.0.norm1.weight");
    assert_eq!(
        named.last().expect("non-empty transformer parameters").name,
        "decoder.layers.1.feed_forward.output.bias"
    );

    let batch = 2;
    let seq_src = 5;
    let seq_tgt = 4;

    let src = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_src, d_model], &backend).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let tgt = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_tgt, d_model], &backend).expect("construct tensor"),
        true,
    ).expect("construct variable");

    let output = transformer.forward_seq2seq(&src, &tgt).expect("run operation");
    assert_eq!(output.tensor.shape(), &[batch, seq_tgt, d_model]);

    let loss = coeus_autograd::sum(&output).expect("run operation");
    loss.backward().expect("run backward");

    assert!(src.grad().is_some());
    assert!(tgt.grad().is_some());
    for (i, p) in params.iter().enumerate() {
        assert!(
            p.grad().is_some(),
            "Seq2Seq Transformer parameter {i} has no gradient"
        );
    }
}
