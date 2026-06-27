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

    let layer = TransformerDecoderLayer::<f64, MoiraiBackend, H, CausalMask, NullMask>::new(
        d_model, d_ff, 0.0,
    );

    // parameters check
    let params = layer.parameters();
    // norm1 (2), self_attn (8), norm2 (2), cross_attn (8), norm3 (2), ffn (4) = 26 parameters
    assert_eq!(params.len(), 26);

    let batch = 2;
    let seq_tgt = 4;
    let seq_src = 5;

    let tgt = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_tgt, d_model], &backend),
        true,
    );
    let memory = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_src, d_model], &backend),
        true,
    );

    let output = layer.forward_decoder(&tgt, &memory);
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
    );
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
    let loss = coeus_autograd::sum(&output);
    loss.backward();

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
    );

    let params = decoder.parameters();
    assert_eq!(params.len(), 26 * N);

    let batch = 2;
    let seq_tgt = 4;
    let seq_src = 5;

    let tgt = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_tgt, d_model], &backend),
        true,
    );
    let memory = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_src, d_model], &backend),
        true,
    );

    let output = decoder.forward_decoder(&tgt, &memory);
    assert_eq!(output.tensor.shape(), &[batch, seq_tgt, d_model]);

    let loss = coeus_autograd::sum(&output);
    loss.backward();

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
    >::new(d_model, d_ff, 0.0);

    let params = transformer.parameters();
    // Encoder layer: norm1 (2), self_attn (8), norm2 (2), ffn (4) = 16 parameters
    // Encoder: 16 * NUM_ENC = 32 parameters
    // Decoder: 26 * NUM_DEC = 52 parameters
    // Total = 84 parameters
    assert_eq!(params.len(), 84);

    let batch = 2;
    let seq_src = 5;
    let seq_tgt = 4;

    let src = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_src, d_model], &backend),
        true,
    );
    let tgt = Var::new(
        Tensor::<f64, MoiraiBackend>::ones_on([batch, seq_tgt, d_model], &backend),
        true,
    );

    let output = transformer.forward_seq2seq(&src, &tgt);
    assert_eq!(output.tensor.shape(), &[batch, seq_tgt, d_model]);

    let loss = coeus_autograd::sum(&output);
    loss.backward();

    assert!(src.grad().is_some());
    assert!(tgt.grad().is_some());
    for (i, p) in params.iter().enumerate() {
        assert!(
            p.grad().is_some(),
            "Seq2Seq Transformer parameter {i} has no gradient"
        );
    }
}
