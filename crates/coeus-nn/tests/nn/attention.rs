use coeus_autograd::Var;
use coeus_nn::Module;
use coeus_tensor::Tensor;

#[test]
fn test_mha_self_attention_shape() {
    use coeus_autograd::NullMask;
    use coeus_nn::attention::mha::MultiHeadAttention;

    let mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 4, NullMask>::new(8, true).expect("construct module");
    let input = Var::new(Tensor::zeros(vec![1, 5, 8]).expect("construct tensor"), true).expect("construct variable");
    let output = mha.forward(&input).expect("run forward");
    assert_eq!(output.tensor.shape(), &[1, 5, 8]);
}

#[test]
fn test_mha_cross_attention_shape() {
    use coeus_autograd::NullMask;
    use coeus_nn::attention::mha::MultiHeadAttention;
    use coeus_nn::multi_head_attention_cross;
    use coeus_nn::MhaProjectionParams;

    let mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 2, NullMask>::new(4, true).expect("construct module");
    let query = Var::new(Tensor::zeros(vec![1, 3, 4]).expect("construct tensor"), true).expect("construct variable");
    let key = Var::new(Tensor::zeros(vec![1, 5, 4]).expect("construct tensor"), false).expect("construct variable");
    let value = Var::new(Tensor::zeros(vec![1, 5, 4]).expect("construct tensor"), false).expect("construct variable");
    let output = mha.forward_cross(&query, &key, &value, None).expect("run forward");
    let output_fn = multi_head_attention_cross::<f64, coeus_core::MoiraiBackend, 2, NullMask>(
        &query,
        &key,
        &value,
        MhaProjectionParams {
            w_q: &mha.w_q,
            b_q: mha.b_q.as_ref(),
            w_k: &mha.w_k,
            b_k: mha.b_k.as_ref(),
            w_v: &mha.w_v,
            b_v: mha.b_v.as_ref(),
            w_o: &mha.w_o,
            b_o: mha.b_o.as_ref(),
        },
        None,
    ).expect("run operation");
    assert_eq!(output.tensor.shape(), &[1, 3, 4]);
    assert_eq!(output_fn.tensor.shape(), &[1, 3, 4]);
    for (a, b) in output
        .tensor
        .as_slice()
        .iter()
        .zip(output_fn.tensor.as_slice())
    {
        assert!(
            (a - b).abs() < 1e-10,
            "MHA module/functional mismatch: {a} vs {b}"
        );
    }
}

#[test]
fn test_mha_backward_gradients_exist() {
    use coeus_autograd::NullMask;
    use coeus_nn::attention::mha::MultiHeadAttention;

    let mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 2, NullMask>::new(4, true).expect("construct module");
    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 4], &[0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let output = mha.forward(&input).expect("run forward");
    output.backward().expect("run backward");
    assert!(input.grad().is_some());
    assert!(mha.w_q.grad().is_some());
    assert!(mha.w_k.grad().is_some());
    assert!(mha.w_v.grad().is_some());
    assert!(mha.w_o.grad().is_some());
    assert!(mha.b_q.as_ref().unwrap().grad().is_some());
    assert!(mha.b_k.as_ref().unwrap().grad().is_some());
    assert!(mha.b_v.as_ref().unwrap().grad().is_some());
    assert!(mha.b_o.as_ref().unwrap().grad().is_some());
}
