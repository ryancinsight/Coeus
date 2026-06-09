use coeus_autograd::Var;
use coeus_nn::Module;
use coeus_tensor::Tensor;

#[test]
fn test_mha_self_attention_shape() {
    use coeus_autograd::NullMask;
    use coeus_nn::attention::mha::MultiHeadAttention;

    let mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 4, NullMask>::new(8, true);
    let input = Var::new(Tensor::zeros(vec![1, 5, 8]), true);
    let output = mha.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 5, 8]);
}

#[test]
fn test_mha_cross_attention_shape() {
    use coeus_autograd::NullMask;
    use coeus_nn::attention::mha::MultiHeadAttention;

    let mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 2, NullMask>::new(4, true);
    let query = Var::new(Tensor::zeros(vec![1, 3, 4]), true);
    let key = Var::new(Tensor::zeros(vec![1, 5, 4]), false);
    let value = Var::new(Tensor::zeros(vec![1, 5, 4]), false);
    let output = mha.forward_cross(&query, &key, &value, None);
    assert_eq!(output.tensor.shape(), &[1, 3, 4]);
}

#[test]
fn test_mha_backward_gradients_exist() {
    use coeus_autograd::NullMask;
    use coeus_nn::attention::mha::MultiHeadAttention;

    let mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 2, NullMask>::new(4, true);
    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 4], &[0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        true,
    );
    let output = mha.forward(&input);
    output.backward();
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
