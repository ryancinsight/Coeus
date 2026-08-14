#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
use coeus_autograd::Var;
use coeus_nn::Module;
use coeus_tensor::Tensor;

#[test]
fn test_mha_self_attention_shape() {
    use coeus_autograd::NullMask;
    use coeus_nn::attention::mha::MultiHeadAttention;

    let mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 4, NullMask>::new(8, true)
        .expect("valid four-head attention fixture");
    let input = Var::new(Tensor::zeros(vec![1, 5, 8]), true);
    let output = mha.forward(&input).expect("valid MultiHeadAttention input");
    assert_eq!(output.tensor.shape(), &[1, 5, 8]);
}

#[test]
fn test_mha_cross_attention_shape() {
    use coeus_autograd::NullMask;
    use coeus_nn::attention::mha::MultiHeadAttention;
    use coeus_nn::multi_head_attention_cross;
    use coeus_nn::MhaProjectionParams;

    let mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 2, NullMask>::new(4, true)
        .expect("valid two-head cross-attention fixture");
    let query = Var::new(Tensor::zeros(vec![1, 3, 4]), true);
    let key = Var::new(Tensor::zeros(vec![1, 5, 4]), false);
    let value = Var::new(Tensor::zeros(vec![1, 5, 4]), false);
    let output = mha
        .forward_cross(&query, &key, &value, None)
        .expect("valid MultiHeadAttention cross-attention input");
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
    )
    .expect("valid functional MultiHeadAttention cross-attention input");
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

    let mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 2, NullMask>::new(4, true)
        .expect("valid two-head backward fixture");
    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 4], &[0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        true,
    );
    let output = mha.forward(&input).expect("valid MultiHeadAttention input");
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
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

#[test]
fn scaled_attention_rejects_rank_before_indexing() {
    use coeus_autograd::NullMask;
    use coeus_nn::{ModuleError, ScaledDotProductAttention};

    let attention = ScaledDotProductAttention::<f64, coeus_core::MoiraiBackend, NullMask>::new();
    let input = Var::new(Tensor::zeros(vec![2, 4]), false);

    let error = Module::forward(&attention, &input)
        .err()
        .expect("rank-two input must be rejected");
    assert!(matches!(
        error,
        ModuleError::InvalidRank {
            module: "ScaledDotProductAttention",
            expected: "3",
            actual: 2,
        }
    ));
}

#[test]
fn mha_rejects_non_divisible_head_count() {
    use coeus_autograd::NullMask;
    use coeus_nn::{MhaProjectionParams, ModuleError, MultiHeadAttention};

    let mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 2, NullMask>::new(4, true)
        .expect("valid projection fixture");
    let input = Var::new(Tensor::zeros(vec![1, 2, 4]), false);
    let error =
        coeus_nn::multi_head_attention_cross::<f64, coeus_core::MoiraiBackend, 3, NullMask>(
            &input,
            &input,
            &input,
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
        )
        .err()
        .expect("four model features cannot be partitioned into three heads");

    assert!(matches!(
        error,
        ModuleError::ShapeMismatch {
            module: "MultiHeadAttention",
            parameter: "attention heads",
            expected,
            actual,
        } if expected == vec![4] && actual == vec![3]
    ));
}

#[test]
fn mha_rejects_projection_and_mask_shapes() {
    use coeus_autograd::NullMask;
    use coeus_nn::{ModuleError, MultiHeadAttention};

    let mut mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 2, NullMask>::new(4, true)
        .expect("valid mutable projection fixture");
    let input = Var::new(Tensor::zeros(vec![1, 3, 4]), false);
    mha.w_k = Var::new(Tensor::zeros(vec![3, 4]), true);
    let projection_error = mha
        .forward_cross(&input, &input, &input, None)
        .err()
        .expect("invalid key projection must be rejected");
    assert!(matches!(
        projection_error,
        ModuleError::ShapeMismatch {
            module: "MultiHeadAttention",
            parameter: "key projection",
            expected,
            actual,
        } if expected == vec![4, 4] && actual == vec![3, 4]
    ));

    mha.w_k = Var::new(Tensor::zeros(vec![4, 4]), true);
    let mask = Var::new(Tensor::ones(vec![1, 4]), false);
    let mask_error = mha
        .forward_cross(&input, &input, &input, Some(&mask))
        .err()
        .expect("mask sequence mismatch must be rejected");
    assert!(matches!(
        mask_error,
        ModuleError::ShapeMismatch {
            module: "MultiHeadAttention",
            parameter: "key padding mask",
            expected,
            actual,
        } if expected == vec![1, 3] && actual == vec![1, 4]
    ));
}

#[test]
fn mha_rejects_incompatible_query_key_value_shapes() {
    use coeus_autograd::NullMask;
    use coeus_nn::{ModuleError, MultiHeadAttention};

    let mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 2, NullMask>::new(4, true)
        .expect("valid incompatible-shape fixture");
    let query = Var::new(Tensor::zeros(vec![2, 3, 4]), false);
    let key = Var::new(Tensor::zeros(vec![1, 5, 4]), false);
    let value = Var::new(Tensor::zeros(vec![2, 5, 4]), false);
    let batch_error = mha
        .forward_cross(&query, &key, &value, None)
        .err()
        .expect("query and key batches must match");
    assert!(matches!(
        batch_error,
        ModuleError::ShapeMismatch {
            module: "MultiHeadAttention",
            parameter: "key batch",
            expected,
            actual,
        } if expected == vec![2] && actual == vec![1]
    ));

    let key = Var::new(Tensor::zeros(vec![2, 5, 4]), false);
    let value = Var::new(Tensor::zeros(vec![2, 4, 4]), false);
    let sequence_error = mha
        .forward_cross(&query, &key, &value, None)
        .err()
        .expect("key and value sequence lengths must match");
    assert!(matches!(
        sequence_error,
        ModuleError::ShapeMismatch {
            module: "MultiHeadAttention",
            parameter: "value sequence",
            expected,
            actual,
        } if expected == vec![5] && actual == vec![4]
    ));
}
