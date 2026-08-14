#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
use coeus_autograd::Var;
use coeus_nn::{init, Conv1d, Module};
use coeus_tensor::{Tensor, Transpose};

#[test]
fn test_conv1d_forward_shape() {
    let conv = Conv1d::<f64>::new(3, 8, 3, true);
    let input = Var::new(Tensor::zeros(vec![2, 3, 32]), true);
    let output = conv.forward(&input).expect("valid Conv1d input");

    assert_eq!(output.tensor.shape(), &[2, 8, 30]);
    let params = conv.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_conv1d_forward_no_bias() {
    let conv = Conv1d::<f64>::new(1, 1, 3, false);
    let params = conv.parameters();
    assert_eq!(params.len(), 1);
    assert!(conv.bias.is_none());
}

#[test]
fn test_conv1d_forward_computation() {
    let mut conv = Conv1d::<f64>::new(1, 1, 2, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.0);
    }

    let input = Var::new(
        Tensor::from_slice(vec![1, 1, 4], &[1.0f64, 2.0, 3.0, 4.0]),
        true,
    );

    let output = conv.forward(&input).expect("valid Conv1d input");
    assert_eq!(output.tensor.shape(), &[1, 1, 3]);

    let out_slice = output.tensor.as_slice();
    assert_eq!(out_slice[0], 3.0);
    assert_eq!(out_slice[1], 5.0);
    assert_eq!(out_slice[2], 7.0);
}

#[test]
fn test_conv1d_forward_multi_channel() {
    let mut conv = Conv1d::<f64>::new(2, 1, 2, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.0);
    }

    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 3], &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
        true,
    );

    let output = conv.forward(&input).expect("valid Conv1d input");
    assert_eq!(output.tensor.shape(), &[1, 1, 2]);

    let out_slice = output.tensor.as_slice();
    assert_eq!(out_slice[0], 12.0);
    assert_eq!(out_slice[1], 16.0);
}

#[test]
fn test_conv1d_backward_gradients_match_reference() {
    let mut conv = Conv1d::<f64>::new(1, 1, 2, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.5);
    }

    let input = Var::new(
        Tensor::from_slice(vec![1, 1, 4], &[1.0, 2.0, 3.0, 4.0]),
        true,
    );

    let output = conv.forward(&input).expect("valid Conv1d input");
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    let input_grad = input.grad().expect("input gradient must be set");
    assert_eq!(input_grad.shape(), &[1, 1, 4]);
    assert_eq!(input_grad.as_slice(), &[1.0, 2.0, 2.0, 1.0]);

    let weight_grad = conv.weight.grad().expect("weight gradient must be set");
    assert_eq!(weight_grad.shape(), &[1, 1, 2]);
    assert_eq!(weight_grad.as_slice(), &[6.0, 9.0]);

    if let Some(ref b) = conv.bias {
        let bias_grad = b.grad().expect("bias gradient must be set");
        assert_eq!(bias_grad.shape(), &[1]);
        assert_eq!(bias_grad.as_slice(), &[3.0]);
    }
}

#[test]
fn test_conv1d_with_padding() {
    let mut conv = Conv1d::<f64>::with_params(1, 1, 3, 1, 1, 1, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.0);
    }

    let input = Var::new(Tensor::zeros(vec![1, 1, 4]), true);
    let output = conv.forward(&input).expect("valid Conv1d input");
    assert_eq!(output.tensor.shape(), &[1, 1, 4]);
}

#[test]
fn test_conv1d_with_stride() {
    let mut conv = Conv1d::<f64>::with_params(1, 1, 3, 2, 0, 1, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.0);
    }

    let input = Var::new(Tensor::zeros(vec![1, 1, 7]), true);
    let output = conv.forward(&input).expect("valid Conv1d input");
    assert_eq!(output.tensor.shape(), &[1, 1, 3]);
}

#[test]
fn test_conv1d_with_dilation() {
    let mut conv = Conv1d::<f64>::with_params(1, 1, 2, 1, 0, 2, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.0);
    }

    let input = Var::new(
        Tensor::from_slice(vec![1, 1, 7], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        true,
    );

    let output = conv.forward(&input).expect("valid Conv1d input");
    assert_eq!(output.tensor.shape(), &[1, 1, 5]);

    let out_slice = output.tensor.as_slice();
    assert_eq!(out_slice[0], 4.0);
    assert_eq!(out_slice[1], 6.0);
    assert_eq!(out_slice[2], 8.0);
    assert_eq!(out_slice[3], 10.0);
    assert_eq!(out_slice[4], 12.0);
}

#[test]
fn test_non_contiguous_cross_entropy() {
    use coeus_nn::cross_entropy_loss;
    let logits_raw = Tensor::<f64, coeus_core::MoiraiBackend>::from_slice(
        vec![3, 2],
        &[1.0f64, 0.0, 2.0, 2.0, 0.0, 1.0],
    );
    let logits_t = logits_raw.transpose();
    let logits = Var::new(logits_t, true);
    let targets = vec![1, 2];

    let loss_ce = cross_entropy_loss(&logits, &targets)
        .expect("invariant: test inputs have valid cross-entropy shapes and targets");
    assert_eq!(loss_ce.tensor.shape(), &[1]);

    let logits_cont = Var::new(
        Tensor::<f64, coeus_core::MoiraiBackend>::from_slice(
            vec![2, 3],
            &[1.0f64, 2.0, 0.0, 0.0, 2.0, 1.0],
        ),
        true,
    );
    let loss_ce_cont = cross_entropy_loss(&logits_cont, &targets)
        .expect("invariant: contiguous test inputs have valid cross-entropy shapes and targets");
    assert!((loss_ce.tensor.as_slice()[0] - loss_ce_cont.tensor.as_slice()[0]).abs() < 1e-7);

    loss_ce
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(logits.grad().is_some());
}

#[test]
fn test_sliced_offset_cross_entropy() {
    use coeus_nn::cross_entropy_loss;
    let logits_raw = Tensor::<f64, coeus_core::MoiraiBackend>::from_slice(
        vec![4, 3],
        &[
            99.0, 99.0, 99.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 99.0, 99.0, 99.0,
        ],
    );

    let logits_sliced = logits_raw.slice(&[(1, 3), (0, 3)]);
    assert_eq!(logits_sliced.layout().offset(), 3);
    assert!(logits_sliced.is_contiguous());

    let logits = Var::new(logits_sliced, true);
    let targets = vec![1, 2];

    let loss_ce = cross_entropy_loss(&logits, &targets)
        .expect("invariant: test inputs have valid cross-entropy shapes and targets");
    assert_eq!(loss_ce.tensor.shape(), &[1]);

    let logits_cont = Var::new(
        Tensor::<f64, coeus_core::MoiraiBackend>::from_slice(
            vec![2, 3],
            &[1.0, 2.0, 0.0, 0.0, 2.0, 1.0],
        ),
        true,
    );
    let loss_ce_cont = cross_entropy_loss(&logits_cont, &targets)
        .expect("invariant: contiguous test inputs have valid cross-entropy shapes and targets");
    assert!((loss_ce.tensor.as_slice()[0] - loss_ce_cont.tensor.as_slice()[0]).abs() < 1e-7);

    loss_ce
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(logits.grad().is_some());
    let grad = logits.grad().unwrap();

    loss_ce_cont
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad_cont = logits_cont.grad().unwrap();
    assert_eq!(grad.shape(), &[2, 3]);
    for i in 0..6 {
        assert!((grad.as_slice()[i] - grad_cont.as_slice()[i]).abs() < 1e-7);
    }
}
