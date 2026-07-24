use coeus_autograd::Var;
use coeus_nn::{init, Conv2d, Module};
use coeus_tensor::Tensor;

#[test]
fn test_conv2d_forward_shape() {
    let conv = Conv2d::<f64>::new(3, 8, 3, true);
    let input = Var::new(Tensor::zeros(vec![2, 3, 32, 32]), true);
    let output = conv.forward(&input);

    assert_eq!(output.tensor.shape(), &[2, 8, 30, 30]);

    let params = conv.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_conv2d_forward_no_bias() {
    let conv = Conv2d::<f64>::new(1, 1, 3, false);
    let params = conv.parameters();
    assert_eq!(params.len(), 1);
    assert!(conv.bias.is_none());
}

#[test]
fn test_conv2d_forward_computation() {
    let mut conv = Conv2d::<f64>::new(1, 1, 2, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.0);
    }

    let input = Var::new(
        Tensor::from_slice(
            vec![1, 1, 3, 3],
            &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ),
        true,
    );

    let output = conv.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 1, 2, 2]);

    let out_slice = output.tensor.as_slice();
    assert_eq!(out_slice[0], 12.0);
    assert_eq!(out_slice[1], 16.0);
    assert_eq!(out_slice[2], 24.0);
    assert_eq!(out_slice[3], 28.0);
}

#[test]
fn test_conv2d_backward_gradients_match_reference() {
    let mut conv = Conv2d::<f64>::new(1, 1, 2, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.5);
    }

    let input = Var::new(
        Tensor::from_slice(
            vec![1, 1, 3, 3],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ),
        true,
    );

    let output = conv.forward(&input);
    output.backward();

    let input_grad = input.grad().expect("input gradient must be set");
    assert_eq!(input_grad.shape(), &[1, 1, 3, 3]);
    assert_eq!(
        input_grad.as_slice(),
        &[1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0]
    );

    let weight_grad = conv.weight.grad().expect("weight gradient must be set");
    assert_eq!(weight_grad.shape(), &[1, 1, 2, 2]);
    assert_eq!(weight_grad.as_slice(), &[12.0, 16.0, 24.0, 28.0]);

    if let Some(ref b) = conv.bias {
        let bias_grad = b.grad().expect("bias gradient must be set");
        assert_eq!(bias_grad.shape(), &[1]);
        assert_eq!(bias_grad.as_slice(), &[4.0]);
    }
}

#[test]
fn test_conv2d_with_padding() {
    let mut conv = Conv2d::<f64>::with_params(1, 1, 3, 1, 1, 1, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.0);
    }

    let input = Var::new(Tensor::zeros(vec![1, 1, 4, 4]), true);
    let output = conv.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 1, 4, 4]);
}

#[test]
fn test_conv2d_with_stride() {
    let mut conv = Conv2d::<f64>::with_params(1, 1, 3, 2, 0, 1, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.0);
    }

    let input = Var::new(Tensor::zeros(vec![1, 1, 7, 7]), true);
    let output = conv.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 1, 3, 3]);
}
