use coeus_autograd::Var;
use coeus_nn::{cross_entropy_loss, gelu, init, mse_loss, relu, sigmoid, tanh, Linear, Module};
use coeus_tensor::Tensor;

#[test]
fn test_linear_layer() {
    let mut layer = Linear::<f64>::new(3, 2, true);
    init::constant(&mut layer.weight, 1.0);
    if let Some(ref mut b) = layer.bias {
        init::constant(b, 0.5);
    }

    let input = Var::new(Tensor::from_slice(vec![1, 3], &[1.0f64, 2.0, 3.0]), true);
    let output = layer.forward(&input);

    assert_eq!(output.tensor.shape(), &[1, 2]);
    assert_eq!(output.tensor.as_slice(), &[6.5, 6.5]);

    output.backward();
    assert!(input.grad().is_some());
    assert!(layer.weight.grad().is_some());
    if let Some(ref b) = layer.bias {
        assert!(b.grad().is_some());
    }
}

#[test]
fn test_activations() {
    let input: Var<f64> = Var::new(
        Tensor::from_slice(vec![4], &[-2.0f64, -0.5, 0.5, 2.0]),
        true,
    );

    // ReLU
    let out_relu = relu(&input);
    assert_eq!(out_relu.tensor.as_slice(), &[0.0, 0.0, 0.5, 2.0]);
    out_relu.backward();
    assert_eq!(input.grad().unwrap().as_slice(), &[0.0, 0.0, 1.0, 1.0]);

    // Sigmoid
    input.zero_grad();
    let out_sig = sigmoid(&input);
    assert!((out_sig.tensor.as_slice()[2] - 0.622459f64).abs() < 1e-4);

    // Tanh
    input.zero_grad();
    let out_tanh = tanh(&input);
    assert!((out_tanh.tensor.as_slice()[2] - 0.462117f64).abs() < 1e-4);

    // GeLU
    input.zero_grad();
    let out_gelu = gelu(&input);
    assert!(out_gelu.tensor.as_slice()[0] < 0.1);
}

#[test]
fn test_losses() {
    let pred: Var<f64> = Var::new(Tensor::from_slice(vec![2], &[0.5f64, 1.5]), true);
    let target = Var::new(Tensor::from_slice(vec![2], &[1.0f64, 1.0]), false);

    // MSE
    let loss_mse = mse_loss(&pred, &target);
    assert_eq!(loss_mse.tensor.as_slice(), &[0.25]);
    loss_mse.backward();
    assert!(pred.grad().is_some());

    // Cross entropy
    let logits: Var<f64> = Var::new(
        Tensor::from_slice(vec![2, 3], &[1.0f64, 2.0, 0.0, 0.0, 2.0, 1.0]),
        true,
    );
    let targets = vec![1, 2];
    let loss_ce = cross_entropy_loss(&logits, &targets);
    assert_eq!(loss_ce.tensor.shape(), &[1]);
    loss_ce.backward();
    assert!(logits.grad().is_some());
}

#[test]
fn test_initializers() {
    let mut weight = Var::<f64>::new(Tensor::zeros(vec![1000]), true);

    init::normal(&mut weight, 5.0, 2.0);
    let w_slice = weight.tensor.as_slice();
    let sum: f64 = w_slice.iter().sum();
    let mean = sum / w_slice.len() as f64;
    assert!((mean - 5.0).abs() < 0.2);

    init::xavier_uniform(&mut weight, 100, 100);
    let limit = (6.0f64 / 200.0).sqrt();
    for &val in weight.tensor.as_slice() {
        assert!(val >= -limit && val <= limit);
    }
}
