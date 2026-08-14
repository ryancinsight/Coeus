#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
use coeus_autograd::Var;
use coeus_nn::{init, AvgPool3d, BatchNorm3d, Conv3d, MaxPool3d, Module};
use coeus_tensor::Tensor;

#[test]
fn test_conv3d_forward_shape() {
    let conv = Conv3d::<f64>::new(2, 4, 3, true);
    let input = Var::new(Tensor::zeros(vec![2, 2, 8, 8, 8]), true);
    let output = conv.forward(&input).expect("valid Conv3d input");

    assert_eq!(output.tensor.shape(), &[2, 4, 6, 6, 6]);

    let params = conv.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_conv3d_forward_computation() {
    let mut conv = Conv3d::<f64>::new(1, 1, 2, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.0);
    }

    let input = Var::new(
        Tensor::from_slice(
            vec![1, 1, 2, 2, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ),
        true,
    );

    let output = conv.forward(&input).expect("valid Conv3d input");
    assert_eq!(output.tensor.shape(), &[1, 1, 1, 1, 1]);

    let out_slice = output.tensor.as_slice();
    assert_eq!(out_slice[0], 36.0); // Sum of 1..8
}

#[test]
fn test_conv3d_backward_gradients_match_reference() {
    let mut conv = Conv3d::<f64>::new(1, 1, 2, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.5);
    }

    let input = Var::new(
        Tensor::from_slice(
            vec![1, 1, 2, 2, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ),
        true,
    );

    let output = conv.forward(&input).expect("valid Conv3d input");
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    let input_grad = input.grad().expect("input gradient must be set");
    assert_eq!(input_grad.shape(), &[1, 1, 2, 2, 2]);
    assert_eq!(input_grad.as_slice(), &[1.0; 8]);

    let weight_grad = conv.weight.grad().expect("weight gradient must be set");
    assert_eq!(weight_grad.shape(), &[1, 1, 2, 2, 2]);
    assert_eq!(
        weight_grad.as_slice(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );

    if let Some(ref b) = conv.bias {
        let bias_grad = b.grad().expect("bias gradient must be set");
        assert_eq!(bias_grad.shape(), &[1]);
        assert_eq!(bias_grad.as_slice(), &[1.0]);
    }
}

#[test]
fn test_batchnorm3d_forward_and_backward() {
    let bn = BatchNorm3d::<f64>::new(2, 1e-5, 0.1);
    let input = Var::new(
        Tensor::from_slice(
            vec![1, 2, 2, 2, 2],
            &[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // channel 0
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, // channel 1
            ],
        ),
        true,
    );

    let output = bn.forward(&input).expect("valid BatchNorm3d input");
    assert_eq!(output.tensor.shape(), &[1, 2, 2, 2, 2]);

    // Identical values per channel → normalized to 0 → output = gamma*0 + beta = 0.
    for &val in output.tensor.as_slice() {
        assert!(val.abs() < 1e-7);
    }

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(input.grad().is_some());
    assert!(bn.weight.grad().is_some());
    assert!(bn.bias.grad().is_some());
}

#[test]
fn test_max_pool3d_forward_and_backward() {
    let pool = MaxPool3d::<f64>::with_params(2, 2, 0, 1);
    let input = Var::new(
        Tensor::from_slice(
            vec![1, 1, 2, 2, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ),
        true,
    );

    let output = pool.forward(&input).expect("valid MaxPool3d input");
    assert_eq!(output.tensor.shape(), &[1, 1, 1, 1, 1]);
    assert_eq!(output.tensor.as_slice(), &[8.0]);

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert_eq!(
        input.grad().unwrap().as_slice(),
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn test_avg_pool3d_forward_and_backward() {
    let pool = AvgPool3d::<f64>::with_params(2, 2, 0, 1);
    let input = Var::new(
        Tensor::from_slice(
            vec![1, 1, 2, 2, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ),
        true,
    );

    let output = pool.forward(&input).expect("valid AvgPool3d input");
    assert_eq!(output.tensor.shape(), &[1, 1, 1, 1, 1]);
    assert_eq!(output.tensor.as_slice(), &[4.5]);

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = input.grad().unwrap();
    for &g in grad.as_slice() {
        assert!((g - 0.125).abs() < 1e-7);
    }
}
