use coeus_tensor::{Tensor, Transpose};
use coeus_autograd::Var;
use coeus_nn::{
    Linear, Module, Conv1d, Conv2d, Conv3d,
    BatchNorm3d, MaxPool3d, AvgPool3d,
    relu, sigmoid, tanh, gelu,
    mse_loss, cross_entropy_loss,
    init,
};

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
    let input: Var<f64> = Var::new(Tensor::from_slice(vec![4], &[-2.0f64, -0.5, 0.5, 2.0]), true);

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
    let logits: Var<f64> = Var::new(Tensor::from_slice(vec![2, 3], &[
        1.0f64, 2.0, 0.0,
        0.0, 2.0, 1.0,
    ]), true);
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
    let w_slice_x = weight.tensor.as_slice();
    for &val in w_slice_x {
        assert!(val >= -limit && val <= limit);
    }
}

// ── Conv2d tests ──

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

    let input = Var::new(Tensor::from_slice(vec![1, 1, 3, 3], &[
        1.0f64, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ]), true);

    let output = conv.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 1, 2, 2]);

    let out_slice = output.tensor.as_slice();
    assert_eq!(out_slice[0], 12.0);
    assert_eq!(out_slice[1], 16.0);
    assert_eq!(out_slice[2], 24.0);
    assert_eq!(out_slice[3], 28.0);
}

#[test]
fn test_conv2d_backward_gradients_exist() {
    let mut conv = Conv2d::<f64>::new(1, 1, 2, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.5);
    }

    let input = Var::new(Tensor::from_slice(vec![1, 1, 3, 3], &[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    ]), true);

    let output = conv.forward(&input);
    output.backward();

    assert!(input.grad().is_some());
    assert!(conv.weight.grad().is_some());
    if let Some(ref b) = conv.bias {
        assert!(b.grad().is_some());
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

// ── Conv1d tests ──

#[test]
fn test_conv1d_forward_shape() {
    let conv = Conv1d::<f64>::new(3, 8, 3, true);
    let input = Var::new(Tensor::zeros(vec![2, 3, 32]), true);
    let output = conv.forward(&input);

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

    let input = Var::new(Tensor::from_slice(vec![1, 1, 4], &[
        1.0f64, 2.0, 3.0, 4.0,
    ]), true);

    let output = conv.forward(&input);
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

    let input = Var::new(Tensor::from_slice(vec![1, 2, 3], &[
        1.0f64, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ]), true);

    let output = conv.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 1, 2]);

    let out_slice = output.tensor.as_slice();
    assert_eq!(out_slice[0], 12.0);
    assert_eq!(out_slice[1], 16.0);
}

#[test]
fn test_conv1d_backward_gradients_exist() {
    let mut conv = Conv1d::<f64>::new(1, 1, 2, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.5);
    }

    let input = Var::new(Tensor::from_slice(vec![1, 1, 4], &[
        1.0, 2.0, 3.0, 4.0,
    ]), true);

    let output = conv.forward(&input);
    output.backward();

    assert!(input.grad().is_some());
    assert!(conv.weight.grad().is_some());
    if let Some(ref b) = conv.bias {
        assert!(b.grad().is_some());
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
    let output = conv.forward(&input);
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
    let output = conv.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 1, 3]);
}

#[test]
fn test_conv1d_with_dilation() {
    let mut conv = Conv1d::<f64>::with_params(1, 1, 2, 1, 0, 2, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.0);
    }

    let input = Var::new(Tensor::from_slice(vec![1, 1, 7], &[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    ]), true);

    let output = conv.forward(&input);
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
    let logits_raw = Tensor::<f64, coeus_core::MoiraiBackend>::from_slice(vec![3, 2], &[
        1.0f64, 0.0,
        2.0, 2.0,
        0.0, 1.0,
    ]);
    let logits_t = logits_raw.transpose(); // shape [2, 3], non-contiguous
    let logits = Var::new(logits_t, true);
    let targets = vec![1, 2];

    let loss_ce = cross_entropy_loss(&logits, &targets);
    assert_eq!(loss_ce.tensor.shape(), &[1]);

    // Check that loss value matches expected computation on contiguous version
    let logits_cont = Var::new(Tensor::<f64, coeus_core::MoiraiBackend>::from_slice(vec![2, 3], &[
        1.0f64, 2.0, 0.0,
        0.0, 2.0, 1.0,
    ]), true);
    let loss_ce_cont = cross_entropy_loss(&logits_cont, &targets);
    assert!((loss_ce.tensor.as_slice()[0] - loss_ce_cont.tensor.as_slice()[0]).abs() < 1e-7);

    loss_ce.backward();
    assert!(logits.grad().is_some());
}

#[test]
fn test_sliced_offset_cross_entropy() {
    let logits_raw = Tensor::<f64, coeus_core::MoiraiBackend>::from_slice(vec![4, 3], &[
        // Row 0: offset 0 (ignored by slice)
        99.0, 99.0, 99.0,
        // Row 1: offset 3
        1.0, 2.0, 0.0,
        // Row 2: offset 6
        0.0, 2.0, 1.0,
        // Row 3: offset 9 (ignored by slice)
        99.0, 99.0, 99.0,
    ]);
    
    // Slice logits_raw to [2, 3] starting at row 1 (offset 3)
    let logits_sliced = logits_raw.slice(&[(1, 3), (0, 3)]);
    assert_eq!(logits_sliced.layout().offset(), 3);
    assert!(logits_sliced.is_contiguous());

    let logits = Var::new(logits_sliced, true);
    let targets = vec![1, 2];

    let loss_ce = cross_entropy_loss(&logits, &targets);
    assert_eq!(loss_ce.tensor.shape(), &[1]);

    // Check against contiguous version
    let logits_cont = Var::new(Tensor::<f64, coeus_core::MoiraiBackend>::from_slice(vec![2, 3], &[
        1.0, 2.0, 0.0,
        0.0, 2.0, 1.0,
    ]), true);
    let loss_ce_cont = cross_entropy_loss(&logits_cont, &targets);
    assert!((loss_ce.tensor.as_slice()[0] - loss_ce_cont.tensor.as_slice()[0]).abs() < 1e-7);

    loss_ce.backward();
    assert!(logits.grad().is_some());
    let grad = logits.grad().unwrap();
    
    // Check gradient against contiguous version
    loss_ce_cont.backward();
    let grad_cont = logits_cont.grad().unwrap();
    assert_eq!(grad.shape(), &[2, 3]);
    for i in 0..6 {
        assert!((grad.as_slice()[i] - grad_cont.as_slice()[i]).abs() < 1e-7);
    }
}

// ── Conv3d tests ──

#[test]
fn test_conv3d_forward_shape() {
    let conv = Conv3d::<f64>::new(2, 4, 3, true);
    let input = Var::new(Tensor::zeros(vec![2, 2, 8, 8, 8]), true);
    let output = conv.forward(&input);

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

    let input = Var::new(Tensor::from_slice(vec![1, 1, 2, 2, 2], &[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ]), true);

    let output = conv.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 1, 1, 1, 1]);

    let out_slice = output.tensor.as_slice();
    assert_eq!(out_slice[0], 36.0); // Sum of 1..8
}

#[test]
fn test_conv3d_backward_gradients_exist() {
    let mut conv = Conv3d::<f64>::new(1, 1, 2, true);
    init::constant(&mut conv.weight, 1.0);
    if let Some(ref mut b) = conv.bias {
        init::constant(b, 0.5);
    }

    let input = Var::new(Tensor::from_slice(vec![1, 1, 2, 2, 2], &[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ]), true);

    let output = conv.forward(&input);
    output.backward();

    assert!(input.grad().is_some());
    assert!(conv.weight.grad().is_some());
    if let Some(ref b) = conv.bias {
        assert!(b.grad().is_some());
    }
}

// ── BatchNorm3d tests ──

#[test]
fn test_batchnorm3d_forward_and_backward() {
    let bn = BatchNorm3d::<f64>::new(2, 1e-5, 0.1);
    let input = Var::new(Tensor::from_slice(vec![1, 2, 2, 2, 2], &[
        // channel 0
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        // channel 1
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    ]), true);

    let output = bn.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 2, 2, 2, 2]);

    // All elements in a channel are identical, so their mean is the element and variance is 0.
    // Normalized values will be 0.
    // Output = gamma * 0 + beta = beta = 0.
    let out_slice = output.tensor.as_slice();
    for &val in out_slice {
        assert!(val.abs() < 1e-7);
    }

    output.backward();
    assert!(input.grad().is_some());
    assert!(bn.weight.grad().is_some());
    assert!(bn.bias.grad().is_some());
}

// ── Pool3d tests ──

#[test]
fn test_max_pool3d_forward_and_backward() {
    let pool = MaxPool3d::<f64>::with_params(2, 2, 0, 1);
    let input = Var::new(Tensor::from_slice(vec![1, 1, 2, 2, 2], &[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ]), true);

    let output = pool.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 1, 1, 1, 1]);
    assert_eq!(output.tensor.as_slice(), &[8.0]);

    output.backward();
    assert_eq!(input.grad().unwrap().as_slice(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn test_avg_pool3d_forward_and_backward() {
    let pool = AvgPool3d::<f64>::with_params(2, 2, 0, 1);
    let input = Var::new(Tensor::from_slice(vec![1, 1, 2, 2, 2], &[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ]), true);

    let output = pool.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 1, 1, 1, 1]);
    assert_eq!(output.tensor.as_slice(), &[4.5]); // mean of 1..8

    output.backward();
    let grad = input.grad().unwrap();
    for &g in grad.as_slice() {
        assert!((g - 0.125).abs() < 1e-7);
    }
}// ── GroupNorm tests ──

#[test]
fn test_groupnorm_forward_and_backward() {
    use coeus_nn::normalization::groupnorm::GroupNorm;

    // G=2, C=4: each group contains 2 channels.
    // Input: [N=1, C=4, L=3] — interpreted internally as [N*G, group_size*L].
    let gn = GroupNorm::<f64, coeus_core::MoiraiBackend, 2>::new(4, 1e-5);
    let input = Var::new(
        Tensor::from_slice(vec![1, 4, 3], &[
            1.0f64, 2.0, 3.0,   // ch 0
            4.0, 5.0, 6.0,      // ch 1
            7.0, 8.0, 9.0,      // ch 2
            10.0, 11.0, 12.0,   // ch 3
        ]),
        true,
    );
    let output = gn.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 4, 3]);

    // The sum of normalized values within each group should be ≈ 0 (mean-subtracted)
    // before the affine transform (weight=1, bias=0 by default).
    let out_slice = output.tensor.as_slice();
    // Group 0 spans channels 0-1, group 1 spans channels 2-3.
    // Mean of group 0 = mean([1..6]) = 3.5; after normalization output sums to ~0.
    let group0_sum: f64 = out_slice[..6].iter().sum();
    assert!(group0_sum.abs() < 1e-5, "group0_sum={group0_sum}");

    output.backward();
    assert!(input.grad().is_some());
    assert!(gn.weight.grad().is_some());
    assert!(gn.bias.grad().is_some());
}

#[test]
fn test_groupnorm_g1_is_layernorm() {
    use coeus_nn::normalization::groupnorm::GroupNorm;

    // G=1 with C features is equivalent to LayerNorm over all C*spatial dimensions.
    let gn = GroupNorm::<f64, coeus_core::MoiraiBackend, 1>::new(4, 1e-5);
    let input = Var::new(
        Tensor::from_slice(vec![2, 4], &[
            1.0f64, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]),
        true,
    );
    let output = gn.forward(&input);
    assert_eq!(output.tensor.shape(), &[2, 4]);

    output.backward();
    assert!(input.grad().is_some());
}

// ── InstanceNorm tests ──

#[test]
fn test_instancenorm1d_forward_and_backward() {
    use coeus_nn::normalization::instancenorm::InstanceNorm1d;

    // Input: [N=2, C=3, L=4].
    let inst = InstanceNorm1d::<f64, coeus_core::MoiraiBackend>::new(3, 1e-5);
    let input = Var::new(Tensor::zeros(vec![2, 3, 4]), true);
    let output = inst.forward(&input);
    assert_eq!(output.tensor.shape(), &[2, 3, 4]);

    // All-zero input → normalized output is all-zero (0 / (0 + eps)^0.5 * 1 + 0).
    let out_slice = output.tensor.as_slice();
    for &v in out_slice {
        assert!(v.abs() < 1e-5);
    }

    output.backward();
    assert!(input.grad().is_some());
    assert!(inst.weight.grad().is_some());
    assert!(inst.bias.grad().is_some());
}

#[test]
fn test_instancenorm1d_non_constant_backward() {
    use coeus_nn::normalization::instancenorm::InstanceNorm1d;

    let inst = InstanceNorm1d::<f64, coeus_core::MoiraiBackend>::new(2, 1e-5);
    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 4], &[
            1.0f64, 2.0, 3.0, 4.0,  // ch 0 → mean=2.5, var=1.25
            0.0, 0.5, 1.0, 1.5,     // ch 1 → mean=0.75, var=0.3125
        ]),
        true,
    );
    let output = inst.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 2, 4]);

    // Each channel normalised to zero mean.
    let s = output.tensor.as_slice();
    let mean0: f64 = s[..4].iter().sum::<f64>() / 4.0;
    assert!(mean0.abs() < 1e-5);

    output.backward();
    assert!(input.grad().is_some());
}

#[test]
fn test_instancenorm2d_forward_and_backward() {
    use coeus_nn::normalization::instancenorm::InstanceNorm2d;

    // Input: [N=1, C=2, H=3, W=3].
    let inst = InstanceNorm2d::<f64, coeus_core::MoiraiBackend>::new(2, 1e-5);
    let data: Vec<f64> = (0..18).map(|i| i as f64).collect();
    let input = Var::new(Tensor::from_slice(vec![1, 2, 3, 3], &data), true);
    let output = inst.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 2, 3, 3]);

    // Each (n, c) slice has mean 0 after normalization.
    let s = output.tensor.as_slice();
    let mean0: f64 = s[..9].iter().sum::<f64>() / 9.0;
    let mean1: f64 = s[9..].iter().sum::<f64>() / 9.0;
    assert!(mean0.abs() < 1e-5, "mean0={mean0}");
    assert!(mean1.abs() < 1e-5, "mean1={mean1}");

    output.backward();
    assert!(input.grad().is_some());
    assert!(inst.weight.grad().is_some());
    assert!(inst.bias.grad().is_some());
}

// ── MultiHeadAttention tests ──

#[test]
fn test_mha_self_attention_shape() {
    use coeus_nn::attention::mha::MultiHeadAttention;
    use coeus_autograd::NullMask;

    // H=4 heads, d_model=8 → d_head=2.
    // Input: [batch=1, seq=5, d_model=8].
    let mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 4, NullMask>::new(8, true);
    let input = Var::new(Tensor::zeros(vec![1, 5, 8]), true);
    let output = mha.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 5, 8]);
}

#[test]
fn test_mha_cross_attention_shape() {
    use coeus_nn::attention::mha::MultiHeadAttention;
    use coeus_autograd::NullMask;

    let mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 2, NullMask>::new(4, true);
    let query = Var::new(Tensor::zeros(vec![1, 3, 4]), true);
    let key   = Var::new(Tensor::zeros(vec![1, 5, 4]), false);
    let value = Var::new(Tensor::zeros(vec![1, 5, 4]), false);
    let output = mha.forward_cross(&query, &key, &value, None);
    assert_eq!(output.tensor.shape(), &[1, 3, 4]);
}

#[test]
fn test_mha_backward_gradients_exist() {
    use coeus_nn::attention::mha::MultiHeadAttention;
    use coeus_autograd::NullMask;

    let mha = MultiHeadAttention::<f64, coeus_core::MoiraiBackend, 2, NullMask>::new(4, true);
    let input = Var::new(Tensor::from_slice(vec![1, 2, 4], &[
        0.1f64, 0.2, 0.3, 0.4,
        0.5, 0.6, 0.7, 0.8,
    ]), true);
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

// ── log_softmax tests ──

#[test]
fn test_log_softmax_probabilities() {
    // exp(log_softmax(x)) must sum to 1 along the softmax axis.
    let input: Var<f64> = Var::new(
        Tensor::from_slice(vec![2, 4], &[
            1.0f64, 2.0, 3.0, 4.0,
            0.5, 1.5, 2.5, 3.5,
        ]),
        true,
    );
    let log_probs = coeus_autograd::log_softmax(&input, 1);
    assert_eq!(log_probs.tensor.shape(), &[2, 4]);

    let s = log_probs.tensor.as_slice();
    // Row 0: exp of each element, sum must be ≈ 1.
    let row0_sum: f64 = s[..4].iter().map(|x| x.exp()).sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "row0_sum={row0_sum}");
    let row1_sum: f64 = s[4..].iter().map(|x| x.exp()).sum();
    assert!((row1_sum - 1.0).abs() < 1e-5, "row1_sum={row1_sum}");
}

#[test]
fn test_log_softmax_backward() {
    let input: Var<f64> = Var::new(
        Tensor::from_slice(vec![1, 3], &[1.0f64, 2.0, 3.0]),
        true,
    );
    let log_probs = coeus_autograd::log_softmax(&input, 1);
    // Use mse_loss as a scalar reducer to drive backward.
    let target: Var<f64> = Var::new(
        Tensor::from_slice(vec![1, 3], &[0.0f64, 1.0, 0.0]),
        false,
    );
    let loss = coeus_nn::loss::mse_loss(&log_probs, &target);
    loss.backward();
    assert!(input.grad().is_some());
    let g = input.grad().unwrap();
    assert_eq!(g.shape(), &[1, 3]);
    // Gradient must be non-zero for at least the target element.
    assert!(g.as_slice().iter().any(|&v| v.abs() > 1e-7));
}

// ── cat tests ──

#[test]
fn test_cat_forward_shape() {
    let a = Var::<f64>::new(Tensor::zeros(vec![2, 3]), true);
    let b = Var::<f64>::new(Tensor::zeros(vec![2, 4]), true);
    let out = coeus_autograd::cat(&[&a, &b], 1);
    assert_eq!(out.tensor.shape(), &[2, 7]);
}

#[test]
fn test_cat_backward_gradient_split() {
    // cat of [2, 3] and [2, 4] along dim=1 → [2, 7].
    // After backward, gradient of `a` must cover only its 3 columns,
    // gradient of `b` must cover only its 4 columns.
    let a_data = vec![1.0f64; 6];
    let b_data = vec![2.0f64; 8];
    let a = Var::<f64>::new(Tensor::from_slice(vec![2, 3], &a_data), true);
    let b = Var::<f64>::new(Tensor::from_slice(vec![2, 4], &b_data), true);
    let out = coeus_autograd::cat(&[&a, &b], 1);
    out.backward();
    assert!(a.grad().is_some());
    assert!(b.grad().is_some());
    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();
    assert_eq!(ga.shape(), &[2, 3]);
    assert_eq!(gb.shape(), &[2, 4]);
}

#[test]
fn test_cat_along_dim0() {
    let a = Var::<f64>::new(Tensor::zeros(vec![2, 5]), true);
    let b = Var::<f64>::new(Tensor::zeros(vec![3, 5]), true);
    let c = Var::<f64>::new(Tensor::zeros(vec![1, 5]), true);
    let out = coeus_autograd::cat(&[&a, &b, &c], 0);
    assert_eq!(out.tensor.shape(), &[6, 5]);
    out.backward();
    assert_eq!(a.grad().unwrap().shape(), &[2, 5]);
    assert_eq!(b.grad().unwrap().shape(), &[3, 5]);
    assert_eq!(c.grad().unwrap().shape(), &[1, 5]);
}

// ── split tests ──

#[test]
fn test_split_even_chunks() {
    let input = Var::<f64>::new(Tensor::zeros(vec![1, 6]), true);
    let chunks = coeus_autograd::split(&input, 2, 1);
    assert_eq!(chunks.len(), 3);
    for ch in &chunks {
        assert_eq!(ch.tensor.shape(), &[1, 2]);
    }
}

#[test]
fn test_split_remainder_chunk() {
    let input = Var::<f64>::new(Tensor::zeros(vec![1, 7]), true);
    let chunks = coeus_autograd::split(&input, 3, 1);
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].tensor.shape(), &[1, 3]);
    assert_eq!(chunks[1].tensor.shape(), &[1, 3]);
    assert_eq!(chunks[2].tensor.shape(), &[1, 1]); // remainder
}

#[test]
fn test_split_backward_accumulation() {
    // split [1, 4] into 2 chunks of 2. Each chunk backward accumulates into its slice.
    let input = Var::<f64>::new(
        Tensor::from_slice(vec![1, 4], &[1.0f64, 2.0, 3.0, 4.0]),
        true,
    );
    let chunks = coeus_autograd::split(&input, 2, 1);
    // Drive loss from chunk 0 only.
    let target = Var::<f64>::new(Tensor::from_slice(vec![1, 2], &[0.0f64, 0.0]), false);
    let loss = coeus_nn::loss::mse_loss(&chunks[0], &target);
    loss.backward();
    let g = input.grad().unwrap();
    assert_eq!(g.shape(), &[1, 4]);
    // Gradient in columns 2-3 (chunk 1) must be zero (not driven).
    assert_eq!(g.as_slice()[2], 0.0);
    assert_eq!(g.as_slice()[3], 0.0);
    // Gradient in columns 0-1 must be non-zero.
    assert!(g.as_slice()[0].abs() > 0.0 || g.as_slice()[1].abs() > 0.0);
}

// ── RotaryEmbedding tests ──

#[test]
fn test_rope_forward_shape() {
    use coeus_nn::positional::RotaryEmbedding;

    // max_len = 16, d_head = 4.
    // Input: [batch = 2, seq_len = 4, num_heads = 3, d_head = 4].
    let rope = RotaryEmbedding::<f64, coeus_core::MoiraiBackend>::new(16, 4, 10000.0);
    let input = Var::new(Tensor::zeros(vec![2, 4, 3, 4]), true);
    let output = rope.forward(&input);
    assert_eq!(output.tensor.shape(), &[2, 4, 3, 4]);
}

#[test]
fn test_rope_backward() {
    use coeus_nn::positional::RotaryEmbedding;

    let rope = RotaryEmbedding::<f64, coeus_core::MoiraiBackend>::new(16, 4, 10000.0);
    let input = Var::new(Tensor::from_slice(vec![1, 2, 1, 4], &[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ]), true);
    let output = rope.forward(&input);
    output.backward();
    assert!(input.grad().is_some());
    let g = input.grad().unwrap();
    assert_eq!(g.shape(), &[1, 2, 1, 4]);
}

#[test]
fn test_rope_numerical_correctness() {
    use coeus_nn::positional::RotaryEmbedding;

    // For pos = 1, theta_0 = 1.0 (since base^-0 = 1), angle = 1.0.
    // input is [1, 2, 1, 2]
    // pos 0: [x1, y1] -> angle = 0. x1_rot = x1, y1_rot = y1
    // pos 1: [x2, y2] -> angle = 1. x2_rot = x2*cos(1) - y2*sin(1), y2_rot = y2*cos(1) + x2*sin(1)
    let rope = RotaryEmbedding::<f64, coeus_core::MoiraiBackend>::new(4, 2, 1.0);
    let input = Var::new(Tensor::from_slice(vec![1, 2, 1, 2], &[
        1.0, 2.0, // pos 0
        3.0, 4.0, // pos 1
    ]), false);

    let output = rope.forward(&input);
    let out_slice = output.tensor.as_slice();

    // pos 0
    assert!((out_slice[0] - 1.0).abs() < 1e-6);
    assert!((out_slice[1] - 2.0).abs() < 1e-6);

    // pos 1
    let cos1 = 1.0_f64.cos();
    let sin1 = 1.0_f64.sin();
    let expected_x2 = 3.0 * cos1 - 4.0 * sin1;
    let expected_y2 = 4.0 * cos1 + 3.0 * sin1;

    assert!((out_slice[2] - expected_x2).abs() < 1e-6, "expected: {expected_x2}, got: {}", out_slice[2]);
    assert!((out_slice[3] - expected_y2).abs() < 1e-6, "expected: {expected_y2}, got: {}", out_slice[3]);
}

#[test]
fn test_general_transpose_autograd() {
    let input = Var::<f64>::new(
        Tensor::from_slice(vec![2, 3, 4], &(0..24).map(|i| i as f64).collect::<Vec<f64>>()),
        true,
    );
    // Swap dim 0 and dim 2: [2, 3, 4] -> [4, 3, 2]
    let transposed = coeus_autograd::transpose(&input, 0, 2);
    assert_eq!(transposed.tensor.shape(), &[4, 3, 2]);

    // Backward pass
    let sum = coeus_autograd::sum(&transposed);
    sum.backward();
    assert!(input.grad().is_some());
    let g = input.grad().unwrap();
    assert_eq!(g.shape(), &[2, 3, 4]);
    for &val in g.as_slice() {
        assert!((val - 1.0).abs() < 1e-6);
    }
}


