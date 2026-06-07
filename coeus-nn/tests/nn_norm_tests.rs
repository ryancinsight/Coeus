use coeus_tensor::Tensor;
use coeus_autograd::Var;
use coeus_nn::{
    Module, LayerNorm, RMSNorm, Dropout,
    BatchNorm1d,
    softmax, Softmax, BatchNorm2d,
    init, AvgPool2d, MaxPool2d,
};

#[test]
fn test_layernorm() {
    let mut ln = LayerNorm::<f64>::new(4, 1e-5);
    init::constant(&mut ln.weight, 1.0);
    init::constant(&mut ln.bias, 0.0);

    let input = Var::new(Tensor::from_slice(vec![2, 4], &[
        1.0f64, 2.0, 3.0, 4.0,
        10.0, 20.0, 30.0, 40.0,
    ]), true);

    let output = ln.forward(&input);
    assert_eq!(output.tensor.shape(), &[2, 4]);

    // Output elements for each batch should have mean ~0 and std ~1
    let out_slice = output.tensor.as_slice();
    for i in 0..2 {
        let offset = i * 4;
        let mut mean = 0.0f64;
        for j in 0..4 {
            mean += out_slice[offset + j];
        }
        mean /= 4.0;
        assert!(mean.abs() < 1e-5);
    }

    // Test backward pass
    output.backward();
    assert!(input.grad().is_some());
    assert!(ln.weight.grad().is_some());
    assert!(ln.bias.grad().is_some());
}

#[test]
fn test_rmsnorm() {
    let mut rms = RMSNorm::<f64>::new(3, 1e-5);
    init::constant(&mut rms.weight, 1.0);

    let input = Var::new(Tensor::from_slice(vec![1, 3], &[1.0f64, 2.0, 3.0]), true);
    let output = rms.forward(&input);

    assert_eq!(output.tensor.shape(), &[1, 3]);

    output.backward();
    assert!(input.grad().is_some());
    assert!(rms.weight.grad().is_some());
}

#[test]
fn test_dropout() {
    let mut do_layer = Dropout::new(0.5);
    let input: Var<f64> = Var::new(Tensor::ones(vec![100]), true);

    // Evaluation mode: no dropout, output should be identical
    do_layer.set_training(false);
    let out_eval = do_layer.forward(&input);
    assert_eq!(out_eval.tensor.as_slice(), input.tensor.as_slice());

    // Training mode: elements should be dropped or scaled by 2.0
    do_layer.set_training(true);
    let out_train = do_layer.forward(&input);
    let o_slice = out_train.tensor.as_slice();

    let zero_count = o_slice.iter().filter(|&&x| x == 0.0).count();
    let scale_count = o_slice.iter().filter(|&&x| x == 2.0).count();

    assert!(zero_count > 0);
    assert!(scale_count > 0);
    assert_eq!(zero_count + scale_count, 100);

    // Backward pass should propagate through non-zeroed masks
    out_train.backward();
    assert!(input.grad().is_some());
}

#[test]
fn test_batchnorm1d_forward_shape() {
    let bn = BatchNorm1d::<f64>::new(4, 1e-5, 0.1);
    let input = Var::new(Tensor::zeros(vec![2, 4, 10]), true);
    let output = bn.forward(&input);

    assert_eq!(output.tensor.shape(), &[2, 4, 10]);

    let params = bn.parameters();
    assert_eq!(params.len(), 2); // weight + bias
}

#[test]
fn test_batchnorm1d_backward_gradients_exist() {
    let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1);
    let input = Var::new(Tensor::from_slice(vec![2, 2, 3], &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ]), true);

    let output = bn.forward(&input);
    output.backward();

    assert!(input.grad().is_some());
    assert!(bn.weight.grad().is_some());
    assert!(bn.bias.grad().is_some());
}

#[test]
fn test_batchnorm1d_running_stats_update() {
    let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1);

    let input = Var::new(Tensor::from_slice(vec![1, 2, 4], &[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ]), false);

    let rm_before = bn.running_mean.borrow().clone();
    assert_eq!(rm_before.as_slice()[0], 0.0);
    assert_eq!(rm_before.as_slice()[1], 0.0);

    let _ = bn.forward(&input);

    let rm_after = bn.running_mean.borrow();
    assert!((rm_after.as_slice()[0] - 0.25).abs() < 1e-6);
    assert!((rm_after.as_slice()[1] - 0.65).abs() < 1e-6);
}

#[test]
fn test_batchnorm2d_forward_shape() {
    let bn = BatchNorm2d::<f64>::new(4, 1e-5, 0.1);
    let input = Var::new(Tensor::zeros(vec![2, 4, 3, 3]), true);
    let output = bn.forward(&input);

    assert_eq!(output.tensor.shape(), &[2, 4, 3, 3]);

    let params = bn.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_batchnorm2d_backward_gradients_exist() {
    let bn = BatchNorm2d::<f64>::new(2, 1e-5, 0.1);
    let input = Var::new(Tensor::from_slice(vec![1, 2, 2, 2], &[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ]), true);

    let output = bn.forward(&input);
    output.backward();

    assert!(input.grad().is_some());
    assert!(bn.weight.grad().is_some());
    assert!(bn.bias.grad().is_some());
}

#[test]
fn test_batchnorm2d_running_stats_update() {
    let bn = BatchNorm2d::<f64>::new(2, 1e-5, 0.1);

    let input = Var::new(Tensor::from_slice(vec![1, 2, 2, 2], &[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ]), false);

    let rm_before = bn.running_mean.borrow().clone();
    assert_eq!(rm_before.as_slice()[0], 0.0);
    assert_eq!(rm_before.as_slice()[1], 0.0);

    let _ = bn.forward(&input);

    let rm_after = bn.running_mean.borrow();
    assert!((rm_after.as_slice()[0] - 0.25).abs() < 1e-6);
    assert!((rm_after.as_slice()[1] - 0.65).abs() < 1e-6);
}

#[test]
fn test_batchnorm2d_multi_channel_forward() {
    let mut bn = BatchNorm2d::<f64>::new(2, 1e-5, 0.1);
    init::constant(&mut bn.weight, 1.0);
    init::constant(&mut bn.bias, 0.0);

    let input = Var::new(Tensor::from_slice(vec![1, 2, 2, 2], &[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ]), true);

    let output = bn.forward(&input);
    assert_eq!(output.tensor.shape(), &[1, 2, 2, 2]);

    let out_slice = output.tensor.as_slice();
    let ch0_mean: f64 = out_slice[0..4].iter().sum::<f64>() / 4.0;
    assert!(ch0_mean.abs() < 1e-5);
    let ch1_mean: f64 = out_slice[4..8].iter().sum::<f64>() / 4.0;
    assert!(ch1_mean.abs() < 1e-5);

    output.backward();
    assert!(input.grad().is_some());
}

#[test]
fn test_softmax_forward_shapes() {
    let input: Var<f64> = Var::new(Tensor::from_slice(vec![2, 3], &[
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ]), true);

    let output = softmax(&input, -1);
    assert_eq!(output.tensor.shape(), &[2, 3]);
}

#[test]
fn test_softmax_sums_to_one() {
    let input: Var<f64> = Var::new(Tensor::from_slice(vec![1, 3], &[1.0, 2.0, 3.0]), true);
    let output = softmax(&input, -1);

    let s = output.tensor.as_slice();
    let sum: f64 = s.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(s[0] < s[1]);
    assert!(s[1] < s[2]);
}

#[test]
fn test_softmax_backward_uniform_seed() {
    let input: Var<f64> = Var::new(Tensor::from_slice(vec![1, 3], &[1.0, 2.0, 3.0]), true);
    let output = softmax(&input, -1);

    output.backward();
    assert!(input.grad().is_some());

    let g = input.grad().unwrap();
    assert_eq!(g.shape(), &[1, 3]);
    for &v in g.as_slice() {
        assert!((v - 0.0).abs() < 1e-10);
    }
}

#[test]
fn test_softmax_backward_nonuniform_seed() {
    let input: Var<f64> = Var::new(Tensor::from_slice(vec![1, 3], &[1.0, 2.0, 3.0]), true);
    let output = softmax(&input, -1);

    let seed = Tensor::from_slice(vec![1, 3], &[1.0, 0.0, 0.0]);
    output.backward_with_seed(seed);

    let g = input.grad().unwrap();
    assert_eq!(g.shape(), &[1, 3]);

    let y = output.tensor.as_slice();
    let g_slice = g.as_slice();

    let expected_dx0 = y[0] * (1.0 - y[0]);
    let expected_dx1 = -y[1] * y[0];
    let expected_dx2 = -y[2] * y[0];

    assert!((g_slice[0] - expected_dx0).abs() < 1e-6);
    assert!((g_slice[1] - expected_dx1).abs() < 1e-6);
    assert!((g_slice[2] - expected_dx2).abs() < 1e-6);

    assert!(g_slice[0].abs() > 1e-10);
    assert!(g_slice[1].abs() > 1e-10);
}

#[test]
fn test_softmax_module() {
    let sm = Softmax::new(-1);
    let input: Var<f64> = Var::new(Tensor::from_slice(vec![2, 4], &[
        1.0, 2.0, 3.0, 4.0,
        1.0, 1.0, 1.0, 1.0,
    ]), true);

    let output = sm.forward(&input);
    assert_eq!(output.tensor.shape(), &[2, 4]);
    assert_eq!(Module::<f64>::parameters(&sm).len(), 0);

    let s = output.tensor.as_slice();
    let row0_sum: f64 = s[0..4].iter().sum();
    let row1_sum: f64 = s[4..8].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-6);
    assert!((row1_sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_stability() {
    let sm = Softmax::new(-1);
    let input: Var<f64> = Var::new(Tensor::from_slice(vec![1, 3], &[
        800.0, 801.0, 802.0
    ]), true);

    let output = sm.forward(&input);
    let s = output.tensor.as_slice();

    assert!(!s[0].is_nan() && !s[0].is_infinite());
    assert!(!s[1].is_nan() && !s[1].is_infinite());
    assert!(!s[2].is_nan() && !s[2].is_infinite());

    let sum: f64 = s.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    assert!((s[0] - 0.09003).abs() < 1e-4);
    assert!((s[1] - 0.244728).abs() < 1e-4);
    assert!((s[2] - 0.66524).abs() < 1e-4);
}

#[test]
fn test_layernorm_various_shapes() {
    let shapes: Vec<(usize, usize)> = vec![(1, 8), (3, 16), (8, 4), (16, 32)];

    for &(batch, dim) in &shapes {
        let mut ln = LayerNorm::<f64>::new(dim, 1e-5);
        init::constant(&mut ln.weight, 1.0);
        init::constant(&mut ln.bias, 0.0);

        let mut data = Vec::with_capacity(batch * dim);
        for i in 0..(batch * dim) {
            data.push((i + 1) as f64);
        }
        let input = Var::new(Tensor::from_slice(vec![batch, dim], &data), true);

        let output = ln.forward(&input);
        assert_eq!(output.tensor.shape(), &[batch, dim]);

        let out_slice = output.tensor.as_slice();
        for i in 0..batch {
            let offset = i * dim;
            let mut mean = 0.0f64;
            for j in 0..dim {
                mean += out_slice[offset + j];
            }
            mean /= dim as f64;
            assert!(mean.abs() < 1e-5);
        }

        output.backward();
        assert!(input.grad().is_some());
        assert!(ln.weight.grad().is_some());
        assert!(ln.bias.grad().is_some());
        assert_eq!(Module::<f64>::parameters(&ln).len(), 2);
    }
}

#[test]
fn test_layernorm_single_element() {
    let mut ln = LayerNorm::<f64>::new(1, 1e-5);
    init::constant(&mut ln.weight, 2.0);
    init::constant(&mut ln.bias, 1.0);

    let input = Var::new(Tensor::from_slice(vec![2, 1], &[3.0f64, 5.0]), true);
    let output = ln.forward(&input);

    assert_eq!(output.tensor.shape(), &[2, 1]);
    let s = output.tensor.as_slice();
    assert!((s[0] - 1.0).abs() < 1e-3);
    assert!((s[1] - 1.0).abs() < 1e-3);

    output.backward();
    assert!(input.grad().is_some());
}

#[test]
fn test_avg_pool2d_forward_backward() {
    let pool = AvgPool2d::<f64>::with_params(2, 2, 0, 1);
    let input_data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    let input = Var::new(Tensor::from_slice(vec![1, 1, 4, 4], &input_data), true);
    let output = pool.forward(&input);

    assert_eq!(output.tensor.shape(), &[1, 1, 2, 2]);
    let out_slice = output.tensor.as_slice();
    assert_eq!(out_slice, &[3.5, 5.5, 11.5, 13.5]);

    output.backward();
    let grad = input.grad().unwrap();
    assert_eq!(grad.shape(), &[1, 1, 4, 4]);
    for &val in grad.as_slice() {
        assert_eq!(val, 0.25);
    }
}

#[test]
fn test_max_pool2d_forward_backward() {
    let pool = MaxPool2d::<f64>::with_params(2, 2, 0, 1);
    let input_data: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    let input = Var::new(Tensor::from_slice(vec![1, 1, 4, 4], &input_data), true);
    let output = pool.forward(&input);

    assert_eq!(output.tensor.shape(), &[1, 1, 2, 2]);
    let out_slice = output.tensor.as_slice();
    assert_eq!(out_slice, &[6.0, 8.0, 14.0, 16.0]);

    output.backward();
    let grad = input.grad().unwrap();
    assert_eq!(grad.shape(), &[1, 1, 4, 4]);
    let grad_slice = grad.as_slice();
    
    #[allow(clippy::needless_range_loop)]
    for i in 0..16 {
        if i == 5 || i == 7 || i == 13 || i == 15 {
            assert_eq!(grad_slice[i], 1.0);
        } else {
            assert_eq!(grad_slice[i], 0.0);
        }
    }
}
