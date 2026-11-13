use coeus_autograd::Var;
use coeus_nn::{init, BatchNorm1d, BatchNorm2d, Module, ModuleError};
use coeus_tensor::Tensor;

const BATCH_NORM_EPS: f64 = 1e-5;

fn assert_slice_close(actual: &[f64], expected: &[f64], tolerance: f64, context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: length mismatch");
    for (index, (actual_value, expected_value)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (actual_value - expected_value).abs();
        assert!(
            diff <= tolerance,
            "{context}: index {index} differs: actual={actual_value}, expected={expected_value}, diff={diff}, tol={tolerance}"
        );
    }
}

#[test]
fn test_batchnorm1d_forward_shape() {
    let bn = BatchNorm1d::<f64>::new(4, BATCH_NORM_EPS, 0.1)
        .expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::from_slice(
            vec![2, 4, 2],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = bn.forward(&input).expect("valid BatchNorm1d input");

    assert_eq!(output.tensor.shape(), &[2, 4, 2]);

    // Default affine params are weight=1, bias=0. Each channel has four values
    // [v, v+1, v+8, v+9] with mean m=v+4.5 and population variance 16.25.
    let std = (16.25_f64 + BATCH_NORM_EPS).sqrt();
    let expected: Vec<f64> = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]
    .iter()
    .enumerate()
    .map(|(index, value)| {
        let channel = (index / 2) % 4;
        let mean = channel as f64 + 4.5;
        (*value - mean) / std
    })
    .collect();
    assert_slice_close(
        output.tensor.as_slice(),
        &expected,
        1e-10,
        "batchnorm1d forward",
    );

    let params = bn.parameters();
    assert_eq!(params.len(), 2); // weight + bias
}

#[test]
fn test_batchnorm1d_backward_gradients_exist() {
    let bn = BatchNorm1d::<f64>::new(2, BATCH_NORM_EPS, 0.1)
        .expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::from_slice(
            vec![2, 2, 3],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = bn.forward(&input).expect("valid BatchNorm1d input");
    // Var::backward seeds tensor outputs with ones, so the objective is sum(output).
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    // d(sum y)/dbias_c = m = 6 and d(sum y)/dweight_c = sum(xhat_c) = 0.
    assert_slice_close(
        bn.bias.grad().expect("bias gradient must exist").as_slice(),
        &[6.0, 6.0],
        1e-10,
        "batchnorm1d dbias",
    );
    assert_slice_close(
        bn.weight
            .grad()
            .expect("weight gradient must exist")
            .as_slice(),
        &[0.0, 0.0],
        1e-10,
        "batchnorm1d dweight",
    );
    // For uniform upstream gradient and loss=sum(output), BatchNorm input gradient is zero.
    assert_slice_close(
        input.grad().expect("input gradient must exist").as_slice(),
        &[0.0; 12],
        1e-10,
        "batchnorm1d dinput",
    );
}

#[test]
fn test_batchnorm1d_running_stats_update() {
    let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1).expect("invariant: test operation succeeds");

    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let rm_before = bn.running_mean.borrow().clone();
    assert_eq!(rm_before.as_slice()[0], 0.0);
    assert_eq!(rm_before.as_slice()[1], 0.0);

    bn.forward(&input).expect("valid BatchNorm1d input");

    let rm_after = bn.running_mean.borrow();
    assert!((rm_after.as_slice()[0] - 0.25).abs() < 1e-6);
    assert!((rm_after.as_slice()[1] - 0.65).abs() < 1e-6);
}

#[test]
fn batchnorm1d_rejects_single_training_element_without_state_mutation() {
    let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1).expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::from_slice(vec![1, 2], &[1.0, 2.0])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let mean_before = bn.running_mean.borrow().clone();
    let variance_before = bn.running_var.borrow().clone();

    let error = match bn.forward(&input) {
        Ok(_) => panic!("one value per channel cannot define unbiased running variance"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        ModuleError::InsufficientElements {
            module: "BatchNorm1d",
            minimum: 2,
            actual: 1
        }
    ));
    assert_eq!(bn.running_mean.borrow().as_slice(), mean_before.as_slice());
    assert_eq!(
        bn.running_var.borrow().as_slice(),
        variance_before.as_slice()
    );
}

#[test]
fn batchnorm1d_state_borrow_failure_is_typed_and_transactional() {
    let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1).expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");
    let mean_before = bn.running_mean.borrow().clone();
    let variance_before = bn.running_var.borrow().clone();
    let running_mean_guard = bn.running_mean.borrow();

    let error = match bn.forward(&input) {
        Ok(_) => panic!("conflicting running-mean borrow must fail"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        ModuleError::StateBorrow {
            module: "BatchNorm1d",
            state: "running_mean"
        }
    ));
    drop(running_mean_guard);
    assert_eq!(bn.running_mean.borrow().as_slice(), mean_before.as_slice());
    assert_eq!(
        bn.running_var.borrow().as_slice(),
        variance_before.as_slice()
    );
}

#[test]
fn test_batchnorm2d_forward_shape() {
    let bn = BatchNorm2d::<f64>::new(4, BATCH_NORM_EPS, 0.1)
        .expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::from_slice(
            vec![2, 4, 1, 2],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = bn.forward(&input).expect("valid BatchNorm2d input");

    assert_eq!(output.tensor.shape(), &[2, 4, 1, 2]);

    // Spatial reduction over N,H,W gives the same per-channel statistics as the 1D fixture.
    let std = (16.25_f64 + BATCH_NORM_EPS).sqrt();
    let expected: Vec<f64> = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]
    .iter()
    .enumerate()
    .map(|(index, value)| {
        let channel = (index / 2) % 4;
        let mean = channel as f64 + 4.5;
        (*value - mean) / std
    })
    .collect();
    assert_slice_close(
        output.tensor.as_slice(),
        &expected,
        1e-10,
        "batchnorm2d forward",
    );

    let params = bn.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_batchnorm2d_backward_gradients_exist() {
    let bn = BatchNorm2d::<f64>::new(2, BATCH_NORM_EPS, 0.1)
        .expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = bn.forward(&input).expect("valid BatchNorm2d input");
    // Var::backward seeds tensor outputs with ones, so the objective is sum(output).
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    // m = N*H*W = 4 per channel. ch0=[1,2,3,4], ch1=[5,6,7,8].
    let variance = 1.25_f64; // population variance of [1,2,3,4]
    let std = (variance + BATCH_NORM_EPS).sqrt();
    let expected_xhat = [-1.5 / std, -0.5 / std, 0.5 / std, 1.5 / std];
    let mut expected_out = [0.0_f64; 8];
    expected_out[0..4].copy_from_slice(&expected_xhat);
    expected_out[4..8].copy_from_slice(&expected_xhat);
    assert_slice_close(
        output.tensor.as_slice(),
        &expected_out,
        1e-10,
        "batchnorm2d forward values",
    );

    assert_slice_close(
        bn.bias.grad().expect("bias gradient must exist").as_slice(),
        &[4.0, 4.0],
        1e-10,
        "batchnorm2d dbias",
    );
    assert_slice_close(
        bn.weight
            .grad()
            .expect("weight gradient must exist")
            .as_slice(),
        &[0.0, 0.0],
        1e-10,
        "batchnorm2d dweight",
    );
    assert_slice_close(
        input.grad().expect("input gradient must exist").as_slice(),
        &[0.0; 8],
        1e-10,
        "batchnorm2d dinput",
    );
}

#[test]
fn test_batchnorm2d_running_stats_update() {
    let bn = BatchNorm2d::<f64>::new(2, 1e-5, 0.1).expect("invariant: test operation succeeds");

    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let rm_before = bn.running_mean.borrow().clone();
    assert_eq!(rm_before.as_slice()[0], 0.0);
    assert_eq!(rm_before.as_slice()[1], 0.0);

    bn.forward(&input).expect("valid BatchNorm2d input");

    let rm_after = bn.running_mean.borrow();
    assert!((rm_after.as_slice()[0] - 0.25).abs() < 1e-6);
    assert!((rm_after.as_slice()[1] - 0.65).abs() < 1e-6);
}

#[test]
fn test_batchnorm2d_multi_channel_forward() {
    let mut bn = BatchNorm2d::<f64>::new(2, 1e-5, 0.1).expect("invariant: test operation succeeds");
    init::constant(&mut bn.weight, 1.0).expect("invariant: test operation succeeds");
    init::constant(&mut bn.bias, 0.0).expect("invariant: test operation succeeds");

    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = bn.forward(&input).expect("valid BatchNorm2d input");
    assert_eq!(output.tensor.shape(), &[1, 2, 2, 2]);

    let out_slice = output.tensor.as_slice();
    let ch0_mean: f64 = out_slice[0..4].iter().sum::<f64>() / 4.0;
    assert!(ch0_mean.abs() < 1e-5);
    let ch1_mean: f64 = out_slice[4..8].iter().sum::<f64>() / 4.0;
    assert!(ch1_mean.abs() < 1e-5);

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(input.grad().is_some());
}
