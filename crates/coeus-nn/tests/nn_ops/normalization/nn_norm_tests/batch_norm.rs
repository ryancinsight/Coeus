use coeus_autograd::Var;
use coeus_nn::{init, BatchNorm1d, BatchNorm2d, Module, ModuleError};
use coeus_tensor::Tensor;

#[test]
fn test_batchnorm1d_forward_shape() {
    let bn = BatchNorm1d::<f64>::new(4, 1e-5, 0.1).expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::zeros(vec![2, 4, 10]).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = bn.forward(&input).expect("valid BatchNorm1d input");

    assert_eq!(output.tensor.shape(), &[2, 4, 10]);
    // Zero input, gamma=1, beta=0: normalized values are 0.
    for (index, &value) in output.tensor.as_slice().iter().enumerate() {
        assert_eq!(
            value, 0.0,
            "batchnorm1d output[{index}]={value}, expected 0"
        );
    }

    // Running stats: mean stays 0, variance moves from 1 toward 0 by momentum.
    let rm = bn.running_mean.borrow();
    assert_eq!(rm.as_slice(), &[0.0, 0.0, 0.0, 0.0]);
    let rv = bn.running_var.borrow();
    assert_eq!(rv.as_slice(), &[0.9, 0.9, 0.9, 0.9]);

    let params = bn.parameters();
    assert_eq!(params.len(), 2); // weight + bias
}

#[test]
fn test_batchnorm1d_backward_gradients_exist() {
    let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1).expect("invariant: test operation succeeds");
    let input_data = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let input = Var::new(
        Tensor::from_slice(vec![2, 2, 3], &input_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = bn.forward(&input).expect("valid BatchNorm1d input");
    assert_eq!(output.tensor.shape(), &[2, 2, 3]);

    // Channel 0 over N*L=6: values [1,2,3,7,8,9], mean=5, pop var=29/3.
    // Channel 1: [4,5,6,10,11,12], mean=8, pop var=29/3. Gamma=1, beta=0.
    let eps = 1e-5_f64;
    let inv = 1.0 / (29.0 / 3.0 + eps).sqrt();
    let expected: Vec<f64> = input_data
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let channel = (index / 3) % 2;
            let mean = if channel == 0 { 5.0 } else { 8.0 };
            (value - mean) * inv
        })
        .collect();
    let tol = 1e-11_f64;
    for (index, (&actual, &exp)) in output
        .tensor
        .as_slice()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert!(
            (actual - exp).abs() <= tol,
            "batchnorm1d output[{index}]={actual}, expected {exp}"
        );
    }

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    // Sum loss over batch-normalized outputs: per-channel sum of (x-mean)/sigma and its
    // input gradient are zero; d/dgamma = sum of normalized outputs, d/dbeta = element count.
    let input_grad = input
        .grad()
        .expect("invariant: input gradient must exist after backward");
    for (index, &grad) in input_grad.as_slice().iter().enumerate() {
        assert!(
            grad.abs() <= 1e-10,
            "input grad[{index}]={grad}, expected 0 for normalized sum"
        );
    }
    let weight_grad = bn
        .weight
        .grad()
        .expect("invariant: weight gradient must exist after backward");
    for channel in 0..2 {
        assert!(
            weight_grad.as_slice()[channel].abs() <= 1e-10,
            "weight grad[{channel}]={}, expected 0",
            weight_grad.as_slice()[channel]
        );
    }
    let bias_grad = bn
        .bias
        .grad()
        .expect("invariant: bias gradient must exist after backward");
    assert_eq!(bias_grad.as_slice(), &[6.0, 6.0]);
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
    assert_eq!(rm_before.as_slice(), &[0.0, 0.0]);

    bn.forward(&input).expect("valid BatchNorm1d input");

    // m=4: batch means [2.5, 6.5], pop vars [1.25, 1.25], unbiased [5/3, 5/3].
    // EMA with momentum 0.1 from (0,1): mean [0.25, 0.65], var [0.9 + 0.1*5/3, ...].
    let tol = 1e-12_f64;
    let rm_after = bn.running_mean.borrow();
    assert!((rm_after.as_slice()[0] - 0.25).abs() <= tol);
    assert!((rm_after.as_slice()[1] - 0.65).abs() <= tol);
    let expected_var = 0.9 + 0.1 * (5.0 / 3.0);
    let rv_after = bn.running_var.borrow();
    assert!((rv_after.as_slice()[0] - expected_var).abs() <= tol);
    assert!((rv_after.as_slice()[1] - expected_var).abs() <= tol);
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
    let bn = BatchNorm2d::<f64>::new(4, 1e-5, 0.1).expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::zeros(vec![2, 4, 3, 3]).expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = bn.forward(&input).expect("valid BatchNorm2d input");

    assert_eq!(output.tensor.shape(), &[2, 4, 3, 3]);
    for (index, &value) in output.tensor.as_slice().iter().enumerate() {
        assert_eq!(
            value, 0.0,
            "batchnorm2d output[{index}]={value}, expected 0"
        );
    }
    assert_eq!(bn.running_mean.borrow().as_slice(), &[0.0, 0.0, 0.0, 0.0]);
    assert_eq!(bn.running_var.borrow().as_slice(), &[0.9, 0.9, 0.9, 0.9]);

    let params = bn.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_batchnorm2d_backward_gradients_exist() {
    let bn = BatchNorm2d::<f64>::new(2, 1e-5, 0.1).expect("invariant: test operation succeeds");
    let input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 2, 2], &input_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = bn.forward(&input).expect("valid BatchNorm2d input");
    assert_eq!(output.tensor.shape(), &[1, 2, 2, 2]);

    // Per channel over H*W=4: [1,2,3,4] mean 2.5, pop var 1.25; [5,6,7,8] mean 6.5.
    let eps = 1e-5_f64;
    let inv = 1.0 / (1.25 + eps).sqrt();
    let expected: Vec<f64> = input_data
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let mean = if index < 4 { 2.5 } else { 6.5 };
            (value - mean) * inv
        })
        .collect();
    let tol = 1e-12_f64;
    for (index, (&actual, &exp)) in output
        .tensor
        .as_slice()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert!(
            (actual - exp).abs() <= tol,
            "batchnorm2d output[{index}]={actual}, expected {exp}"
        );
    }

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let input_grad = input
        .grad()
        .expect("invariant: input gradient must exist after backward");
    for (index, &grad) in input_grad.as_slice().iter().enumerate() {
        assert!(
            grad.abs() <= 1e-10,
            "input grad[{index}]={grad}, expected 0 for normalized sum"
        );
    }
    let weight_grad = bn
        .weight
        .grad()
        .expect("invariant: weight gradient must exist after backward");
    for channel in 0..2 {
        assert!(
            weight_grad.as_slice()[channel].abs() <= 1e-10,
            "weight grad[{channel}]={}, expected 0",
            weight_grad.as_slice()[channel]
        );
    }
    let bias_grad = bn
        .bias
        .grad()
        .expect("invariant: bias gradient must exist after backward");
    assert_eq!(bias_grad.as_slice(), &[4.0, 4.0]);
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
    assert_eq!(rm_before.as_slice(), &[0.0, 0.0]);

    bn.forward(&input).expect("valid BatchNorm2d input");

    // Per channel H*W=4: means [2.5,6.5], unbiased var [5/3,5/3].
    let tol = 1e-12_f64;
    let rm_after = bn.running_mean.borrow();
    assert!((rm_after.as_slice()[0] - 0.25).abs() <= tol);
    assert!((rm_after.as_slice()[1] - 0.65).abs() <= tol);
    let expected_var = 0.9 + 0.1 * (5.0 / 3.0);
    let rv_after = bn.running_var.borrow();
    assert!((rv_after.as_slice()[0] - expected_var).abs() <= tol);
    assert!((rv_after.as_slice()[1] - expected_var).abs() <= tol);
}

#[test]
fn test_batchnorm2d_multi_channel_forward() {
    let mut bn = BatchNorm2d::<f64>::new(2, 1e-5, 0.1).expect("invariant: test operation succeeds");
    init::constant(&mut bn.weight, 1.0).expect("invariant: test operation succeeds");
    init::constant(&mut bn.bias, 0.0).expect("invariant: test operation succeeds");

    let input_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 2, 2], &input_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = bn.forward(&input).expect("valid BatchNorm2d input");
    assert_eq!(output.tensor.shape(), &[1, 2, 2, 2]);

    let eps = 1e-5_f64;
    let inv = 1.0 / (1.25 + eps).sqrt();
    let expected: Vec<f64> = input_data
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let mean = if index < 4 { 2.5 } else { 6.5 };
            (value - mean) * inv
        })
        .collect();
    let tol = 1e-12_f64;
    for (index, (&actual, &exp)) in output
        .tensor
        .as_slice()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert!(
            (actual - exp).abs() <= tol,
            "batchnorm2d multi-channel output[{index}]={actual}, expected {exp}"
        );
    }

    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let input_grad = input
        .grad()
        .expect("invariant: input gradient must exist after backward");
    for (index, &grad) in input_grad.as_slice().iter().enumerate() {
        assert!(
            grad.abs() <= 1e-10,
            "input grad[{index}]={grad}, expected 0 for normalized sum"
        );
    }
    let weight_grad = bn
        .weight
        .grad()
        .expect("invariant: weight gradient must exist after backward");
    for channel in 0..2 {
        assert!(
            weight_grad.as_slice()[channel].abs() <= 1e-10,
            "weight grad[{channel}]={}, expected 0",
            weight_grad.as_slice()[channel]
        );
    }
    let bias_grad = bn
        .bias
        .grad()
        .expect("invariant: bias gradient must exist after backward");
    assert_eq!(bias_grad.as_slice(), &[4.0, 4.0]);
}
