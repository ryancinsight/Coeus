use coeus_autograd::Var;
use coeus_nn::{init, BatchNorm1d, BatchNorm2d, Module, ModuleError};
use coeus_tensor::Tensor;

#[test]
fn test_batchnorm1d_forward_shape() {
    let bn = BatchNorm1d::<f64>::new(4, 1e-5, 0.1);
    let input = Var::new(Tensor::zeros(vec![2, 4, 10]), true);
    let output = bn.forward(&input).expect("valid BatchNorm1d input");

    assert_eq!(output.tensor.shape(), &[2, 4, 10]);

    let params = bn.parameters();
    assert_eq!(params.len(), 2); // weight + bias
}

#[test]
fn test_batchnorm1d_backward_gradients_exist() {
    let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1);
    let input = Var::new(
        Tensor::from_slice(
            vec![2, 2, 3],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        ),
        true,
    );

    let output = bn.forward(&input).expect("valid BatchNorm1d input");
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    assert!(input.grad().is_some());
    assert!(bn.weight.grad().is_some());
    assert!(bn.bias.grad().is_some());
}

#[test]
fn test_batchnorm1d_running_stats_update() {
    let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1);

    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        false,
    );

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
    let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1);
    let input = Var::new(Tensor::from_slice(vec![1, 2], &[1.0, 2.0]), false);
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
    let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1);
    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        false,
    );
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
    let bn = BatchNorm2d::<f64>::new(4, 1e-5, 0.1);
    let input = Var::new(Tensor::zeros(vec![2, 4, 3, 3]), true);
    let output = bn.forward(&input).expect("valid BatchNorm2d input");

    assert_eq!(output.tensor.shape(), &[2, 4, 3, 3]);

    let params = bn.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_batchnorm2d_backward_gradients_exist() {
    let bn = BatchNorm2d::<f64>::new(2, 1e-5, 0.1);
    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        true,
    );

    let output = bn.forward(&input).expect("valid BatchNorm2d input");
    output
        .backward()
        .expect("invariant: valid autograd fixture completes backward");

    assert!(input.grad().is_some());
    assert!(bn.weight.grad().is_some());
    assert!(bn.bias.grad().is_some());
}

#[test]
fn test_batchnorm2d_running_stats_update() {
    let bn = BatchNorm2d::<f64>::new(2, 1e-5, 0.1);

    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        false,
    );

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
    let mut bn = BatchNorm2d::<f64>::new(2, 1e-5, 0.1);
    init::constant(&mut bn.weight, 1.0);
    init::constant(&mut bn.bias, 0.0);

    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        true,
    );

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
