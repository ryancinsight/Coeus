use crate::normalization::value_oracle::{
    assert_matches_reference, channel_groups, fixture_upstream, fixture_values, Scenario,
    NORMALIZATION_TOLERANCE,
};
use coeus_autograd::Var;
use coeus_nn::{BatchNorm1d, BatchNorm2d, Module, ModuleError};
use coeus_tensor::Tensor;

/// Running variance after one training step on channels `[1, 2, 3, 4]`:
/// the sample variance of `[1, 2, 3, 4]` is `5 / 3`, so with momentum `0.1`
/// and initial variance `1` the update is `0.9 * 1 + 0.1 * 5 / 3`.
const RUNNING_VAR_AFTER_ONE_STEP: f64 = 0.9 + 0.1 * (5.0 / 3.0);

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

    let params = bn.parameters();
    assert_eq!(params.len(), 2); // weight + bias
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
    assert_eq!(bn.running_var.borrow().as_slice(), &[1.0, 1.0]);

    bn.forward(&input).expect("valid BatchNorm1d input");

    // Channel means are 2.5 and 6.5; momentum 0.1 from a zero start.
    let rm_after = bn.running_mean.borrow();
    assert_eq!(rm_after.as_slice().len(), 2);
    assert!((rm_after.as_slice()[0] - 0.25).abs() <= NORMALIZATION_TOLERANCE);
    assert!((rm_after.as_slice()[1] - 0.65).abs() <= NORMALIZATION_TOLERANCE);
    // Both channels hold four consecutive integers, so share one variance.
    let rv_after = bn.running_var.borrow();
    for (channel, &variance) in rv_after.as_slice().iter().enumerate() {
        assert!(
            (variance - RUNNING_VAR_AFTER_ONE_STEP).abs() <= NORMALIZATION_TOLERANCE,
            "channel {channel} running variance {variance}, expected {RUNNING_VAR_AFTER_ONE_STEP}"
        );
    }
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

    let params = bn.parameters();
    assert_eq!(params.len(), 2);
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
    assert_eq!(bn.running_var.borrow().as_slice(), &[1.0, 1.0]);

    bn.forward(&input).expect("valid BatchNorm2d input");

    // Channel means are 2.5 and 6.5; momentum 0.1 from a zero start.
    let rm_after = bn.running_mean.borrow();
    assert_eq!(rm_after.as_slice().len(), 2);
    assert!((rm_after.as_slice()[0] - 0.25).abs() <= NORMALIZATION_TOLERANCE);
    assert!((rm_after.as_slice()[1] - 0.65).abs() <= NORMALIZATION_TOLERANCE);
    // Both channels hold four consecutive integers, so share one variance.
    let rv_after = bn.running_var.borrow();
    for (channel, &variance) in rv_after.as_slice().iter().enumerate() {
        assert!(
            (variance - RUNNING_VAR_AFTER_ONE_STEP).abs() <= NORMALIZATION_TOLERANCE,
            "channel {channel} running variance {variance}, expected {RUNNING_VAR_AFTER_ONE_STEP}"
        );
    }
}

/// Hand-derived anchor independent of the shared oracle: channel 0 holds
/// `[1, 2, 3, 4]` (mean 2.5, population variance 1.25) and channel 1 holds
/// `[5, 6, 7, 8]` (mean 6.5, variance 1.25), so both normalize to
/// `[-1.5, -0.5, 0.5, 1.5] / sqrt(1.25 + eps)`.
#[test]
fn batchnorm2d_known_values_match_hand_derivation() {
    const EPS: f64 = 1e-5;
    let bn = BatchNorm2d::<f64>::new(2, EPS, 0.1).expect("invariant: test operation succeeds");
    let input = Var::new(
        Tensor::from_slice(vec![1, 2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let output = bn.forward(&input).expect("valid BatchNorm2d input");

    let inv_std = 1.0 / (1.25_f64 + EPS).sqrt();
    let channel = [-1.5, -0.5, 0.5, 1.5].map(|centered| centered * inv_std);
    for (index, &got) in output.tensor.as_slice().iter().enumerate() {
        let want = channel[index % 4];
        assert!(
            (got - want).abs() <= NORMALIZATION_TOLERANCE,
            "output[{index}] = {got}, expected {want}"
        );
    }
}

#[test]
fn batchnorm1d_forward_and_gradients_match_reference() {
    const EPS: f64 = 1e-5;
    let (batch, channels, length) = (4, 3, 2);
    let bn =
        BatchNorm1d::<f64>::new(channels, EPS, 0.1).expect("invariant: test operation succeeds");
    let total = u32::try_from(batch * channels * length).expect("invariant: fixture fits u32");
    assert_matches_reference(
        &Scenario {
            shape: vec![batch, channels, length],
            values: fixture_values(total),
            upstream: fixture_upstream(total),
            statistic_groups: channel_groups(batch, channels, length),
            channels: channel_groups(batch, channels, length),
            eps: EPS,
        },
        |input| bn.forward(input).expect("valid BatchNorm1d input"),
        &bn.weight,
        &bn.bias,
        "BatchNorm1d",
    );
}

#[test]
fn batchnorm2d_forward_and_gradients_match_reference() {
    const EPS: f64 = 1e-5;
    let (batch, channels, height, width) = (2, 2, 2, 3);
    let bn =
        BatchNorm2d::<f64>::new(channels, EPS, 0.1).expect("invariant: test operation succeeds");
    let total =
        u32::try_from(batch * channels * height * width).expect("invariant: fixture fits u32");
    assert_matches_reference(
        &Scenario {
            shape: vec![batch, channels, height, width],
            values: fixture_values(total),
            upstream: fixture_upstream(total),
            statistic_groups: channel_groups(batch, channels, height * width),
            channels: channel_groups(batch, channels, height * width),
            eps: EPS,
        },
        |input| bn.forward(input).expect("valid BatchNorm2d input"),
        &bn.weight,
        &bn.bias,
        "BatchNorm2d",
    );
}
