use coeus_autograd::Var;
use coeus_nn::{
    binary_cross_entropy, huber_loss, nll_loss, GroupNorm, InstanceNorm1d, InstanceNorm2d, Linear,
    Module, ModuleExt, Sequential, StaticSeq,
};
use coeus_tensor::Tensor;

type B = coeus_core::MoiraiBackend;

/// Check a parameter-gradient entry against a central difference. `evaluate`
/// installs the perturbed parameter list into the module and returns the
/// scalar forward reduction; `Module::parameters` returns copies, so the
/// perturbed value must be loaded rather than mutated behind the module.
fn assert_norm_affine_gradients(
    weight_grad: &Tensor<f64, B>,
    bias_grad: &Tensor<f64, B>,
    elements_per_channel: f64,
    channels_per_group: usize,
) {
    let channels = bias_grad.shape()[0];
    assert_eq!(weight_grad.shape(), &[channels]);
    let weights = weight_grad.as_slice();
    let biases = bias_grad.as_slice();
    for group_start in (0..channels).step_by(channels_per_group) {
        let group_end = (group_start + channels_per_group).min(channels);
        let mut weight_sum = 0.0_f64;
        for channel in group_start..group_end {
            assert!(
                (biases[channel] - elements_per_channel).abs() <= 1e-12,
                "norm bias grad[{channel}] = {}, expected {elements_per_channel}",
                biases[channel]
            );
            weight_sum += weights[channel];
        }
        assert!(
            weight_sum.abs() <= 1e-9,
            "norm weight grad sum over group {group_start} = {weight_sum}, expected 0"
        );
    }
}

/// Check selected input-gradient entries against central differences of the
/// scalar forward reduction, reusing the same module weights.
fn assert_input_gradient_finite_differences<F>(
    baseline: &[f64],
    grad: &[f64],
    indices: &[usize],
    evaluate: F,
) where
    F: Fn(&[f64]) -> f64,
{
    let step = 1e-6_f64;
    for &index in indices {
        let mut plus = baseline.to_vec();
        plus[index] += step;
        let mut minus = baseline.to_vec();
        minus[index] -= step;
        let numeric = (evaluate(&plus) - evaluate(&minus)) / (2.0 * step);
        let analytic = grad[index];
        assert!(
            (analytic - numeric).abs() <= 1e-5,
            "input grad[{index}]: analytic {analytic} vs numeric {numeric}"
        );
    }
}

#[test]
fn test_groupnorm_forward_backward() {
    let backend = B::default();
    let num_features = 4;
    let gn = GroupNorm::<f64, B, 2>::new(num_features, 1e-5)
        .expect("invariant: test operation succeeds");

    // N=2, C=4, L=3
    let input_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, -1.0, -2.0, -3.0, -4.0,
        -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0,
    ];
    let input = Var::new(
        Tensor::from_slice_on([2, 4, 3], &input_data, &backend)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = gn.forward(&input).expect("valid GroupNorm input");

    assert_eq!(output.tensor.shape(), &[2, 4, 3]);

    // Backward pass
    let loss = coeus_autograd::sum(&output).expect("invariant: test operation succeeds");
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");

    let input_grad = input
        .grad()
        .expect("invariant: group norm gives its input a gradient");
    let weight_grad = gn
        .weight
        .grad()
        .expect("invariant: group norm gives its weight a gradient");
    let bias_grad = gn
        .bias
        .grad()
        .expect("invariant: group norm gives its bias a gradient");
    assert_eq!(input_grad.shape(), &[2, 4, 3]);
    // Two groups per batch, two channels per group, three spatial elements:
    // each channel receives 2 * 3 upstream units.
    assert_norm_affine_gradients(&weight_grad, &bias_grad, 6.0, 2);
    let evaluate = |data: &[f64]| -> f64 {
        let probe = Var::new(
            Tensor::from_slice_on([2, 4, 3], data, &backend)
                .expect("invariant: test backend operation succeeds"),
            false,
        )
        .expect("invariant: test backend operation succeeds");
        gn.forward(&probe)
            .expect("invariant: probe forward succeeds")
            .tensor
            .as_slice()
            .iter()
            .sum::<f64>()
    };
    assert_input_gradient_finite_differences(
        &input_data,
        input_grad.as_slice(),
        &[0, 7, 23],
        evaluate,
    );
}

#[test]
fn test_instancenorm1d_forward_backward() {
    let backend = B::default();
    let num_features = 3;
    let in1d = InstanceNorm1d::<f64, B>::new(num_features, 1e-5)
        .expect("invariant: test operation succeeds");

    // N=2, C=3, L=4
    let input_data = (1..=24).map(|x| x as f64).collect::<Vec<_>>();
    let input = Var::new(
        Tensor::from_slice_on([2, 3, 4], &input_data, &backend)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = in1d.forward(&input).expect("valid InstanceNorm1d input");

    assert_eq!(output.tensor.shape(), &[2, 3, 4]);

    let loss = coeus_autograd::sum(&output).expect("invariant: test operation succeeds");
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");

    let input_grad = input
        .grad()
        .expect("invariant: instance norm gives its input a gradient");
    let weight_grad = in1d
        .weight
        .grad()
        .expect("invariant: instance norm gives its weight a gradient");
    let bias_grad = in1d
        .bias
        .grad()
        .expect("invariant: instance norm gives its bias a gradient");
    assert_eq!(input_grad.shape(), &[2, 3, 4]);
    // Instance norm normalizes each (batch, channel) slice: two slices of four
    // elements per channel contribute to its affine gradients.
    assert_norm_affine_gradients(&weight_grad, &bias_grad, 8.0, 1);
    let evaluate = |data: &[f64]| -> f64 {
        let probe = Var::new(
            Tensor::from_slice_on([2, 3, 4], data, &backend)
                .expect("invariant: test backend operation succeeds"),
            false,
        )
        .expect("invariant: test backend operation succeeds");
        in1d.forward(&probe)
            .expect("invariant: probe forward succeeds")
            .tensor
            .as_slice()
            .iter()
            .sum::<f64>()
    };
    assert_input_gradient_finite_differences(
        &input_data,
        input_grad.as_slice(),
        &[0, 11, 23],
        evaluate,
    );
}

#[test]
fn test_instancenorm2d_forward_backward() {
    let backend = B::default();
    let num_features = 2;
    let in2d = InstanceNorm2d::<f64, B>::new(num_features, 1e-5)
        .expect("invariant: test operation succeeds");

    // N=1, C=2, H=3, W=3
    let input_data = (1..=18).map(|x| x as f64).collect::<Vec<_>>();
    let input = Var::new(
        Tensor::from_slice_on([1, 2, 3, 3], &input_data, &backend)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let output = in2d.forward(&input).expect("valid InstanceNorm2d input");

    assert_eq!(output.tensor.shape(), &[1, 2, 3, 3]);

    let loss = coeus_autograd::sum(&output).expect("invariant: test operation succeeds");
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");

    let input_grad = input
        .grad()
        .expect("invariant: instance norm gives its input a gradient");
    let weight_grad = in2d
        .weight
        .grad()
        .expect("invariant: instance norm gives its weight a gradient");
    let bias_grad = in2d
        .bias
        .grad()
        .expect("invariant: instance norm gives its bias a gradient");
    assert_eq!(input_grad.shape(), &[1, 2, 3, 3]);
    // One batch of 3x3 spatial elements per channel.
    assert_norm_affine_gradients(&weight_grad, &bias_grad, 9.0, 1);
    let evaluate = |data: &[f64]| -> f64 {
        let probe = Var::new(
            Tensor::from_slice_on([1, 2, 3, 3], data, &backend)
                .expect("invariant: test backend operation succeeds"),
            false,
        )
        .expect("invariant: test backend operation succeeds");
        in2d.forward(&probe)
            .expect("invariant: probe forward succeeds")
            .tensor
            .as_slice()
            .iter()
            .sum::<f64>()
    };
    assert_input_gradient_finite_differences(
        &input_data,
        input_grad.as_slice(),
        &[0, 8, 17],
        evaluate,
    );
}

#[test]
fn test_binary_cross_entropy_loss() {
    let backend = B::default();
    let pred = Var::new(
        Tensor::from_slice_on([3], &[0.1, 0.9, 0.5], &backend)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let target = Var::new(
        Tensor::from_slice_on([3], &[0.0, 1.0, 0.0], &backend)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let loss =
        binary_cross_entropy(&pred, &target, 1e-7).expect("valid binary cross entropy inputs");
    assert_eq!(loss.tensor.shape(), &[1]);

    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let pred_grad = pred
        .grad()
        .expect("invariant: binary cross entropy gives its prediction a gradient");
    assert_eq!(pred_grad.shape(), &[3]);
    // dL/dpred = -(t/p - (1-t)/(1-p)) / n with p = [0.1, 0.9, 0.5] and
    // t = [0, 1, 0]; every prediction is inside the clamp interval.
    let expected = [
        1.0_f64 / (0.9 * 3.0),
        -1.0_f64 / (0.9 * 3.0),
        1.0_f64 / (0.5 * 3.0),
    ];
    for (index, (&actual, &value)) in pred_grad.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - value).abs() <= 1e-12,
            "binary_cross_entropy grad[{index}] = {actual}, expected {value}"
        );
    }
}

#[test]
fn test_nll_loss() {
    let backend = B::default();
    let log_probs = Var::new(
        Tensor::from_slice_on([2, 3], &[-0.5, -2.0, -1.5, -1.0, -0.2, -3.0], &backend)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let targets = vec![0, 1];

    let loss = nll_loss(&log_probs, &targets).expect("invariant: test operation succeeds");
    assert_eq!(loss.tensor.shape(), &[1]);

    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let log_probs_grad = log_probs
        .grad()
        .expect("invariant: nll loss gives its log-probabilities a gradient");
    assert_eq!(log_probs_grad.shape(), &[2, 3]);
    // Mean-reduced NLL: dL/dlog_probs[i, j] = -onehot(j == target[i]) / 2.
    for (index, &value) in log_probs_grad.as_slice().iter().enumerate() {
        let row = index / 3;
        let class = index % 3;
        let expected = if class == targets[row] {
            -0.5_f64
        } else {
            0.0_f64
        };
        assert!(
            (value - expected).abs() <= 1e-12,
            "nll_loss grad[{index}] = {value}, expected {expected}"
        );
    }
}

#[test]
fn test_huber_loss() {
    let backend = B::default();
    let pred = Var::new(
        Tensor::from_slice_on([3], &[1.0, 3.0, 1.5], &backend)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let target = Var::new(
        Tensor::from_slice_on([3], &[1.5, 2.0, 4.0], &backend)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let loss = huber_loss(&pred, &target, 1.0)
        .expect("invariant: matching non-empty shapes and positive finite delta");
    assert_eq!(loss.tensor.shape(), &[1]);

    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let pred_grad = pred
        .grad()
        .expect("invariant: huber loss gives its prediction a gradient");
    assert_eq!(pred_grad.shape(), &[3]);
    // z = pred - target = [-0.5, 1.0, -2.5], delta = 1: the first two take the
    // quadratic branch (z / n) and the last one the linear branch (sign(z) / n).
    let expected = [-0.5_f64 / 3.0, 1.0 / 3.0, -1.0 / 3.0];
    for (index, (&actual, &value)) in pred_grad.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - value).abs() <= 1e-12,
            "huber_loss grad[{index}] = {actual}, expected {value}"
        );
    }
}

#[path = "nn_new_ops_tests/sequential.rs"]
mod sequential;
