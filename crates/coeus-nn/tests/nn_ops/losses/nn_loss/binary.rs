//! Binary and piecewise-regression loss contracts.

use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::bce_with_logits;
use coeus_nn::binary_cross_entropy;
use coeus_nn::huber_loss;
use coeus_nn::l1_loss;
use coeus_nn::smooth_l1_loss;
use coeus_tensor::Tensor;

#[test]
fn test_binary_cross_entropy() {
    let pred_data = vec![0.1f64, 0.9, 0.8, 0.2];
    let target_data = vec![0.0f64, 1.0, 1.0, 0.0];

    let pred = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([4], &pred_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([4], &target_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let loss =
        binary_cross_entropy(&pred, &target, 1e-7).expect("valid binary cross entropy inputs");
    assert_eq!(loss.tensor.shape(), &[1]);

    let loss_val = loss.tensor.as_slice()[0];
    // BCE = -1/N * sum(t_i * ln(p_i) + (1-t_i) * ln(1-p_i))
    let mut expected_loss = 0.0;
    for i in 0..4 {
        let p = pred_data[i];
        let t = target_data[i];
        expected_loss -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();
    }
    expected_loss /= 4.0;
    assert!((loss_val - expected_loss).abs() < 1e-7);

    // Backward
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let pred_grad = pred
        .grad()
        .expect("invariant: binary cross entropy gives pred a gradient");
    assert_eq!(pred_grad.shape(), &[4]);
    // dL/dpred = -(t/p - (1-t)/(1-p)) / n with p = clamp(pred, eps, 1-eps).
    // The fixture stays inside the clamp interval, so p = pred:
    // p = 0.1 (t = 0), 0.9 (t = 1), 0.8 (t = 1), 0.2 (t = 0).
    let expected_grad = [
        (1.0 / 0.9) / 4.0,
        -(1.0 / 0.9) / 4.0,
        -(1.0 / 0.8) / 4.0,
        (1.0 / 0.8) / 4.0,
    ];
    for (index, (&actual, &expected)) in pred_grad
        .as_slice()
        .iter()
        .zip(expected_grad.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "binary_cross_entropy grad[{index}] = {actual}, expected {expected}"
        );
    }
}

#[test]
fn test_binary_cross_entropy_clamping() {
    // Check stability at 0.0 and 1.0
    let pred_data = vec![0.0f64, 1.0];
    let target_data = vec![0.0f64, 1.0];

    let pred = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &pred_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &target_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let loss =
        binary_cross_entropy(&pred, &target, 1e-7).expect("valid binary cross entropy inputs");
    assert_eq!(loss.tensor.shape(), &[1]);
    let loss_val = loss.tensor.as_slice()[0];
    assert!(!loss_val.is_nan());
    assert!(!loss_val.is_infinite());

    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    assert!(!pred.grad().unwrap().as_slice()[0].is_nan());
}

#[test]
fn test_huber_loss() {
    let pred_data = vec![1.0f64, 4.0, -2.0];
    let target_data = vec![1.5f64, 2.0, -1.0];

    let pred = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([3], &pred_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([3], &target_data)
            .expect("invariant: test backend operation succeeds"),
        false,
    )
    .expect("invariant: test backend operation succeeds");

    let delta = 1.0;
    let loss = huber_loss(&pred, &target, delta)
        .expect("invariant: matching non-empty shapes and positive finite delta");
    assert_eq!(loss.tensor.shape(), &[1]);

    let loss_val = loss.tensor.as_slice()[0];
    // Diffs: -0.5 (abs <= 1.0), 2.0 (abs > 1.0), -1.0 (abs <= 1.0)
    // Loss per item:
    // 0.5 * (-0.5)^2 / 1.0 = 0.125
    // 2.0 - 0.5 * 1.0 = 1.5
    // 0.5 * (-1.0)^2 / 1.0 = 0.5
    // Average = (0.125 + 1.5 + 0.5) / 3 = 2.125 / 3 = 0.70833333...
    let expected = (0.125 + 1.5 + 0.5) / 3.0;
    assert!((loss_val - expected).abs() < 1e-7);

    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let pred_grad = pred
        .grad()
        .expect("invariant: huber loss gives pred a gradient");
    assert_eq!(pred_grad.shape(), &[3]);
    // dL/dpred = (|z| <= delta ? z : sign(z) * delta) / n with z = pred - target.
    // z = [-0.5, 2.0, -1.0] and delta = 1.0: the first and third elements are
    // in the quadratic region, the second one in the linear region.
    let expected_grad = [-0.5 / 3.0, 1.0 / 3.0, -1.0 / 3.0];
    for (index, (&actual, &expected)) in pred_grad
        .as_slice()
        .iter()
        .zip(expected_grad.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "huber_loss grad[{index}] = {actual}, expected {expected}"
        );
    }
}

#[test]
fn test_huber_loss_regions() {
    // delta = 1.5 separates the classical Huber definition from the smooth-L1
    // alternative: the quadratic branch is 0.5 * z^2 and the linear branch is
    // delta * |z| - 0.5 * delta^2.
    // z = pred - target = [-0.5, 2.0, -1.0]; |z| <= 1.5 holds for the first and
    // the third element, the second one takes the linear branch.
    // Forward: 0.5 * 0.25 = 0.125, 1.5 * 2.0 - 0.5 * 2.25 = 1.875,
    //          0.5 * 1.0 = 0.5; mean = 2.5 / 3.
    // Backward: z / n in the quadratic region, sign(z) * delta / n in the
    // linear region.
    let pred_data = [1.0_f64, 4.0, -2.0];
    let target_data = [1.5_f64, 2.0, -1.0];
    let delta = 1.5_f64;

    let pred = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([3], &pred_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([3], &target_data)
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let loss = huber_loss(&pred, &target, delta)
        .expect("invariant: matching non-empty shapes and positive finite delta");
    assert_eq!(loss.tensor.shape(), &[1]);
    let expected_loss = (0.125 + 1.875 + 0.5) / 3.0;
    let loss_val = loss.tensor.as_slice()[0];
    assert!(
        (loss_val - expected_loss).abs() <= 1e-12,
        "huber_loss regions forward = {loss_val}, expected {expected_loss}"
    );

    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let pred_grad = pred
        .grad()
        .expect("invariant: huber loss gives pred a gradient");
    let target_grad = target
        .grad()
        .expect("invariant: huber loss gives target a gradient");
    let expected_grad = [-0.5 / 3.0, 1.5 / 3.0, -1.0 / 3.0];
    for (index, (&actual, &expected)) in pred_grad
        .as_slice()
        .iter()
        .zip(expected_grad.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "huber_loss regions grad[{index}] = {actual}, expected {expected}"
        );
    }
    for (index, (&actual, &expected)) in target_grad
        .as_slice()
        .iter()
        .zip(expected_grad.iter())
        .enumerate()
    {
        assert!(
            (actual + expected).abs() <= 1e-12,
            "huber_loss regions target grad[{index}] = {actual}, expected {}",
            -expected
        );
    }
}

#[test]
fn test_l1_loss() {
    // pred-target over shape [2, 2] gives diffs [3,-1,0.5,0].
    // forward: mean(|diff|) = (3 + 1 + 0.5 + 0) / 4 = 1.125 exactly.
    // backward: d/d_pred = sign(diff)/n = [1/4, -1/4, 1/4, 0].
    let pred = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[3.0, -1.0, 0.5, 4.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[0.0, 0.0, 0.0, 4.0])
            .expect("invariant: test backend operation succeeds"),
        true,
    )
    .expect("invariant: test backend operation succeeds");

    let loss = l1_loss(&pred, &target).expect("invariant: test operation succeeds");
    assert_eq!(loss.tensor.shape(), &[1]);
    let loss_val = loss.tensor.as_slice()[0];
    let expected = (3.0 + 1.0 + 0.5) / 4.0;
    assert!(
        (loss_val - expected).abs() <= 4.0 * f64::EPSILON * expected,
        "l1_loss forward: got {loss_val:.17}, expected {expected:.17}"
    );

    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = pred.grad().expect("pred must receive a gradient");
    assert_eq!(grad.shape(), &[2, 2], "pred grad preserves input shape");
    let quarter = 1.0 / 4.0;
    let expected_grad = [quarter, -quarter, quarter, 0.0];
    for (i, (&g, &e)) in grad.as_slice().iter().zip(expected_grad.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 4.0 * f64::EPSILON,
            "l1_loss grad[{i}]: got {g:.17}, expected {e:.17} (sign(diff)/n)"
        );
    }

    let target_grad = target.grad().expect("target must receive a gradient");
    assert_eq!(
        target_grad.shape(),
        &[2, 2],
        "target grad preserves input shape"
    );
    for (i, (&g, &e)) in target_grad
        .as_slice()
        .iter()
        .zip(expected_grad.iter())
        .enumerate()
    {
        assert!(
            (g + e).abs() <= 4.0 * f64::EPSILON,
            "l1_loss target grad[{i}]: got {g:.17}, expected {:.17}",
            -e
        );
    }
}

#[test]
fn test_l1_loss_transposed_provider_layout() {
    // A transposed view must remain logical-order correct without host staging.
    let pred_base =
        Tensor::<f64, MoiraiBackend>::from_slice([2, 3], &[3.0, -1.0, 0.5, 4.0, 2.0, -2.0])
            .expect("invariant: test backend operation succeeds");
    let target_base =
        Tensor::<f64, MoiraiBackend>::from_slice([2, 3], &[0.0, 0.0, 0.0, 1.0, 1.0, -1.0])
            .expect("invariant: test backend operation succeeds");
    let pred = Var::new(pred_base.permute(&[1, 0]), true)
        .expect("invariant: test backend operation succeeds");
    let target = Var::new(target_base.permute(&[1, 0]), true)
        .expect("invariant: test backend operation succeeds");

    let loss = l1_loss(&pred, &target).expect("invariant: test operation succeeds");
    assert_eq!(loss.tensor.shape(), &[1]);
    // Logical differences are [3, 3, -1, 1, 0.5, -1], so mean abs = 9.5 / 6.
    assert!((loss.tensor.as_slice()[0] - (9.5 / 6.0)).abs() < 1e-12);

    loss.backward()
        .expect("invariant: transposed L1 backward completes");
    let pred_grad = pred.grad().expect("transposed pred gradient");
    let target_grad = target.grad().expect("transposed target gradient");
    assert_eq!(pred_grad.shape(), &[3, 2]);
    assert_eq!(target_grad.shape(), &[3, 2]);
    let expected_pred = [
        1.0 / 6.0,
        1.0 / 6.0,
        -1.0 / 6.0,
        1.0 / 6.0,
        1.0 / 6.0,
        -1.0 / 6.0,
    ];
    for (i, (&actual, &expected)) in pred_grad
        .as_slice()
        .iter()
        .zip(expected_pred.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() < 1e-12,
            "transposed L1 pred grad[{i}] = {actual}, expected {expected}"
        );
    }
    for (actual, expected) in target_grad.as_slice().iter().zip(expected_pred.iter()) {
        assert!((*actual + expected).abs() < 1e-12);
    }
}

#[path = "binary/extended.rs"]
mod extended;
