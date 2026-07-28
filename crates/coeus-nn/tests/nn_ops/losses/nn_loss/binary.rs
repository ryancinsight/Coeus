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
        Tensor::<f64, MoiraiBackend>::from_slice([4], &pred_data),
        true,
    );
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([4], &target_data),
        false,
    );

    let loss = binary_cross_entropy(&pred, &target, 1e-7);
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
    assert!(pred.grad().is_some());
}

#[test]
fn test_binary_cross_entropy_clamping() {
    // Check stability at 0.0 and 1.0
    let pred_data = vec![0.0f64, 1.0];
    let target_data = vec![0.0f64, 1.0];

    let pred = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &pred_data),
        true,
    );
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &target_data),
        false,
    );

    let loss = binary_cross_entropy(&pred, &target, 1e-7);
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
        Tensor::<f64, MoiraiBackend>::from_slice([3], &pred_data),
        true,
    );
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([3], &target_data),
        false,
    );

    let delta = 1.0;
    let loss = huber_loss(&pred, &target, delta);
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
    assert!(pred.grad().is_some());
}

#[test]
fn test_l1_loss() {
    // pred-target over shape [2, 2] gives diffs [3,-1,0.5,0].
    // forward: mean(|diff|) = (3 + 1 + 0.5 + 0) / 4 = 1.125 exactly.
    // backward: d/d_pred = sign(diff)/n = [1/4, -1/4, 1/4, 0].
    let pred = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[3.0, -1.0, 0.5, 4.0]),
        true,
    );
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[0.0, 0.0, 0.0, 4.0]),
        true,
    );

    let loss = l1_loss(&pred, &target);
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
fn test_bce_with_logits() {
    // Oracle: BCEWithLogits(z, y) == BCE(sigmoid(z), y), computed independently in f64.
    // logits z=[0, 2, -1], target y=[1, 0, 1].
    let zs = [0.0_f64, 2.0, -1.0];
    let ys = [1.0_f64, 0.0, 1.0];
    let n = zs.len() as f64;

    let logits = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([3], &zs), true);
    let target = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([3], &ys), true);

    let loss = bce_with_logits(&logits, &target);
    assert_eq!(loss.tensor.shape(), &[1]);

    // Reference forward via sigmoid + BCE.
    let sigmoid = |z: f64| 1.0 / (1.0 + (-z).exp());
    let mut expected = 0.0;
    for (&z, &y) in zs.iter().zip(ys.iter()) {
        let s = sigmoid(z);
        expected -= y * s.ln() + (1.0 - y) * (1.0 - s).ln();
    }
    expected /= n;
    let loss_val = loss.tensor.as_slice()[0];
    assert!(
        (loss_val - expected).abs() <= 1e-12,
        "bce_with_logits forward: got {loss_val:.17}, expected {expected:.17}"
    );

    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    // d/d_logit = (sigmoid(z) - y) / n; d/d_target = -z / n.
    let logit_grad = logits.grad().expect("logits must receive a gradient");
    let target_grad = target.grad().expect("target must receive a gradient");
    for (i, ((&z, &y), (&gz, &gt))) in zs
        .iter()
        .zip(ys.iter())
        .zip(
            logit_grad
                .as_slice()
                .iter()
                .zip(target_grad.as_slice().iter()),
        )
        .enumerate()
    {
        let exp_gz = (sigmoid(z) - y) / n;
        let exp_gt = -z / n;
        assert!(
            (gz - exp_gz).abs() <= 1e-12,
            "bce_with_logits d/d_logit[{i}]: got {gz:.17}, expected {exp_gz:.17}"
        );
        assert!(
            (gt - exp_gt).abs() <= 1e-12,
            "bce_with_logits d/d_target[{i}]: got {gt:.17}, expected {exp_gt:.17}"
        );
    }
}

#[test]
fn test_bce_with_logits_matches_bce_of_sigmoid() {
    // Numerically equal to applying sigmoid then binary_cross_entropy (eps→0 regime,
    // well inside (0,1) so clamping is inert).
    let zs = [0.5_f64, -0.7, 1.3, -2.0];
    let ys = [1.0_f64, 0.0, 1.0, 0.0];
    let logits = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([4], &zs), false);
    let target = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([4], &ys), false);
    let stable = bce_with_logits(&logits, &target).tensor.as_slice()[0];

    let probs: Vec<f64> = zs.iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect();
    let pv = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([4], &probs), false);
    let composed = binary_cross_entropy(&pv, &target, 1e-12).tensor.as_slice()[0];
    assert!(
        (stable - composed).abs() <= 1e-12,
        "bce_with_logits {stable:.17} != bce(sigmoid) {composed:.17}"
    );
}

#[test]
fn test_l1_loss_zero_when_equal() {
    // l1_loss(x, x) = 0 exactly (all diffs zero).
    let pred = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]),
        false,
    );
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([4], &[1.0, 2.0, 3.0, 4.0]),
        false,
    );
    let loss = l1_loss(&pred, &target);
    assert_eq!(loss.tensor.as_slice(), &[0.0_f64], "l1_loss(x, x) = 0");
}

#[test]
fn test_smooth_l1_loss() {
    // pred - target = z = [0.5, 2.0, -3.0], beta = 1.0.
    // |z|<β quadratic branch: 0.5·0.5²/1 = 0.125;
    // |z|≥β linear branch:    |2|-0.5·1 = 1.5,  |−3|-0.5·1 = 2.5.
    // mean = (0.125 + 1.5 + 2.5) / 3 = 1.375.
    let pred_data = vec![0.5_f64, 2.0, -3.0];
    let target_data = vec![0.0_f64, 0.0, 0.0];
    let pred = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([3], &pred_data),
        true,
    );
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([3], &target_data),
        false,
    );

    let loss = smooth_l1_loss(&pred, &target, 1.0);
    assert_eq!(loss.tensor.shape(), &[1]);
    let expected = (0.125 + 1.5 + 2.5) / 3.0;
    assert!(
        (loss.tensor.as_slice()[0] - expected).abs() < 1e-12,
        "smooth_l1 forward: got {}, want {expected}",
        loss.tensor.as_slice()[0]
    );

    // d loss / d pred_i = (1/N)·(z/β if |z|<β else sign(z)):
    //   z=0.5 -> 0.5/1 / 3;  z=2 -> +1 / 3;  z=-3 -> -1 / 3.
    loss.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let grad = pred.grad().expect("smooth_l1 pred grad");
    let expected_grad = [0.5 / 3.0, 1.0 / 3.0, -1.0 / 3.0];
    for (g, e) in grad.as_slice().iter().zip(expected_grad.iter()) {
        assert!((g - e).abs() < 1e-12, "smooth_l1 grad: got {g}, want {e}");
    }
}
