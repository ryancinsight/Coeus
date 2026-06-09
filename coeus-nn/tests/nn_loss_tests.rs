use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{binary_cross_entropy, huber_loss, nll_loss};
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
    loss.backward();
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

    loss.backward();
    assert!(!pred.grad().unwrap().as_slice()[0].is_nan());
}

#[test]
fn test_nll_loss() {
    // [3, 3] log-probabilities
    let log_probs_data = vec![-0.1f64, -2.3, -3.0, -1.5, -0.25, -4.0, -5.0, -1.0, -0.4];
    let targets = vec![0, 1, 2];

    let log_probs = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([3, 3], &log_probs_data),
        true,
    );
    let loss = nll_loss(&log_probs, &targets);
    assert_eq!(loss.tensor.shape(), &[1]);

    let loss_val = loss.tensor.as_slice()[0];
    // NLL = -1/N * sum(log_probs[i, target_i])
    let expected = -(-0.1 - 0.25 - 0.4) / 3.0;
    assert!((loss_val - expected).abs() < 1e-7);

    loss.backward();
    assert!(log_probs.grad().is_some());

    // Check gradients: -1 / N at target index, 0 elsewhere
    let grad = log_probs.grad().unwrap();
    let grad_slice = grad.as_slice();
    for (i, &target) in targets.iter().enumerate() {
        for j in 0..3 {
            let idx = i * 3 + j;
            let expected_g = if j == target { -1.0 / 3.0 } else { 0.0 };
            assert!((grad_slice[idx] - expected_g).abs() < 1e-7);
        }
    }
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

    loss.backward();
    assert!(pred.grad().is_some());
}
