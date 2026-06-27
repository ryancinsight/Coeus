use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{
    binary_cross_entropy, cosine_embedding_loss, huber_loss, kl_divergence, margin_ranking_loss,
    nll_loss,
};
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

#[test]
fn test_kl_divergence_loss() {
    let input_data = [0.25_f64.ln(), 0.75_f64.ln()];
    let target_data = [0.25_f64, 0.75_f64];
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &input_data),
        true,
    );
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &target_data),
        false,
    );

    let loss = kl_divergence(&input, &target);
    assert_eq!(loss.tensor.shape(), &[1]);
    assert!(loss.tensor.as_slice()[0].abs() <= 2.0 * f64::EPSILON);

    loss.backward();
    let grad = input.grad().expect("invariant: KL input requires grad");
    let grad_slice = grad.as_slice();
    assert!((grad_slice[0] + 0.125).abs() < 1e-12);
    assert!((grad_slice[1] + 0.375).abs() < 1e-12);
}

#[test]
fn test_margin_ranking_loss() {
    let input1 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([4], &[2.0, 0.0, 1.0, 2.0]),
        true,
    );
    let input2 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([4], &[1.0, 1.0, 1.0, 1.0]),
        true,
    );
    let target = [1.0_f64, -1.0, 1.0, -1.0];

    let loss = margin_ranking_loss(&input1, &input2, &target, 0.5);
    assert_eq!(loss.tensor.shape(), &[1]);
    assert_eq!(loss.tensor.as_slice(), &[0.5_f64]);

    loss.backward();
    let g1 = input1
        .grad()
        .expect("invariant: margin ranking input1 requires grad");
    let g2 = input2
        .grad()
        .expect("invariant: margin ranking input2 requires grad");
    assert_eq!(g1.as_slice(), &[0.0, 0.0, -0.25, 0.25]);
    assert_eq!(g2.as_slice(), &[0.0, 0.0, 0.25, -0.25]);
}

#[test]
fn test_cosine_embedding_loss() {
    // cosine_embedding_loss semantics:
    //   y = 1:  loss_i = 1 − cos(x1_i, x2_i)
    //   y = −1: loss_i = max(0, cos(x1_i, x2_i) − margin)
    //   total   = mean(loss_i)
    //
    // All inputs are unit vectors so cos_sim is exact.

    // ── Case 1: identical unit vectors, y=1 → loss = 1−1 = 0.0 ──────────
    let x1 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[1.0_f64, 0.0]),
        false,
    );
    let x2 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[1.0_f64, 0.0]),
        false,
    );
    let loss_0 = cosine_embedding_loss(&x1, &x2, &[1.0_f64], 0.0);
    assert!(
        (loss_0.tensor.as_slice()[0] - 0.0).abs() < 1e-10,
        "identical y=1"
    );

    // ── Case 2: orthogonal unit vectors, y=1 → loss = 1−0 = 1.0 ─────────
    let x2_orth = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0_f64, 1.0]),
        false,
    );
    let loss_1 = cosine_embedding_loss(&x1, &x2_orth, &[1.0_f64], 0.0);
    assert!(
        (loss_1.tensor.as_slice()[0] - 1.0).abs() < 1e-10,
        "orthogonal y=1"
    );

    // ── Case 3: opposite vectors, y=−1, margin=0 → max(0,−1−0)=0.0 ───────
    let x2_opp = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[-1.0_f64, 0.0]),
        false,
    );
    let loss_2 = cosine_embedding_loss(&x1, &x2_opp, &[-1.0_f64], 0.0);
    assert!(
        (loss_2.tensor.as_slice()[0] - 0.0).abs() < 1e-10,
        "opposite y=-1 margin=0"
    );

    // ── Case 4: identical vectors, y=−1, margin=0 → max(0, 1−0)=1.0 ─────
    let loss_3 = cosine_embedding_loss(&x1, &x1, &[-1.0_f64], 0.0);
    assert!(
        (loss_3.tensor.as_slice()[0] - 1.0).abs() < 1e-10,
        "identical y=-1 margin=0"
    );

    // ── Case 5: batch of 2, y=[1,1] → mean([0.0, 1.0]) = 0.5 ────────────
    // pair 0: [[1,0]] vs [[1,0]] → cos=1, loss=0
    // pair 1: [[1,0]] vs [[0,1]] → cos=0, loss=1
    let x1_b = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[1.0_f64, 0.0, 1.0, 0.0]),
        false,
    );
    let x2_b = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[1.0_f64, 0.0, 0.0, 1.0]),
        false,
    );
    let loss_b = cosine_embedding_loss(&x1_b, &x2_b, &[1.0_f64, 1.0], 0.0);
    assert!(
        (loss_b.tensor.as_slice()[0] - 0.5).abs() < 1e-10,
        "batch mean"
    );

    // ── Backward: gradients must exist when requires_grad=true ────────────
    let x1_g = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[1.0_f64, 0.0]),
        true,
    );
    let x2_g = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0_f64, 1.0]),
        true,
    );
    cosine_embedding_loss(&x1_g, &x2_g, &[1.0_f64], 0.0).backward();
    assert!(x1_g.grad().is_some(), "x1 grad");
    assert!(x2_g.grad().is_some(), "x2 grad");
}
