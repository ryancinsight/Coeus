use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{
    bce_with_logits, binary_cross_entropy, cosine_embedding_loss, huber_loss, kl_divergence,
    l1_loss, margin_ranking_loss, nll_loss, pairwise_distance, poisson_nll, soft_margin, triplet_margin_loss,
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

    loss.backward();
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

    loss.backward();
    // d/d_logit = (sigmoid(z) - y) / n; d/d_target = -z / n.
    let logit_grad = logits.grad().expect("logits must receive a gradient");
    let target_grad = target.grad().expect("target must receive a gradient");
    for (i, ((&z, &y), (&gz, &gt))) in zs
        .iter()
        .zip(ys.iter())
        .zip(logit_grad.as_slice().iter().zip(target_grad.as_slice().iter()))
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
fn test_poisson_nll() {
    // log-input form: loss = mean(exp(z) - y*z); d/dz = (exp(z)-y)/n; d/dy = -z/n.
    let zs = [0.0_f64, 1.0, -0.5];
    let ys = [2.0_f64, 0.0, 3.0];
    let n = zs.len() as f64;

    let input = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([3], &zs), true);
    let target = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([3], &ys), true);

    let loss = poisson_nll(&input, &target);
    assert_eq!(loss.tensor.shape(), &[1]);

    let mut expected = 0.0;
    for (&z, &y) in zs.iter().zip(ys.iter()) {
        expected += z.exp() - y * z;
    }
    expected /= n;
    let loss_val = loss.tensor.as_slice()[0];
    assert!(
        (loss_val - expected).abs() <= 1e-12,
        "poisson_nll forward: got {loss_val:.17}, expected {expected:.17}"
    );

    loss.backward();
    let input_grad = input.grad().expect("input must receive a gradient");
    let target_grad = target.grad().expect("target must receive a gradient");
    for (i, ((&z, &y), (&gz, &gt))) in zs
        .iter()
        .zip(ys.iter())
        .zip(input_grad.as_slice().iter().zip(target_grad.as_slice().iter()))
        .enumerate()
    {
        let exp_gz = (z.exp() - y) / n;
        let exp_gt = -z / n;
        assert!(
            (gz - exp_gz).abs() <= 1e-12,
            "poisson_nll d/d_input[{i}]: got {gz:.17}, expected {exp_gz:.17}"
        );
        assert!(
            (gt - exp_gt).abs() <= 1e-12,
            "poisson_nll d/d_target[{i}]: got {gt:.17}, expected {exp_gt:.17}"
        );
    }
}

#[test]
fn test_soft_margin() {
    // loss = mean(log(1+exp(-y*x))); d/dx = -y*sigmoid(-y*x)/n; d/dy = -x*sigmoid(-y*x)/n.
    let xs = [0.5_f64, -1.2, 2.0];
    let ys = [1.0_f64, -1.0, 1.0];
    let n = xs.len() as f64;

    let input = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([3], &xs), true);
    let target = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([3], &ys), true);

    let loss = soft_margin(&input, &target);
    assert_eq!(loss.tensor.shape(), &[1]);

    let mut expected = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        expected += (1.0 + (-y * x).exp()).ln();
    }
    expected /= n;
    let loss_val = loss.tensor.as_slice()[0];
    assert!(
        (loss_val - expected).abs() <= 1e-12,
        "soft_margin forward: got {loss_val:.17}, expected {expected:.17}"
    );

    loss.backward();
    let sigmoid = |z: f64| 1.0 / (1.0 + (-z).exp());
    let input_grad = input.grad().expect("input must receive a gradient");
    let target_grad = target.grad().expect("target must receive a gradient");
    for (i, ((&x, &y), (&gx, &gt))) in xs
        .iter()
        .zip(ys.iter())
        .zip(input_grad.as_slice().iter().zip(target_grad.as_slice().iter()))
        .enumerate()
    {
        let sig = sigmoid(-y * x);
        let exp_gx = -y * sig / n;
        let exp_gt = -x * sig / n;
        assert!(
            (gx - exp_gx).abs() <= 1e-12,
            "soft_margin d/d_input[{i}]: got {gx:.17}, expected {exp_gx:.17}"
        );
        assert!(
            (gt - exp_gt).abs() <= 1e-12,
            "soft_margin d/d_target[{i}]: got {gt:.17}, expected {exp_gt:.17}"
        );
    }
}

#[test]
fn test_pairwise_distance() {
    let p = 2.0_f64;
    let eps = 1e-6_f64;
    let x1 = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]), true);
    let x2 = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[0.0, 0.0, 1.0, 1.0]), true);
    let dist = pairwise_distance(&x1, &x2, p, eps);
    assert_eq!(dist.tensor.shape(), &[2]);
    let diffs = [[1.0_f64, 2.0], [2.0, 3.0]];
    let s: Vec<f64> = diffs.iter().map(|r| r.iter().map(|d| d.abs().powf(p)).sum::<f64>()).collect();
    let out = dist.tensor.as_slice();
    for i in 0..2 {
        let exp = (s[i] + eps).powf(1.0 / p);
        assert!((out[i] - exp).abs() <= 1e-12, "pd fwd {}", i);
    }
    let total = coeus_autograd::sum(&dist);
    total.backward();
    let g1 = x1.grad().expect("x1 grad");
    let g2 = x2.grad().expect("x2 grad");
    for i in 0..2 {
        let scale = (s[i] + eps).powf(1.0 / p - 1.0);
        for k in 0..2 {
            let d = diffs[i][k];
            let eg = scale * d.abs().powf(p - 1.0) * d.signum();
            assert!((g1.as_slice()[i * 2 + k] - eg).abs() <= 1e-12, "pd gx1 {} {}", i, k);
            assert!((g2.as_slice()[i * 2 + k] + eg).abs() <= 1e-12, "pd gx2 {} {}", i, k);
        }
    }
}

#[test]
fn test_triplet_margin_loss() {
    // anchor=[0,0], positive=[2,0], negative=[0,2.5], margin=1, p=2, eps=0.
    // d_ap=2, d_an=2.5, hinge=max(0, 2 - 2.5 + 1)=0.5 (active) → loss=0.5.
    // grads (N=1): d/anchor=[-1,1], d/positive=[1,0], d/negative=[0,-1].
    let anchor = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0, 0.0]), true);
    let positive = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[2.0, 0.0]), true);
    let negative = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0, 2.5]), true);

    let loss = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, 0.0);
    assert_eq!(loss.tensor.shape(), &[1]);
    assert!(
        (loss.tensor.as_slice()[0] - 0.5).abs() <= 1e-12,
        "triplet forward: got {}, expected 0.5",
        loss.tensor.as_slice()[0]
    );

    loss.backward();
    let ga = anchor.grad().expect("anchor grad");
    let gp = positive.grad().expect("positive grad");
    let gn = negative.grad().expect("negative grad");
    let approx = |a: &[f64], b: [f64; 2], who: &str| {
        for k in 0..2 {
            assert!(
                (a[k] - b[k]).abs() <= 1e-12,
                "triplet d/d_{who}[{k}]: got {}, expected {}",
                a[k],
                b[k]
            );
        }
    };
    approx(ga.as_slice(), [-1.0, 1.0], "anchor");
    approx(gp.as_slice(), [1.0, 0.0], "positive");
    approx(gn.as_slice(), [0.0, -1.0], "negative");
}

#[test]
fn test_triplet_margin_loss_inactive() {
    // Easy triplet (d_ap << d_an by more than margin) → hinge 0, loss 0.
    let anchor = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0, 0.0]), false);
    let positive = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.5, 0.0]), false);
    let negative = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[5.0, 0.0]), false);
    let loss = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, 0.0);
    assert_eq!(loss.tensor.as_slice(), &[0.0_f64], "easy triplet → loss 0");
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
