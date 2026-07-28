//! Distance, ranking, and embedding loss contracts.

use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::cosine_embedding_loss;
use coeus_nn::cosine_similarity;
use coeus_nn::margin_ranking_loss;
use coeus_nn::pairwise_distance;
use coeus_nn::triplet_margin_loss;
use coeus_nn::triplet_margin_with_distance_loss;
use coeus_tensor::Tensor;

#[test]
fn test_pairwise_distance() {
    let p = 2.0_f64;
    let eps = 1e-6_f64;
    let x1 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[1.0, 2.0, 3.0, 4.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x2 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[0.0, 0.0, 1.0, 1.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let dist = pairwise_distance(&x1, &x2, p, eps).expect("run operation");
    assert_eq!(dist.tensor.shape(), &[2]);
    // Torch `pairwise_distance` = `at::norm(x1 - x2 + eps, p)`: eps is added to
    // the difference itself (not clamped onto the summed norm), which keeps the
    // norm strictly positive so the `s^(1/p - 1)` gradient factor stays finite.
    let diffs = [[1.0_f64 + eps, 2.0 + eps], [2.0 + eps, 3.0 + eps]];
    let s: Vec<f64> = diffs
        .iter()
        .map(|r| r.iter().map(|d| d.abs().powf(p)).sum::<f64>())
        .collect();
    let out = dist.tensor.as_slice();
    for i in 0..2 {
        let exp = s[i].powf(1.0 / p);
        assert!((out[i] - exp).abs() <= 1e-12, "pd fwd {}", i);
    }
    let total = coeus_autograd::sum(&dist).expect("run operation");
    total.backward().expect("run backward");
    let g1 = x1.grad().expect("x1 grad");
    let g2 = x2.grad().expect("x2 grad");
    for i in 0..2 {
        let scale = s[i].powf(1.0 / p - 1.0);
        for (k, &d) in diffs[i].iter().enumerate() {
            let eg = scale * d.abs().powf(p - 1.0) * d.signum();
            assert!(
                (g1.as_slice()[i * 2 + k] - eg).abs() <= 1e-12,
                "pd gx1 {} {}",
                i,
                k
            );
            assert!(
                (g2.as_slice()[i * 2 + k] + eg).abs() <= 1e-12,
                "pd gx2 {} {}",
                i,
                k
            );
        }
    }
}

#[test]
fn test_triplet_margin_loss() {
    // anchor=[0,0], positive=[2,0], negative=[0,2.5], margin=1, p=2, eps=0.
    // d_ap=2, d_an=2.5, hinge=max(0, 2 - 2.5 + 1)=0.5 (active) → loss=0.5.
    // grads (N=1): d/anchor=[-1,1], d/positive=[1,0], d/negative=[0,-1].
    let anchor = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0, 0.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let positive = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[2.0, 0.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let negative = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0, 2.5]).expect("construct tensor"),
        true,
    ).expect("construct variable");

    let loss = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, 0.0).expect("run operation");
    assert_eq!(loss.tensor.shape(), &[1]);
    assert!(
        (loss.tensor.as_slice()[0] - 0.5).abs() <= 1e-12,
        "triplet forward: got {}, expected 0.5",
        loss.tensor.as_slice()[0]
    );

    loss.backward().expect("run backward");
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
    let anchor = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0, 0.0]).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let positive = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.5, 0.0]).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let negative = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[5.0, 0.0]).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let loss = triplet_margin_loss(&anchor, &positive, &negative, 1.0, 2.0, 0.0).expect("run operation");
    assert_eq!(loss.tensor.as_slice(), &[0.0_f64], "easy triplet → loss 0");
}

#[test]
fn test_margin_ranking_loss() {
    let input1 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([4], &[2.0, 0.0, 1.0, 2.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let input2 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([4], &[1.0, 1.0, 1.0, 1.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let target = [1.0_f64, -1.0, 1.0, -1.0];

    let loss = margin_ranking_loss(&input1, &input2, &target, 0.5).expect("run operation");
    assert_eq!(loss.tensor.shape(), &[1]);
    assert_eq!(loss.tensor.as_slice(), &[0.5_f64]);

    loss.backward().expect("run backward");
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
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[1.0_f64, 0.0]).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x2 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[1.0_f64, 0.0]).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let loss_0 = cosine_embedding_loss(&x1, &x2, &[1.0_f64], 0.0).expect("run operation");
    assert!(
        (loss_0.tensor.as_slice()[0] - 0.0).abs() < 1e-10,
        "identical y=1"
    );

    // ── Case 2: orthogonal unit vectors, y=1 → loss = 1−0 = 1.0 ─────────
    let x2_orth = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0_f64, 1.0]).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let loss_1 = cosine_embedding_loss(&x1, &x2_orth, &[1.0_f64], 0.0).expect("run operation");
    assert!(
        (loss_1.tensor.as_slice()[0] - 1.0).abs() < 1e-10,
        "orthogonal y=1"
    );

    // ── Case 3: opposite vectors, y=−1, margin=0 → max(0,−1−0)=0.0 ───────
    let x2_opp = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[-1.0_f64, 0.0]).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let loss_2 = cosine_embedding_loss(&x1, &x2_opp, &[-1.0_f64], 0.0).expect("run operation");
    assert!(
        (loss_2.tensor.as_slice()[0] - 0.0).abs() < 1e-10,
        "opposite y=-1 margin=0"
    );

    // ── Case 4: identical vectors, y=−1, margin=0 → max(0, 1−0)=1.0 ─────
    let loss_3 = cosine_embedding_loss(&x1, &x1, &[-1.0_f64], 0.0).expect("run operation");
    assert!(
        (loss_3.tensor.as_slice()[0] - 1.0).abs() < 1e-10,
        "identical y=-1 margin=0"
    );

    // ── Case 5: batch of 2, y=[1,1] → mean([0.0, 1.0]) = 0.5 ────────────
    // pair 0: [[1,0]] vs [[1,0]] → cos=1, loss=0
    // pair 1: [[1,0]] vs [[0,1]] → cos=0, loss=1
    let x1_b = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[1.0_f64, 0.0, 1.0, 0.0]).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let x2_b = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &[1.0_f64, 0.0, 0.0, 1.0]).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let loss_b = cosine_embedding_loss(&x1_b, &x2_b, &[1.0_f64, 1.0], 0.0).expect("run operation");
    assert!(
        (loss_b.tensor.as_slice()[0] - 0.5).abs() < 1e-10,
        "batch mean"
    );

    // ── Backward: gradients must exist when requires_grad=true ────────────
    let x1_g = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[1.0_f64, 0.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x2_g = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 2], &[0.0_f64, 1.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    cosine_embedding_loss(&x1_g, &x2_g, &[1.0_f64], 0.0).expect("run operation").backward().expect("run backward");
    assert!(x1_g.grad().is_some(), "x1 grad");
    assert!(x2_g.grad().is_some(), "x2 grad");
}

#[test]
fn test_cosine_similarity_forward_and_backward() {
    // [N=2, D=2]; row0 = (3,4)·(4,3) / (5·5) = 24/25 = 0.96;
    // row1 = (1,0)·(0,1) / (1·1) = 0.  eps is negligible at 1e-12.
    let x1_data = vec![3.0_f64, 4.0, 1.0, 0.0];
    let x2_data = vec![4.0_f64, 3.0, 0.0, 1.0];
    let x1 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &x1_data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let x2 = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2, 2], &x2_data).expect("construct tensor"),
        true,
    ).expect("construct variable");

    let out = cosine_similarity(&x1, &x2, 1, 1e-12).expect("run operation");
    assert_eq!(out.tensor.shape(), &[2]);
    let s = out.tensor.as_slice();
    assert!((s[0] - 0.96).abs() < 1e-9, "row0 cos: got {}", s[0]);
    assert!(s[1].abs() < 1e-9, "row1 cos: got {}", s[1]);

    // Backward against a central finite-difference reference on sum(cos).
    out.backward().expect("run backward");
    let analytic: Vec<f64> = x1.grad().expect("cosine x1 grad").as_slice().to_vec();
    let h = 1e-6;
    let forward_sum = |d: &[f64]| -> f64 {
        let xv = Var::new(Tensor::<f64, MoiraiBackend>::from_slice([2, 2], d).expect("construct tensor"), false).expect("construct variable");
        cosine_similarity(&xv, &x2, 1, 1e-12).expect("run operation")
            .tensor
            .as_slice()
            .iter()
            .sum::<f64>()
    };
    for i in 0..x1_data.len() {
        let mut dp = x1_data.clone();
        dp[i] += h;
        let mut dm = x1_data.clone();
        dm[i] -= h;
        let numeric = (forward_sum(&dp) - forward_sum(&dm)) / (2.0 * h);
        assert!(
            (analytic[i] - numeric).abs() < 1e-5,
            "cosine dx1[{i}]: analytic {} vs numeric {}",
            analytic[i],
            numeric
        );
    }
    assert!(x2.grad().is_some(), "cosine x2 grad");
}

#[test]
fn test_triplet_margin_with_distance_loss() {
    // distance(a,b) = mean(|a - b|). anchor=[0,0], positive=[2,2], negative=[1,1], margin=0.5.
    //   d_ap = mean(|[-2,-2]|) = 2 ; d_an = mean(|[-1,-1]|) = 1
    //   loss = mean(relu(d_ap - d_an + margin)) = relu(2 - 1 + 0.5) = 1.5.
    let anchor = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &[0.0, 0.0]).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let positive = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &[2.0, 2.0]).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let negative = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &[1.0, 1.0]).expect("construct tensor"),
        false,
    ).expect("construct variable");
    let dist = |a: &Var<f64, MoiraiBackend>, b: &Var<f64, MoiraiBackend>| {
        let difference = coeus_autograd::sub(a, b).expect("run operation");
        let absolute = coeus_autograd::abs(&difference).expect("run operation");
        coeus_autograd::mean(&absolute)
    };

    let loss = triplet_margin_with_distance_loss(&anchor, &positive, &negative, dist, 0.5).expect("run operation");
    assert_eq!(loss.tensor.shape(), &[1]);
    assert!((loss.tensor.as_slice()[0] - 1.5).abs() < 1e-12);

    loss.backward().expect("run backward");
    assert!(anchor.grad().is_some(), "triplet anchor grad");
}
