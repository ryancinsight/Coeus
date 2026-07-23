//! Classification and margin loss contracts.

use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::hinge_embedding_loss;
use coeus_nn::multi_label_soft_margin_loss;
use coeus_nn::multi_margin;
use coeus_nn::nll_loss;
use coeus_nn::soft_margin;
use coeus_tensor::Tensor;

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
        .zip(
            input_grad
                .as_slice()
                .iter()
                .zip(target_grad.as_slice().iter()),
        )
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
fn test_multi_margin() {
    // x=[[0.5, 0.8, -0.6]], target=[0], margin=1, p=1, C=3.
    // j=1: m=1-0.5+0.8=1.3>0 (active); j=2: m=1-0.5-0.6=-0.1<0 (inactive).
    // loss = 1.3 / (N*C) = 1.3/3.  grads: x[0,1]=1/3, x[0,2]=0, x[0,0]=-1/3.
    let x = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 3], &[0.5, 0.8, -0.6]),
        true,
    );
    let loss = multi_margin(&x, &[0], 1.0, 1.0);
    assert_eq!(loss.tensor.shape(), &[1]);
    assert!(
        (loss.tensor.as_slice()[0] - 1.3 / 3.0).abs() <= 1e-12,
        "multi_margin forward: got {}, expected {}",
        loss.tensor.as_slice()[0],
        1.3 / 3.0
    );

    loss.backward();
    let g = x.grad().expect("x must receive a gradient");
    let third = 1.0 / 3.0;
    let expected = [-third, third, 0.0];
    for (k, (&got, &e)) in g.as_slice().iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - e).abs() <= 1e-12,
            "multi_margin grad[{k}]: got {got}, expected {e}"
        );
    }
}

#[test]
fn test_multi_margin_all_inactive() {
    // Target score dominates by > margin → all hinges inactive → loss 0.
    let x = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([1, 3], &[3.0, 0.5, 0.1]),
        false,
    );
    let loss = multi_margin(&x, &[0], 1.0, 1.0);
    assert_eq!(
        loss.tensor.as_slice(),
        &[0.0_f64],
        "dominant target → loss 0"
    );
}

#[test]
fn test_hinge_embedding_loss() {
    // x = [0.5, 2, -1, 0.3], target = [+1, -1, +1, -1], margin = 1.
    // PyTorch HingeEmbeddingLoss: y=+1 -> loss = x (identity, no clamp);
    // y=-1 -> loss = max(0, margin - x). (The prior assertions encoded a wrong
    // formula and are corrected here to torch's documented contract.)
    //   0.5, max(0,1-2)=0, -1.0, max(0,1-0.3)=0.7  =>  mean = 0.2 / 4 = 0.05.
    let x = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([4], &[0.5, 2.0, -1.0, 0.3]),
        true,
    );
    let target = [1.0f64, -1.0, 1.0, -1.0];

    let loss = hinge_embedding_loss(&x, &target, 1.0);
    assert_eq!(loss.tensor.shape(), &[1]);
    assert!(
        (loss.tensor.as_slice()[0] - 0.05).abs() < 1e-12,
        "fwd: {} vs 0.05",
        loss.tensor.as_slice()[0]
    );

    loss.backward();
    // d/dx (seed 1/N from mean, N=4): identity branch (y=+1) -> 1/N; hinge
    // branch (y=-1) -> -1/N when margin > x else 0.
    //   i0: y=+1 -> 0.25            i1: y=-1, 1-2<0 -> 0
    //   i2: y=+1 -> 0.25            i3: y=-1, 1-0.3>0 -> -0.25
    let grad = x.grad().expect("hinge x grad");
    let g = grad.as_slice();
    let expected = [0.25, 0.0, 0.25, -0.25];
    for (i, (&got, &want)) in g.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-12,
            "hinge grad[{i}]: {got} vs {want}"
        );
    }
}

#[test]
fn test_multi_label_soft_margin_loss() {
    // MultiLabelSoftMarginLoss(mean) == BCEWithLogits(mean). x=[0,2], y=[1,0].
    // Per-element BCE: -[y ln σ(x) + (1-y) ln(1-σ(x))].
    //   i0: x=0, σ=0.5, y=1 -> -ln(0.5) = ln 2
    //   i1: x=2, y=0        -> -ln(1-σ(2))
    let x = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &[0.0, 2.0]),
        true,
    );
    let target = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([2], &[1.0, 0.0]),
        false,
    );

    let loss = multi_label_soft_margin_loss(&x, &target);
    assert_eq!(loss.tensor.shape(), &[1]);
    let sig2 = 1.0 / (1.0 + (-2.0f64).exp());
    let expected = (2.0f64.ln() + -(1.0 - sig2).ln()) / 2.0;
    assert!((loss.tensor.as_slice()[0] - expected).abs() < 1e-10);

    loss.backward();
    // d BCEWithLogits/dx = (σ(x) - y)/N: i0 (0.5-1)/2=-0.25, i1 (σ(2)-0)/2.
    let grad = x.grad().expect("mlsm x grad");
    let g = grad.as_slice();
    assert!((g[0] - (-0.25)).abs() < 1e-10, "mlsm grad0: {}", g[0]);
    assert!((g[1] - sig2 / 2.0).abs() < 1e-10, "mlsm grad1: {}", g[1]);
}
