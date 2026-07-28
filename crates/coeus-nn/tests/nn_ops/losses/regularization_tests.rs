// ── G-041 regularization module correctness tests ──

use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_nn::{AlphaDropout, FeatureAlphaDropout, GaussianNoise, LocalResponseNorm, Module};
use coeus_tensor::Tensor;

fn seq_var(shape: impl Into<coeus_core::Shape>, data: &[f32]) -> Var<f32, SequentialBackend> {
    Var::new(
        Tensor::<f32, SequentialBackend>::from_slice(shape, data),
        false,
    )
}

// ── AlphaDropout ──

#[test]
fn alpha_dropout_eval_is_identity() {
    let mut layer = AlphaDropout::new(0.5);
    layer.set_training(false);
    let x = seq_var([2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let y = layer.forward(&x);
    assert_eq!(
        y.tensor.as_slice(),
        x.tensor.as_slice(),
        "eval mode must be identity"
    );
}

#[test]
fn alpha_dropout_p0_is_identity() {
    let layer = AlphaDropout::new(0.0);
    let x = seq_var([4], &[1.0, 2.0, 3.0, 4.0]);
    let y = layer.forward(&x);
    assert_eq!(
        y.tensor.as_slice(),
        x.tensor.as_slice(),
        "p=0 must be identity"
    );
}

#[test]
fn alpha_dropout_shape_preserved() {
    let layer = AlphaDropout::new(0.3);
    let x = seq_var([2, 4, 8], &vec![1.0_f32; 64]);
    let y = layer.forward(&x);
    assert_eq!(y.tensor.shape(), &[2, 4, 8], "shape must be preserved");
}

// ── FeatureAlphaDropout ──

#[test]
fn feature_alpha_dropout_eval_is_identity() {
    let mut layer = FeatureAlphaDropout::new(0.5);
    layer.set_training(false);
    let x = seq_var([2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let y = layer.forward(&x);
    assert_eq!(y.tensor.as_slice(), x.tensor.as_slice());
}

// ── GaussianNoise ──

#[test]
fn gaussian_noise_eval_is_identity() {
    let mut layer = GaussianNoise::new(1.0);
    layer.set_training(false);
    let data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let x = seq_var([4], &data);
    let y = layer.forward(&x);
    assert_eq!(y.tensor.as_slice(), x.tensor.as_slice());
}

#[test]
fn gaussian_noise_std0_is_identity() {
    let layer = GaussianNoise::new(0.0);
    let data = vec![5.0_f32, 6.0, 7.0];
    let x = seq_var([3], &data);
    let y = layer.forward(&x);
    assert_eq!(y.tensor.as_slice(), x.tensor.as_slice());
}

#[test]
fn gaussian_noise_adds_noise_in_training() {
    let layer = GaussianNoise::new(1.0);
    let x = seq_var([100], &vec![0.0_f32; 100]);
    let y = layer.forward(&x);
    assert_eq!(y.tensor.shape(), &[100]);
    // With std=1 and 100 elements, at least some should be non-zero.
    let has_nonzero = y.tensor.as_slice().iter().any(|&v| v.abs() > 1e-7);
    assert!(has_nonzero, "noise should add non-zero values");
}

// ── LocalResponseNorm ──

#[test]
fn lrn_shape_preserved() {
    let lrn = LocalResponseNorm::new(5);
    let x = seq_var([1, 8, 4, 4], &vec![1.0_f32; 128]);
    let y = lrn.forward(&x);
    assert_eq!(y.tensor.shape(), &[1, 8, 4, 4]);
}

#[test]
fn lrn_unit_input_scales_correctly() {
    // Input all-ones [1,4,1,1], size=3, k=1.0, alpha=1.0, beta=1.0.
    // For channel c, window covers at most 3 channels.
    // Edge channels (0,3) cover 2 neighbours: sum_sq = 2, denom = (1+2/3)^1 = 5/3.
    // Inner channels (1,2) cover 3 neighbours: sum_sq = 3, denom = (1+3/3)^1 = 2.
    let lrn = LocalResponseNorm::with_params(3, 1.0, 1.0, 1.0);
    let x = seq_var([1, 4, 1, 1], &[1.0, 1.0, 1.0, 1.0]);
    let y = lrn.forward(&x);
    let s = y.tensor.as_slice();
    let expected_inner = 1.0 / 2.0_f32;
    assert!(
        (s[1] - expected_inner).abs() < 1e-5,
        "inner channel: got {}",
        s[1]
    );
    assert!(
        (s[2] - expected_inner).abs() < 1e-5,
        "inner channel: got {}",
        s[2]
    );
}

#[test]
fn lrn_k1_defaults_match_pytorch() {
    // Default LRN with size=5, alpha=0.0001, beta=0.75, k=1.
    // All-zero input → all-zero output (denominator = k^beta = 1.0).
    let lrn = LocalResponseNorm::new(5);
    let x = seq_var([1, 3, 2, 2], &[0.0_f32; 12]);
    let y = lrn.forward(&x);
    for &v in y.tensor.as_slice() {
        assert!((v).abs() < 1e-7, "zero input should give zero output");
    }
}

#[test]
fn lrn_backward_matches_numerical_gradient() {
    // The autograd-graph forward must propagate gradients to the input. Verify
    // the analytic input gradient of `sum(LRN(x))` against an f64 central
    // finite-difference reference. Window crosses channels, so the gradient
    // genuinely couples neighbouring channels (not a per-element passthrough).
    let lrn = LocalResponseNorm::with_params(3, 1.0, 0.75, 1.0);
    let data = [0.5_f64, -1.0, 2.0, 0.3, 1.5, -0.7];
    let shape = [1, 6, 1, 1];

    // Analytic: backward() seeds grad_output = ones, so x.grad = d sum(y) / d x.
    let x = Var::new(
        Tensor::<f64, SequentialBackend>::from_slice(shape, &data),
        true,
    );
    lrn.forward(&x)
        .backward()
        .expect("invariant: valid autograd fixture completes backward");
    let analytic: Vec<f64> = x.grad().expect("lrn input gradient").as_slice().to_vec();

    // Numerical: central differences of sum(LRN(x)). f64, h=1e-6 ⇒ error ~1e-10;
    // a non-zero coupled gradient must match within a wide 1e-5 margin.
    let h = 1e-6_f64;
    let sum_forward = |d: &[f64]| -> f64 {
        let xv = Var::new(
            Tensor::<f64, SequentialBackend>::from_slice(shape, d),
            false,
        );
        lrn.forward(&xv)
            .tensor
            .as_slice()
            .iter()
            .copied()
            .sum::<f64>()
    };
    for i in 0..data.len() {
        let mut dp = data.to_vec();
        dp[i] += h;
        let mut dm = data.to_vec();
        dm[i] -= h;
        let numeric = (sum_forward(&dp) - sum_forward(&dm)) / (2.0 * h);
        assert!(
            (analytic[i] - numeric).abs() < 1e-5,
            "grad[{i}]: analytic {} vs numeric {} (diff {:.2e})",
            analytic[i],
            numeric,
            (analytic[i] - numeric).abs()
        );
    }
    // Gradient must be genuinely non-trivial (not the forward-only dx=0 bug).
    assert!(
        analytic.iter().any(|g| g.abs() > 1e-6),
        "LRN input gradient is all-zero — backward not propagating"
    );
}
