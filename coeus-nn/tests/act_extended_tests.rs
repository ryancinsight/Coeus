// ── G-037 Extended Activation family: value-semantic tests ──
//
// Each test verifies that the forward output matches the canonical formula
// and that the backward gradient (driven through `sum().backward()`) matches
// the analytic derivative at the tested sample points. Kink/subgradient
// positions are documented inline and excluded from gradient assertions
// to avoid implementation-defined subgradient ambiguity (PyTorch's
// convention is `0` at the post-kink replacement region).
//
// Reference: gap_audit G-037 acceptance criteria.

use coeus_autograd::Var;
use coeus_core::MoiraiBackend;
use coeus_nn::{
    celu, hardshrink, hardsigmoid, hardswish, hardtanh, log_sigmoid, prelu, softshrink, softsign,
    tanhshrink, threshold, Hardsigmoid, Hardswish, Module, Softsign,
};
use coeus_tensor::Tensor;

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "expected {b} got {a} (diff {:.3e})",
        (a - b).abs()
    );
}

fn assert_close_slice(label: &str, got: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (&g, &e) in got.iter().zip(expected.iter()) {
        close(g, e, tol);
    }
}

// ── Hardswish ──────────────────────────────────────────────────────────────

fn hardswish_expected(x: f64) -> f64 {
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        x
    } else {
        x * (x + 3.0) / 6.0
    }
}

fn hardswish_grad_expected(x: f64) -> f64 {
    // Match `torch.nn.functional.hardswish` CPU `hardswish_backward_kernel`
    // branch convention: dx = 0 at `x ≤ -3` (inclusive), `(2x+3)/6` for
    // `-3 < x < 3`, 1 at `x ≥ 3`. The previous exclusive lower bound
    // (`x < -3.0`) missed the kink and produced -0.5 at x = -3 instead of 0.
    if x <= -3.0 {
        0.0
    } else if x >= 3.0 {
        1.0
    } else {
        (2.0 * x + 3.0) / 6.0
    }
}

#[test]
fn hardswish_forward_and_backward() {
    let data = vec![-4.0_f64, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0];
    let expected: Vec<f64> = data.iter().map(|&x| hardswish_expected(x)).collect();
    let expected_grad: Vec<f64> = data.iter().map(|&x| hardswish_grad_expected(x)).collect();

    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = hardswish(&input);
    assert_close_slice(
        "hardswish_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward();
    let grad = input.grad().expect("hardswish requires grad");
    assert_close_slice("hardswish_backward", grad.as_slice(), &expected_grad, 1e-12);
}

// ── Hardsigmoid ────────────────────────────────────────────────────────────

fn hardsigmoid_expected(x: f64) -> f64 {
    (x / 6.0 + 0.5).clamp(0.0, 1.0)
}

fn hardsigmoid_grad_expected(x: f64) -> f64 {
    if x > -3.0 && x < 3.0 {
        1.0 / 6.0
    } else {
        0.0
    }
}

#[test]
fn hardsigmoid_forward_and_backward() {
    let data = vec![-4.0_f64, -3.0, -1.0, 0.0, 1.0, 3.0, 4.0];
    let expected: Vec<f64> = data.iter().map(|&x| hardsigmoid_expected(x)).collect();
    let expected_grad: Vec<f64> = data.iter().map(|&x| hardsigmoid_grad_expected(x)).collect();

    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = hardsigmoid(&input);
    assert_close_slice(
        "hardsigmoid_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward();
    let grad = input.grad().expect("hardsigmoid requires grad");
    assert_close_slice(
        "hardsigmoid_backward",
        grad.as_slice(),
        &expected_grad,
        1e-12,
    );
}

// ── Hardtanh ───────────────────────────────────────────────────────────────

fn hardtanh_expected(x: f64, lo: f64, hi: f64) -> f64 {
    x.clamp(lo, hi)
}

fn hardtanh_grad_expected(x: f64, lo: f64, hi: f64) -> f64 {
    if x > lo && x < hi {
        1.0
    } else {
        0.0
    }
}

#[test]
fn hardtanh_forward_and_backward() {
    let (lo, hi) = (-1.0_f64, 1.0_f64);
    let data = vec![-2.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let expected: Vec<f64> = data.iter().map(|&x| hardtanh_expected(x, lo, hi)).collect();
    let expected_grad: Vec<f64> = data
        .iter()
        .map(|&x| hardtanh_grad_expected(x, lo, hi))
        .collect();

    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = hardtanh(&input, lo, hi);
    assert_close_slice(
        "hardtanh_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward();
    let grad = input.grad().expect("hardtanh requires grad");
    assert_close_slice("hardtanh_backward", grad.as_slice(), &expected_grad, 1e-12);
    // Validate the bit-packing helper is reversible.
    let bits = coeus_autograd::pack_pairs(lo, hi);
    let lo_lo = f32::from_bits(bits as u32) as f64;
    let hi_hi = f32::from_bits((bits >> 32) as u32) as f64;
    close(lo_lo, lo, 0.0);
    close(hi_hi, hi, 0.0);
}

// ── Softshrink ────────────────────────────────────────────────────────────

fn softshrink_expected(x: f64, lam: f64) -> f64 {
    let ax = x.abs();
    if ax > lam {
        x.signum() * (ax - lam)
    } else {
        0.0
    }
}

fn softshrink_grad_expected(x: f64, lam: f64) -> f64 {
    if x.abs() > lam {
        1.0
    } else {
        0.0
    }
}

#[test]
fn softshrink_forward_and_backward() {
    let lam = 0.5_f64;
    let data = vec![-2.0_f64, -0.6, -0.4, 0.0, 0.4, 0.6, 2.0];
    let expected: Vec<f64> = data.iter().map(|&x| softshrink_expected(x, lam)).collect();
    let expected_grad: Vec<f64> = data
        .iter()
        .map(|&x| softshrink_grad_expected(x, lam))
        .collect();

    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = softshrink(&input, lam);
    assert_close_slice(
        "softshrink_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward();
    let grad = input.grad().expect("softshrink requires grad");
    assert_close_slice(
        "softshrink_backward",
        grad.as_slice(),
        &expected_grad,
        1e-12,
    );
}

// ── Hardshrink ────────────────────────────────────────────────────────────

fn hardshrink_expected(x: f64, lam: f64) -> f64 {
    if x.abs() > lam {
        x
    } else {
        0.0
    }
}

fn hardshrink_grad_expected(x: f64, lam: f64) -> f64 {
    if x.abs() > lam {
        1.0
    } else {
        0.0
    }
}

#[test]
fn hardshrink_forward_and_backward() {
    let lam = 0.5_f64;
    let data = vec![-2.0_f64, -0.6, -0.4, 0.0, 0.4, 0.6, 2.0];
    let expected: Vec<f64> = data.iter().map(|&x| hardshrink_expected(x, lam)).collect();
    let expected_grad: Vec<f64> = data
        .iter()
        .map(|&x| hardshrink_grad_expected(x, lam))
        .collect();

    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = hardshrink(&input, lam);
    assert_close_slice(
        "hardshrink_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward();
    let grad = input.grad().expect("hardshrink requires grad");
    assert_close_slice(
        "hardshrink_backward",
        grad.as_slice(),
        &expected_grad,
        1e-12,
    );
}

// ── Softsign ──────────────────────────────────────────────────────────────

fn softsign_expected(x: f64) -> f64 {
    x / (1.0 + x.abs())
}

fn softsign_grad_expected(x: f64) -> f64 {
    1.0 / ((1.0 + x.abs()) * (1.0 + x.abs()))
}

#[test]
fn softsign_forward_and_backward() {
    let data = vec![-2.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let expected: Vec<f64> = data.iter().map(|&x| softsign_expected(x)).collect();
    let expected_grad: Vec<f64> = data.iter().map(|&x| softsign_grad_expected(x)).collect();

    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = softsign(&input);
    assert_close_slice(
        "softsign_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward();
    let grad = input.grad().expect("softsign requires grad");
    assert_close_slice("softsign_backward", grad.as_slice(), &expected_grad, 1e-12);
}

// ── Threshold ────────────────────────────────────────────────────────────

fn threshold_expected(x: f64, thresh: f64, value: f64) -> f64 {
    if x > thresh {
        x
    } else {
        value
    }
}

fn threshold_grad_expected(x: f64, thresh: f64) -> f64 {
    if x > thresh {
        1.0
    } else {
        0.0
    }
}

#[test]
fn threshold_forward_and_backward() {
    let (thresh, value) = (0.0_f64, -1.0_f64);
    let data = vec![-2.0_f64, -0.5, 0.0, 0.5, 1.0, 2.0];
    let expected: Vec<f64> = data
        .iter()
        .map(|&x| threshold_expected(x, thresh, value))
        .collect();
    let expected_grad: Vec<f64> = data
        .iter()
        .map(|&x| threshold_grad_expected(x, thresh))
        .collect();

    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = threshold(&input, thresh, value);
    assert_close_slice(
        "threshold_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward();
    let grad = input.grad().expect("threshold requires grad");
    assert_close_slice("threshold_backward", grad.as_slice(), &expected_grad, 1e-12);
}

// ── Celu ─────────────────────────────────────────────────────────────────

fn celu_expected(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 {
        x
    } else {
        alpha * ((x / alpha).exp() - 1.0)
    }
}

fn celu_grad_expected(x: f64, alpha: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        (x / alpha).exp()
    }
}

#[test]
fn celu_forward_and_backward() {
    let alpha = 1.0_f64;
    let data = vec![-2.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let expected: Vec<f64> = data.iter().map(|&x| celu_expected(x, alpha)).collect();
    let expected_grad: Vec<f64> = data.iter().map(|&x| celu_grad_expected(x, alpha)).collect();

    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = celu(&input, alpha);
    assert_close_slice("celu_forward", output.tensor.as_slice(), &expected, 1e-12);
    output.backward();
    let grad = input.grad().expect("celu requires grad");
    assert_close_slice("celu_backward", grad.as_slice(), &expected_grad, 1e-12);
}

// ── PReLU ────────────────────────────────────────────────────────────────────
//
// PReLU/PyTorch contract: y = x if x > 0 else α · x; dx = 1 if x > 0 else α.
// At the kink position x = 0 the gradient equals α, matching PyTorch's
// `F.prelu` semantics (verified empirically: PReLU(0.0).backward() -> 0.25 for
// α = 0.25). The same convention applies to LeakyReLU since both share
// the underlying negative-slope contract in PyTorch.

fn prelu_expected(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        alpha * x
    }
}

fn prelu_grad_expected(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        alpha
    }
}

#[test]
fn prelu_forward_and_backward() {
    let alpha = 0.25_f64;
    // x = 0.0 is included to exercise the kink position.  Under Coeus'
    // PReLU contract the gradient at x = 0 is α (matches PyTorch).
    let data = vec![-2.0_f64, -1.0, 0.0, 0.5, 1.0];
    let expected: Vec<f64> = data.iter().map(|&x| prelu_expected(x, alpha)).collect();
    let expected_grad: Vec<f64> = data
        .iter()
        .map(|&x| prelu_grad_expected(x, alpha))
        .collect();

    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = prelu(&input, alpha);
    assert_close_slice("prelu_forward", output.tensor.as_slice(), &expected, 1e-12);
    output.backward();
    let grad = input.grad().expect("prelu requires grad");
    assert_close_slice("prelu_backward", grad.as_slice(), &expected_grad, 1e-12);
}

// ── LeakyReLU subgradient at x = 0 (documented contract parity vs PyTorch) ──

#[test]
fn leaky_relu_kink_at_zero_returns_slope() {
    // PyTorch's `F.leaky_relu(x, 0.01)` returns gradient α at x = 0.
    // Coeus' `leaky_relu` (the same LeakyReluGradTag the autograd PreluNode
    // reuses) must agree at the kink. Both ops therefore share the
    // `x > 0 ? 1 : α` predicate rather than `x >= 0`.
    let data = vec![0.0_f64];
    let slope = 0.01_f64;
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = coeus_nn::leaky_relu(&input, slope);
    assert_close_slice(
        "leaky_relu_kink_out",
        output.tensor.as_slice(),
        &data,
        1e-12,
    );
    output.backward();
    let grad = input.grad().expect("leaky_relu requires grad");
    let expected_grad = vec![slope];
    assert_close_slice("leaky_relu_kink_dx", grad.as_slice(), &expected_grad, 1e-12);
}

// ── Module-level forward smoke tests (no parameters) ────────────────────

#[test]
fn hardsigmoid_module_forward() {
    let m = Hardsigmoid;
    let data = vec![-4.0_f64, -2.0, 0.0, 2.0, 4.0];
    let expected: Vec<f64> = data.iter().map(|&x| hardsigmoid_expected(x)).collect();
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = m.forward(&input);
    assert_close_slice(
        "hardsigmoid_module_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
}

#[test]
fn hardswish_module_forward() {
    let m = Hardswish;
    let data = vec![-4.0_f64, -2.0, 0.0, 2.0, 4.0];
    let expected: Vec<f64> = data.iter().map(|&x| hardswish_expected(x)).collect();
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = m.forward(&input);
    assert_close_slice(
        "hardswish_module_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
}

#[test]
fn softsign_module_forward() {
    let m = Softsign;
    let data = vec![-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f64> = data.iter().map(|&x| softsign_expected(x)).collect();
    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = m.forward(&input);
    assert_close_slice(
        "softsign_module_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
}

// ── LogSigmoid ───────────────────────────────────────────────────────────────

#[test]
fn log_sigmoid_forward_and_backward() {
    // logsigmoid(x) = log(sigmoid(x)) = -log(1 + e^-x);
    // d/dx = sigmoid(-x) = 1 / (1 + e^x).
    let data = vec![-4.0_f64, -1.0, 0.0, 1.0, 3.0];
    let expected: Vec<f64> = data.iter().map(|&x| -(1.0 + (-x).exp()).ln()).collect();
    let expected_grad: Vec<f64> = data.iter().map(|&x| 1.0 / (1.0 + x.exp())).collect();

    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = log_sigmoid(&input);
    // Tolerance: stable identity vs coeus's `-softplus(-x)`; agreement is ~1e-15
    // for moderate x, 1e-10 is a safe margin.
    assert_close_slice(
        "log_sigmoid_forward",
        output.tensor.as_slice(),
        &expected,
        1e-10,
    );
    output.backward();
    let grad = input.grad().expect("log_sigmoid requires grad");
    assert_close_slice(
        "log_sigmoid_backward",
        grad.as_slice(),
        &expected_grad,
        1e-10,
    );
}

// ── Tanhshrink ───────────────────────────────────────────────────────────────

#[test]
fn tanhshrink_forward_and_backward() {
    // tanhshrink(x) = x - tanh(x);  d/dx = 1 - sech^2(x) = tanh^2(x).
    let data = vec![-3.0_f64, -1.0, 0.0, 0.5, 2.0];
    let expected: Vec<f64> = data.iter().map(|&x| x - x.tanh()).collect();
    let expected_grad: Vec<f64> = data.iter().map(|&x| x.tanh().powi(2)).collect();

    let input = Var::new(
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data),
        true,
    );
    let output = tanhshrink(&input);
    assert_close_slice(
        "tanhshrink_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward();
    let grad = input.grad().expect("tanhshrink requires grad");
    assert_close_slice(
        "tanhshrink_backward",
        grad.as_slice(),
        &expected_grad,
        1e-12,
    );
}
