use super::support::{assert_close_slice, close};
use super::{
    celu, hardshrink, hardsigmoid, hardswish, hardtanh, softshrink, softsign, threshold,
    MoiraiBackend, Tensor, Var,
};

// ── Hardswish ──────────────────────────────────────────────────────────────

pub(super) fn hardswish_expected(x: f64) -> f64 {
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
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let output = hardswish(&input).expect("run operation");
    assert_close_slice(
        "hardswish_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward().expect("run backward");
    let grad = input.grad().expect("hardswish requires grad");
    assert_close_slice("hardswish_backward", grad.as_slice(), &expected_grad, 1e-12);
}

// ── Hardsigmoid ────────────────────────────────────────────────────────────

pub(super) fn hardsigmoid_expected(x: f64) -> f64 {
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
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let output = hardsigmoid(&input).expect("run operation");
    assert_close_slice(
        "hardsigmoid_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward().expect("run backward");
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
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let output = hardtanh(&input, lo, hi).expect("run operation");
    assert_close_slice(
        "hardtanh_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward().expect("run backward");
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
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let output = softshrink(&input, lam).expect("run operation");
    assert_close_slice(
        "softshrink_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward().expect("run backward");
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
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let output = hardshrink(&input, lam).expect("run operation");
    assert_close_slice(
        "hardshrink_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward().expect("run backward");
    let grad = input.grad().expect("hardshrink requires grad");
    assert_close_slice(
        "hardshrink_backward",
        grad.as_slice(),
        &expected_grad,
        1e-12,
    );
}

// ── Softsign ──────────────────────────────────────────────────────────────

pub(super) fn softsign_expected(x: f64) -> f64 {
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
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let output = softsign(&input).expect("run operation");
    assert_close_slice(
        "softsign_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward().expect("run backward");
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
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let output = threshold(&input, thresh, value).expect("run operation");
    assert_close_slice(
        "threshold_forward",
        output.tensor.as_slice(),
        &expected,
        1e-12,
    );
    output.backward().expect("run backward");
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
        Tensor::<f64, MoiraiBackend>::from_slice([data.len()], &data).expect("construct tensor"),
        true,
    ).expect("construct variable");
    let output = celu(&input, alpha).expect("run operation");
    assert_close_slice("celu_forward", output.tensor.as_slice(), &expected, 1e-12);
    output.backward().expect("run backward");
    let grad = input.grad().expect("celu requires grad");
    assert_close_slice("celu_backward", grad.as_slice(), &expected_grad, 1e-12);
}
