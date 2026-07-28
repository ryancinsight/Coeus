use coeus_autograd::{scalar_div, scalar_sub, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn test_neg_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![4], &[1.0f64, -2.0, 3.0, 0.0], &backend).expect("valid tensor construction");
    let x = Var::new(x_val, true).expect("valid variable construction");

    let y = coeus_autograd::neg(&x).expect("valid autograd operation");
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - (-1.0)).abs() < 1e-10);
    assert!((y_slice[1] - 2.0).abs() < 1e-10);
    assert!((y_slice[2] - (-3.0)).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![4], &[1.0f64, 2.0, 3.0, 4.0], &backend).expect("valid tensor construction");
    y.backward_with_seed(seed).expect("valid backward propagation");
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!((gx_s[0] - (-1.0)).abs() < 1e-10);
    assert!((gx_s[1] - (-2.0)).abs() < 1e-10);
    assert!((gx_s[2] - (-3.0)).abs() < 1e-10);
    assert!((gx_s[3] - (-4.0)).abs() < 1e-10);
}

#[test]
fn test_abs_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[3.0f64, -2.0, 0.5], &backend).expect("valid tensor construction");
    let x = Var::new(x_val, true).expect("valid variable construction");

    let y = coeus_autograd::abs(&x).expect("valid autograd operation");
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 3.0).abs() < 1e-10);
    assert!((y_slice[1] - 2.0).abs() < 1e-10);
    assert!((y_slice[2] - 0.5).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 1.0, 1.0], &backend).expect("valid tensor construction");
    y.backward_with_seed(seed).expect("valid backward propagation");
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!(
        (gx_s[0] - 1.0).abs() < 1e-10,
        "expected +1 for x>0, got {}",
        gx_s[0]
    );
    assert!(
        (gx_s[1] - (-1.0)).abs() < 1e-10,
        "expected -1 for x<0, got {}",
        gx_s[1]
    );
    assert!(
        (gx_s[2] - 1.0).abs() < 1e-10,
        "expected +1 for x>0, got {}",
        gx_s[2]
    );
}

#[test]
fn test_sqrt_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[4.0f64, 9.0, 16.0], &backend).expect("valid tensor construction");
    let x = Var::new(x_val, true).expect("valid variable construction");

    let y = coeus_autograd::sqrt(&x).expect("valid autograd operation");
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 2.0).abs() < 1e-9);
    assert!((y_slice[1] - 3.0).abs() < 1e-9);
    assert!((y_slice[2] - 4.0).abs() < 1e-9);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 1.0, 1.0], &backend).expect("valid tensor construction");
    y.backward_with_seed(seed).expect("valid backward propagation");
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!(
        (gx_s[0] - 0.25).abs() < 1e-9,
        "sqrt backward at 4: {}",
        gx_s[0]
    );
    assert!(
        (gx_s[1] - (1.0 / 6.0)).abs() < 1e-9,
        "sqrt backward at 9: {}",
        gx_s[1]
    );
    assert!(
        (gx_s[2] - 0.125).abs() < 1e-9,
        "sqrt backward at 16: {}",
        gx_s[2]
    );
}

#[test]
fn test_pow_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend).expect("valid tensor construction");
    let x = Var::new(x_val, true).expect("valid variable construction");

    let y = coeus_autograd::pow(&x, 3.0).expect("valid autograd operation");
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 1.0).abs() < 1e-8);
    assert!((y_slice[1] - 8.0).abs() < 1e-8);
    assert!((y_slice[2] - 27.0).abs() < 1e-8);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 1.0, 1.0], &backend).expect("valid tensor construction");
    y.backward_with_seed(seed).expect("valid backward propagation");
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!((gx_s[0] - 3.0).abs() < 1e-6, "pow(1,3) grad: {}", gx_s[0]);
    assert!((gx_s[1] - 12.0).abs() < 1e-6, "pow(2,3) grad: {}", gx_s[1]);
    assert!((gx_s[2] - 27.0).abs() < 1e-6, "pow(3,3) grad: {}", gx_s[2]);
}

/// PyTorch `Tensor.pow(scalar)` is sign-preserving when `scalar` is integer-valued:
///   (-1.0)^3 = -1.0, (-2.0)^3 = -8.0, (0.5)^3 = 0.125.
/// Backward: d/dx x^k = k · x^(k-1), with the same sign convention.
/// The previous `exp(n·ln(x))` composition returns NaN for x ≤ 0; this test pins
/// the integer-exponent forward+backward-fix introduced for parity with PyTorch.
#[test]
fn test_pow_integer_exp_sign_preserving() {
    let backend = MoiraiBackend::new();
    // Covers negative base (x = -1, -2), zero (k > 1), fractional (0.5), and positive.
    let data: [f64; 5] = [1.0, 2.0, -1.0, 0.5, -2.0];
    let x_val = Tensor::from_slice_on(vec![5], &data, &backend).expect("valid tensor construction");
    let x = Var::new(x_val, true).expect("valid variable construction");

    // Forward: 1, 8, -1, 0.125, -8.
    let y = coeus_autograd::pow(&x, 3.0).expect("valid autograd operation");
    let fwd = y.tensor.as_slice();
    assert!((fwd[0] - 1.0).abs() < 1e-10, "fwd[0] = {}", fwd[0]);
    assert!((fwd[1] - 8.0).abs() < 1e-10, "fwd[1] = {}", fwd[1]);
    assert!((fwd[2] - (-1.0)).abs() < 1e-10, "fwd[2] = {}", fwd[2]);
    assert!((fwd[3] - 0.125).abs() < 1e-10, "fwd[3] = {}", fwd[3]);
    assert!((fwd[4] - (-8.0)).abs() < 1e-10, "fwd[4] = {}", fwd[4]);

    // Backward: d/dx x^3 = 3·x^2 — non-negative everywhere; for x = 0 → 0.
    // PyTorch: 3, 12, 3, 0.75, 12 (and 0 at the x = 0 zero index, which we excluded).
    let seed = Tensor::from_slice_on(vec![5], &[1.0f64; 5], &backend).expect("valid tensor construction");
    y.backward_with_seed(seed).expect("valid backward propagation");
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!((gx_s[0] - 3.0).abs() < 1e-10, "bwd[0] = {}", gx_s[0]);
    assert!((gx_s[1] - 12.0).abs() < 1e-10, "bwd[1] = {}", gx_s[1]);
    assert!((gx_s[2] - 3.0).abs() < 1e-10, "bwd[2] = {}", gx_s[2]);
    assert!((gx_s[3] - 0.75).abs() < 1e-10, "bwd[3] = {}", gx_s[3]);
    assert!((gx_s[4] - 12.0).abs() < 1e-10, "bwd[4] = {}", gx_s[4]);
}

/// PyTorch `Tensor.pow(x, 0.0)` = 1 everywhere with d/dx = 0.
#[test]
fn test_pow_integer_exp_zero() {
    let backend = MoiraiBackend::new();
    let data: [f64; 4] = [1.0, -1.0, 0.0, 2.5];
    let x_val = Tensor::from_slice_on(vec![4], &data, &backend).expect("valid tensor construction");
    let x = Var::new(x_val, true).expect("valid variable construction");

    let y = coeus_autograd::pow(&x, 0.0).expect("valid autograd operation");
    let fwd = y.tensor.as_slice();
    for v in fwd.iter() {
        assert!((v - 1.0).abs() < 1e-10, "pow(x, 0) = {} (expected 1)", v);
    }

    let seed = Tensor::from_slice_on(vec![4], &[1.0f64; 4], &backend).expect("valid tensor construction");
    y.backward_with_seed(seed).expect("valid backward propagation");
    let gx = x.grad().unwrap();
    for v in gx.as_slice().iter() {
        assert!(v.abs() < 1e-10, "d/dx x^0 = {} (expected 0)", v);
    }
}

/// PyTorch `Tensor.pow(scalar)` with a non-integer exponent follows IEEE,
/// yielding NaN for negative base.  Coeus matches by composition `exp(n·ln(x))`.
/// Pin that behavior to keep the fractional path from regressing.
#[test]
fn test_pow_fractional_exp_negative_base_nan() {
    let backend = MoiraiBackend::new();
    let data: [f64; 3] = [4.0, -1.0, 9.0];
    let x_val = Tensor::from_slice_on(vec![3], &data, &backend).expect("valid tensor construction");
    let x = Var::new(x_val, true).expect("valid variable construction");

    // 4^0.5 = 2, (-1)^0.5 = NaN (PyTorch), 9^0.5 = 3.
    let y = coeus_autograd::pow(&x, 0.5).expect("valid autograd operation");
    let fwd = y.tensor.as_slice();
    assert!((fwd[0] - 2.0).abs() < 1e-10, "pow(4, 0.5) = {}", fwd[0]);
    assert!(
        fwd[1].is_nan(),
        "pow(-1, 0.5) should be NaN, got {}",
        fwd[1]
    );
    assert!((fwd[2] - 3.0).abs() < 1e-10, "pow(9, 0.5) = {}", fwd[2]);
}

#[test]
fn test_clamp_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![4], &[-1.0f64, 0.5, 1.5, 2.5], &backend).expect("valid tensor construction");
    let x = Var::new(x_val, true).expect("valid variable construction");

    let y = coeus_autograd::clamp(&x, 0.0f64, 2.0f64)
        .expect("valid autograd operation");
    let y_slice = y.tensor.as_slice();
    assert!(
        (y_slice[0] - 0.0).abs() < 1e-10,
        "clamp(-1) = {}",
        y_slice[0]
    );
    assert!(
        (y_slice[1] - 0.5).abs() < 1e-10,
        "clamp(0.5) = {}",
        y_slice[1]
    );
    assert!(
        (y_slice[2] - 1.5).abs() < 1e-10,
        "clamp(1.5) = {}",
        y_slice[2]
    );
    assert!(
        (y_slice[3] - 2.0).abs() < 1e-10,
        "clamp(2.5) = {}",
        y_slice[3]
    );

    let seed = Tensor::from_slice_on(vec![4], &[1.0f64, 1.0, 1.0, 1.0], &backend).expect("valid tensor construction");
    y.backward_with_seed(seed).expect("valid backward propagation");
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!(
        (gx_s[0] - 0.0).abs() < 1e-10,
        "clamp grad at -1: {}",
        gx_s[0]
    );
    assert!(
        (gx_s[1] - 1.0).abs() < 1e-10,
        "clamp grad at 0.5: {}",
        gx_s[1]
    );
    assert!(
        (gx_s[2] - 1.0).abs() < 1e-10,
        "clamp grad at 1.5: {}",
        gx_s[2]
    );
    assert!(
        (gx_s[3] - 0.0).abs() < 1e-10,
        "clamp grad at 2.5: {}",
        gx_s[3]
    );
}

#[test]
fn test_scalar_mul_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend).expect("valid tensor construction");
    let x = Var::new(x_val, true).expect("valid variable construction");

    let y = coeus_autograd::scalar_mul(&x, 3.0f64)
        .expect("valid autograd operation");
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 3.0).abs() < 1e-10);
    assert!((y_slice[1] - 6.0).abs() < 1e-10);
    assert!((y_slice[2] - 9.0).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend).expect("valid tensor construction");
    y.backward_with_seed(seed).expect("valid backward propagation");
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!((gx_s[0] - 3.0).abs() < 1e-10);
    assert!((gx_s[1] - 6.0).abs() < 1e-10);
    assert!((gx_s[2] - 9.0).abs() < 1e-10);
}

#[test]
fn test_scalar_sub_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[5.0f64, 8.0, 12.0], &backend).expect("valid tensor construction");
    let x = Var::new(x_val, true).expect("valid variable construction");

    let y = scalar_sub(&x, 3.0).expect("valid autograd operation");
    let z = (&x - 3.0).expect("valid autograd operation");

    assert!((y.tensor.as_slice()[0] - 2.0).abs() < 1e-10);
    assert!((y.tensor.as_slice()[1] - 5.0).abs() < 1e-10);
    assert!((y.tensor.as_slice()[2] - 9.0).abs() < 1e-10);

    assert!((z.tensor.as_slice()[0] - 2.0).abs() < 1e-10);
    assert!((z.tensor.as_slice()[1] - 5.0).abs() < 1e-10);
    assert!((z.tensor.as_slice()[2] - 9.0).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend).expect("valid tensor construction");
    y.backward_with_seed(seed.clone()).expect("valid backward propagation");
    let gx = x.grad().unwrap();
    assert!((gx.as_slice()[0] - 1.0).abs() < 1e-10);
    assert!((gx.as_slice()[1] - 2.0).abs() < 1e-10);
    assert!((gx.as_slice()[2] - 3.0).abs() < 1e-10);
}

#[test]
fn test_scalar_div_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[6.0f64, 12.0, 18.0], &backend).expect("valid tensor construction");
    let x = Var::new(x_val, true).expect("valid variable construction");

    let y = scalar_div(&x, 3.0).expect("valid autograd operation");
    let z = (&x / 3.0).expect("valid autograd operation");

    assert!((y.tensor.as_slice()[0] - 2.0).abs() < 1e-10);
    assert!((y.tensor.as_slice()[1] - 4.0).abs() < 1e-10);
    assert!((y.tensor.as_slice()[2] - 6.0).abs() < 1e-10);

    assert!((z.tensor.as_slice()[0] - 2.0).abs() < 1e-10);
    assert!((z.tensor.as_slice()[1] - 4.0).abs() < 1e-10);
    assert!((z.tensor.as_slice()[2] - 6.0).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend).expect("valid tensor construction");
    y.backward_with_seed(seed).expect("valid backward propagation");
    let gx = x.grad().unwrap();
    assert!((gx.as_slice()[0] - 1.0 / 3.0).abs() < 1e-10);
    assert!((gx.as_slice()[1] - 2.0 / 3.0).abs() < 1e-10);
    assert!((gx.as_slice()[2] - 3.0 / 3.0).abs() < 1e-10);
}
