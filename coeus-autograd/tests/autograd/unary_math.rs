use coeus_autograd::{scalar_div, scalar_sub, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn test_neg_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![4], &[1.0f64, -2.0, 3.0, 0.0], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::neg(&x);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - (-1.0)).abs() < 1e-10);
    assert!((y_slice[1] - 2.0).abs() < 1e-10);
    assert!((y_slice[2] - (-3.0)).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![4], &[1.0f64, 2.0, 3.0, 4.0], &backend);
    y.backward_with_seed(seed);
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
    let x_val = Tensor::from_slice_on(vec![3], &[3.0f64, -2.0, 0.5], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::abs(&x);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 3.0).abs() < 1e-10);
    assert!((y_slice[1] - 2.0).abs() < 1e-10);
    assert!((y_slice[2] - 0.5).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 1.0, 1.0], &backend);
    y.backward_with_seed(seed);
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
    let x_val = Tensor::from_slice_on(vec![3], &[4.0f64, 9.0, 16.0], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::sqrt(&x);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 2.0).abs() < 1e-9);
    assert!((y_slice[1] - 3.0).abs() < 1e-9);
    assert!((y_slice[2] - 4.0).abs() < 1e-9);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 1.0, 1.0], &backend);
    y.backward_with_seed(seed);
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
    let x_val = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::pow(&x, 3.0);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 1.0).abs() < 1e-8);
    assert!((y_slice[1] - 8.0).abs() < 1e-8);
    assert!((y_slice[2] - 27.0).abs() < 1e-8);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 1.0, 1.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!((gx_s[0] - 3.0).abs() < 1e-6, "pow(1,3) grad: {}", gx_s[0]);
    assert!((gx_s[1] - 12.0).abs() < 1e-6, "pow(2,3) grad: {}", gx_s[1]);
    assert!((gx_s[2] - 27.0).abs() < 1e-6, "pow(3,3) grad: {}", gx_s[2]);
}

#[test]
fn test_clamp_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![4], &[-1.0f64, 0.5, 1.5, 2.5], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::clamp(&x, 0.0f64, 2.0f64);
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

    let seed = Tensor::from_slice_on(vec![4], &[1.0f64, 1.0, 1.0, 1.0], &backend);
    y.backward_with_seed(seed);
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
    let x_val = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend);
    let x = Var::new(x_val, true);

    let y = coeus_autograd::scalar_mul(&x, 3.0f64);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 3.0).abs() < 1e-10);
    assert!((y_slice[1] - 6.0).abs() < 1e-10);
    assert!((y_slice[2] - 9.0).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!((gx_s[0] - 3.0).abs() < 1e-10);
    assert!((gx_s[1] - 6.0).abs() < 1e-10);
    assert!((gx_s[2] - 9.0).abs() < 1e-10);
}

#[test]
fn test_scalar_sub_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[5.0f64, 8.0, 12.0], &backend);
    let x = Var::new(x_val, true);

    let y = scalar_sub(&x, 3.0);
    let z = &x - 3.0;

    assert!((y.tensor.as_slice()[0] - 2.0).abs() < 1e-10);
    assert!((y.tensor.as_slice()[1] - 5.0).abs() < 1e-10);
    assert!((y.tensor.as_slice()[2] - 9.0).abs() < 1e-10);

    assert!((z.tensor.as_slice()[0] - 2.0).abs() < 1e-10);
    assert!((z.tensor.as_slice()[1] - 5.0).abs() < 1e-10);
    assert!((z.tensor.as_slice()[2] - 9.0).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend);
    y.backward_with_seed(seed.clone());
    let gx = x.grad().unwrap();
    assert!((gx.as_slice()[0] - 1.0).abs() < 1e-10);
    assert!((gx.as_slice()[1] - 2.0).abs() < 1e-10);
    assert!((gx.as_slice()[2] - 3.0).abs() < 1e-10);
}

#[test]
fn test_scalar_div_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[6.0f64, 12.0, 18.0], &backend);
    let x = Var::new(x_val, true);

    let y = scalar_div(&x, 3.0);
    let z = &x / 3.0;

    assert!((y.tensor.as_slice()[0] - 2.0).abs() < 1e-10);
    assert!((y.tensor.as_slice()[1] - 4.0).abs() < 1e-10);
    assert!((y.tensor.as_slice()[2] - 6.0).abs() < 1e-10);

    assert!((z.tensor.as_slice()[0] - 2.0).abs() < 1e-10);
    assert!((z.tensor.as_slice()[1] - 4.0).abs() < 1e-10);
    assert!((z.tensor.as_slice()[2] - 6.0).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    assert!((gx.as_slice()[0] - 1.0 / 3.0).abs() < 1e-10);
    assert!((gx.as_slice()[1] - 2.0 / 3.0).abs() < 1e-10);
    assert!((gx.as_slice()[2] - 3.0 / 3.0).abs() < 1e-10);
}
