use coeus_autograd::{cosh, erf, erfc, exp, exp2, log, log10, log2, selu, sinh, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn test_exp_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[0.0f32, 1.0f32, 2.0f32], &backend);
    let x = Var::new(x_val, true);

    let y = exp(&x);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 1.0).abs() < 1e-5);
    assert!((y_slice[1] - std::f32::consts::E).abs() < 1e-5);
    assert!((y_slice[2] - 7.389056).abs() < 1e-5);

    let grad_out = Tensor::from_slice_on(vec![3], &[1.0f32, 2.0f32, 3.0f32], &backend);
    y.backward_with_seed(grad_out)
        .expect("invariant: valid autograd fixture completes backward");

    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    assert!((gx_slice[0] - 1.0).abs() < 1e-5);
    assert!((gx_slice[1] - 5.4365636).abs() < 1e-5);
    assert!((gx_slice[2] - 22.167168).abs() < 1e-5);
}

#[test]
fn test_exp2_autograd() {
    let backend = MoiraiBackend::new();
    let data = [0.0f64, 1.0, 2.0, -1.0];
    let x = Var::new(Tensor::from_slice_on(vec![4], &data, &backend), true);
    let y = exp2(&x);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 1.0).abs() < 1e-12, "exp2(0) = 1");
    assert!((y_slice[1] - 2.0).abs() < 1e-12, "exp2(1) = 2");
    assert!((y_slice[2] - 4.0).abs() < 1e-12, "exp2(2) = 4");
    let grad_seed = Tensor::from_slice_on(vec![4], &[1.0f64; 4], &backend);
    y.backward_with_seed(grad_seed)
        .expect("invariant: valid autograd fixture completes backward");
    let ln2 = core::f64::consts::LN_2;
    let gx = x.grad().unwrap();
    for (i, &xi) in data.iter().enumerate() {
        let expected = 2.0f64.powf(xi) * ln2;
        assert!(
            (gx.as_slice()[i] - expected).abs() < 1e-12,
            "d exp2/dx[{i}]"
        );
    }
}

#[test]
fn test_selu_autograd() {
    let backend = MoiraiBackend::new();
    let data = [-2.0f64, -1.0, 0.0, 1.0, 2.0];
    let x = Var::new(Tensor::from_slice_on(vec![5], &data, &backend), true);
    let y = selu(&x);
    let y_slice = y.tensor.as_slice();
    let alpha = 1.673_263_242_354_377_2f64;
    let scale = 1.050_700_987_355_480_5f64;
    for (i, &xi) in data.iter().enumerate() {
        let expected = if xi > 0.0 {
            scale * xi
        } else {
            scale * alpha * (xi.exp() - 1.0)
        };
        assert!((y_slice[i] - expected).abs() < 1e-12, "selu[{i}]");
    }
    let grad_seed = Tensor::from_slice_on(vec![5], &[1.0f64; 5], &backend);
    y.backward_with_seed(grad_seed)
        .expect("invariant: valid autograd fixture completes backward");
    let gx = x.grad().unwrap();
    for (i, &xi) in data.iter().enumerate() {
        let expected = if xi > 0.0 {
            scale
        } else {
            scale * alpha * xi.exp()
        };
        assert!(
            (gx.as_slice()[i] - expected).abs() < 1e-12,
            "d selu/dx[{i}]"
        );
    }
}

#[test]
fn test_erf_autograd() {
    let backend = MoiraiBackend::new();
    // f64 to hold the reference values to full double precision.
    let x_val = Tensor::from_slice_on(vec![4], &[0.0f64, 1.0, -1.0, 2.0], &backend);
    let x = Var::new(x_val, true);

    let y = erf(&x);
    let y_slice = y.tensor.as_slice();
    // Reference: erf(0) = 0; erf(1) = 0.8427007929497149 (Abramowitz & Stegun
    // 7.1.1); erf is odd; erf(2) = 0.9953222650189527.
    assert!(y_slice[0].abs() < 1e-15, "erf(0): {}", y_slice[0]);
    assert!((y_slice[1] - 0.842_700_792_949_714_9).abs() < 1e-12);
    assert!((y_slice[2] + 0.842_700_792_949_714_9).abs() < 1e-12, "odd");
    assert!((y_slice[3] - 0.995_322_265_018_952_7).abs() < 1e-12);

    let grad_out = Tensor::from_slice_on(vec![4], &[1.0f64, 1.0, 1.0, 1.0], &backend);
    y.backward_with_seed(grad_out)
        .expect("invariant: valid autograd fixture completes backward");

    // Analytic gradient: d/dx erf(x) = (2/√π)·e^(−x²).
    let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    for (i, &xi) in [0.0f64, 1.0, -1.0, 2.0].iter().enumerate() {
        let expected = two_over_sqrt_pi * (-xi * xi).exp();
        assert!(
            (gx_slice[i] - expected).abs() < 1e-12,
            "d erf/dx at {xi}: {} vs {expected}",
            gx_slice[i]
        );
    }

    // Numerical-gradient cross-check (central differences, h = 1e-6): the
    // analytic backward must agree with the finite-difference slope of the
    // forward to O(h²) ≈ 1e-12, giving an implementation-independent oracle.
    let h = 1e-6f64;
    for &xi in &[0.5f64, -1.5] {
        let xp = Var::new(Tensor::from_slice_on(vec![1], &[xi + h], &backend), false);
        let xm = Var::new(Tensor::from_slice_on(vec![1], &[xi - h], &backend), false);
        let numeric = (erf(&xp).tensor.as_slice()[0] - erf(&xm).tensor.as_slice()[0]) / (2.0 * h);
        let analytic = two_over_sqrt_pi * (-xi * xi).exp();
        assert!(
            (numeric - analytic).abs() < 1e-9,
            "numeric {numeric} vs analytic {analytic} at {xi}"
        );
    }
}

#[test]
fn test_erfc_autograd() {
    let backend = MoiraiBackend::new();
    let data = [0.0f64, 1.0, -1.0, 2.0];
    let x = Var::new(Tensor::from_slice_on(vec![4], &data, &backend), true);
    let y = erfc(&x);
    let y_slice = y.tensor.as_slice();
    // erfc(x) = 1 - erf(x)
    let erf_ref = [
        0.0f64,
        0.842_700_792_949_714_9,
        -0.842_700_792_949_714_9,
        0.995_322_265_018_952_7,
    ];
    for i in 0..4 {
        assert!((y_slice[i] - (1.0 - erf_ref[i])).abs() < 1e-12, "erfc[{i}]");
    }
    let grad_seed = Tensor::from_slice_on(vec![4], &[1.0f64; 4], &backend);
    y.backward_with_seed(grad_seed)
        .expect("invariant: valid autograd fixture completes backward");
    let two_over_sqrt_pi = 2.0 / std::f64::consts::PI.sqrt();
    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    for (i, &xi) in data.iter().enumerate() {
        let expected = -two_over_sqrt_pi * (-xi * xi).exp();
        assert!((gx_slice[i] - expected).abs() < 1e-12, "d erfc/dx[{i}]");
    }
}

#[test]
fn test_log_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[1.0f32, 2.0f32, 4.0f32], &backend);
    let x = Var::new(x_val, true);

    let y = log(&x);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 0.0).abs() < 1e-5);
    assert!((y_slice[1] - std::f32::consts::LN_2).abs() < 1e-5);
    assert!((y_slice[2] - 2.0f32 * std::f32::consts::LN_2).abs() < 1e-5);

    let grad_out = Tensor::from_slice_on(vec![3], &[1.0f32, 2.0f32, 3.0f32], &backend);
    y.backward_with_seed(grad_out)
        .expect("invariant: valid autograd fixture completes backward");

    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    assert!((gx_slice[0] - 1.0).abs() < 1e-5);
    assert!((gx_slice[1] - 1.0).abs() < 1e-5);
    assert!((gx_slice[2] - 0.75).abs() < 1e-5);
}

#[test]
fn test_sinh_autograd() {
    let backend = MoiraiBackend::new();
    let data = [-2.0f64, -1.0, 0.0, 1.0, 2.0];
    let x = Var::new(Tensor::from_slice_on(vec![5], &data, &backend), true);

    let y = sinh(&x);
    let y_slice = y.tensor.as_slice();
    for (i, &xi) in data.iter().enumerate() {
        assert!((y_slice[i] - xi.sinh()).abs() < 1e-12, "sinh[{i}]");
    }

    let grad_seed = Tensor::from_slice_on(vec![5], &[1.0f64; 5], &backend);
    y.backward_with_seed(grad_seed)
        .expect("invariant: valid autograd fixture completes backward");
    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    for (i, &xi) in data.iter().enumerate() {
        assert!((gx_slice[i] - xi.cosh()).abs() < 1e-12, "d sinh/dx[{i}]");
    }
}

#[test]
fn test_cosh_autograd() {
    let backend = MoiraiBackend::new();
    let data = [-2.0f64, -1.0, 0.0, 1.0, 2.0];
    let x = Var::new(Tensor::from_slice_on(vec![5], &data, &backend), true);

    let y = cosh(&x);
    let y_slice = y.tensor.as_slice();
    for (i, &xi) in data.iter().enumerate() {
        assert!((y_slice[i] - xi.cosh()).abs() < 1e-12, "cosh[{i}]");
    }

    let grad_seed = Tensor::from_slice_on(vec![5], &[1.0f64; 5], &backend);
    y.backward_with_seed(grad_seed)
        .expect("invariant: valid autograd fixture completes backward");
    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    for (i, &xi) in data.iter().enumerate() {
        assert!((gx_slice[i] - xi.sinh()).abs() < 1e-12, "d cosh/dx[{i}]");
    }
}

#[test]
fn test_log2_autograd() {
    let backend = MoiraiBackend::new();
    let data = [0.5f64, 1.0, 2.0, 4.0, 8.0];
    let x = Var::new(Tensor::from_slice_on(vec![5], &data, &backend), true);

    let y = log2(&x);
    let y_slice = y.tensor.as_slice();
    for (i, &xi) in data.iter().enumerate() {
        assert!((y_slice[i] - xi.log2()).abs() < 1e-12, "log2[{i}]");
    }

    let grad_seed = Tensor::from_slice_on(vec![5], &[1.0f64; 5], &backend);
    y.backward_with_seed(grad_seed)
        .expect("invariant: valid autograd fixture completes backward");
    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    for (i, &xi) in data.iter().enumerate() {
        let expected = 1.0 / (xi * core::f64::consts::LN_2);
        assert!((gx_slice[i] - expected).abs() < 1e-12, "d log2/dx[{i}]");
    }
}

#[test]
fn test_log10_autograd() {
    let backend = MoiraiBackend::new();
    let data = [0.1f64, 1.0, 10.0, 100.0, 1000.0];
    let x = Var::new(Tensor::from_slice_on(vec![5], &data, &backend), true);

    let y = log10(&x);
    let y_slice = y.tensor.as_slice();
    for (i, &xi) in data.iter().enumerate() {
        assert!((y_slice[i] - xi.log10()).abs() < 1e-12, "log10[{i}]");
    }

    let grad_seed = Tensor::from_slice_on(vec![5], &[1.0f64; 5], &backend);
    y.backward_with_seed(grad_seed)
        .expect("invariant: valid autograd fixture completes backward");
    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    for (i, &xi) in data.iter().enumerate() {
        let expected = 1.0 / (xi * core::f64::consts::LN_10);
        assert!((gx_slice[i] - expected).abs() < 1e-12, "d log10/dx[{i}]");
    }
}
