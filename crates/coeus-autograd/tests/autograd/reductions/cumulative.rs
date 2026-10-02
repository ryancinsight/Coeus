//! Cumulative and product reductions: `cumsum`, `cumprod`, `prod`.

use super::*;

#[test]
fn test_cumsum_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![4], &[1.0f64, 2.0, 3.0, 4.0], &backend);
    let x = Var::new(x_val, true);

    let y = cumsum(&x, 0);
    let y_s = y.tensor.as_slice();
    assert!((y_s[0] - 1.0).abs() < 1e-10);
    assert!((y_s[1] - 3.0).abs() < 1e-10);
    assert!((y_s[2] - 6.0).abs() < 1e-10);
    assert!((y_s[3] - 10.0).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![4], &[1.0f64, 2.0, 3.0, 4.0], &backend);
    y.backward_with_seed(seed)
        .expect("invariant: valid autograd fixture completes backward");
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!(
        (gx_s[0] - 10.0).abs() < 1e-10,
        "cumsum grad[0]: {}",
        gx_s[0]
    );
    assert!((gx_s[1] - 9.0).abs() < 1e-10, "cumsum grad[1]: {}", gx_s[1]);
    assert!((gx_s[2] - 7.0).abs() < 1e-10, "cumsum grad[2]: {}", gx_s[2]);
    assert!((gx_s[3] - 4.0).abs() < 1e-10, "cumsum grad[3]: {}", gx_s[3]);
}

#[test]
fn test_var_autograd_matches_analytic() {
    let backend = MoiraiBackend::new();
    // x = [1,2,3,4]: mean = 2.5; unbiased var = ((1.5)^2+(0.5)^2+(0.5)^2+(1.5)^2)/3
    // = 5/3; biased divides by 4 -> 5/4.
    let data = [1.0f64, 2.0, 3.0, 4.0];
    let x = Var::new(Tensor::from_slice_on(vec![4], &data, &backend), true);

    let (v, mu) = var_mean(&x, true);
    assert!((mu.tensor.as_slice()[0] - 2.5).abs() < 1e-14, "mean");
    assert!(
        (v.tensor.as_slice()[0] - 5.0 / 3.0).abs() < 1e-14,
        "unbiased var: {}",
        v.tensor.as_slice()[0]
    );
    let vb = var(&x, false);
    assert!((vb.tensor.as_slice()[0] - 1.25).abs() < 1e-14, "biased var");
    let s = std_dev(&x, true);
    assert!(
        (s.tensor.as_slice()[0] - (5.0f64 / 3.0).sqrt()).abs() < 1e-14,
        "std"
    );

    // Analytic gradient of the unbiased variance: dv/dx_i = 2(x_i - mean)/(n-1)
    // (the mean-path terms cancel because sum(x_j - mean) = 0).
    v.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let gx = x.grad().unwrap();
    let gx = gx.as_slice();
    for (i, &xi) in data.iter().enumerate() {
        let expected = 2.0 * (xi - 2.5) / 3.0;
        assert!(
            (gx[i] - expected).abs() < 1e-14,
            "d var/dx[{i}]: {} vs {expected}",
            gx[i]
        );
    }

    // Numerical-gradient cross-check (central differences, h = 1e-6): an
    // implementation-independent oracle for the composed backward.
    let h = 1e-6f64;
    for i in 0..data.len() {
        let mut dp = data;
        dp[i] += h;
        let mut dm = data;
        dm[i] -= h;
        let vp = var(
            &Var::new(Tensor::from_slice_on(vec![4], &dp, &backend), false),
            true,
        );
        let vm = var(
            &Var::new(Tensor::from_slice_on(vec![4], &dm, &backend), false),
            true,
        );
        let numeric = (vp.tensor.as_slice()[0] - vm.tensor.as_slice()[0]) / (2.0 * h);
        assert!(
            (numeric - gx[i]).abs() < 1e-8,
            "numeric {numeric} vs autograd {} at {i}",
            gx[i]
        );
    }
}

#[test]
fn test_var_axis_autograd_matches_analytic() {
    let backend = MoiraiBackend::new();
    // [[1,2,3],[4,6,8]] axis=1: means [2,6]; unbiased vars [1, 4].
    let data = [1.0f64, 2.0, 3.0, 4.0, 6.0, 8.0];
    let x = Var::new(Tensor::from_slice_on(vec![2, 3], &data, &backend), true);

    let (v, mu) = var_mean_axis(&x, 1, true);
    assert_eq!(v.tensor.shape(), &[2, 1], "keepdim shape");
    let vs = v.tensor.as_slice().to_vec();
    let ms = mu.tensor.as_slice().to_vec();
    assert!(
        (ms[0] - 2.0).abs() < 1e-14 && (ms[1] - 6.0).abs() < 1e-14,
        "means"
    );
    assert!((vs[0] - 1.0).abs() < 1e-14, "row0 var: {}", vs[0]);
    assert!((vs[1] - 4.0).abs() < 1e-14, "row1 var: {}", vs[1]);

    // Row-local gradient: dv_r/dx_ri = 2(x_ri - mean_r)/(extent-1), extent-1 = 2.
    v.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let gx = x.grad().unwrap();
    let gx = gx.as_slice();
    let means = [2.0, 2.0, 2.0, 6.0, 6.0, 6.0];
    for i in 0..6 {
        let expected = 2.0 * (data[i] - means[i]) / 2.0;
        assert!(
            (gx[i] - expected).abs() < 1e-14,
            "d var/dx[{i}]: {} vs {expected}",
            gx[i]
        );
    }
}

#[test]
fn test_prod_autograd_matches_analytic() {
    let backend = MoiraiBackend::new();
    // prod([1,2,3,4]) = 24; d prod/dx_i = prod_{j != i} x_j = [24, 12, 8, 6].
    let x = Var::new(
        Tensor::from_slice_on(vec![4], &[1.0f64, 2.0, 3.0, 4.0], &backend),
        true,
    );
    let y = coeus_autograd::prod(&x);
    assert!((y.tensor.as_slice()[0] - 24.0).abs() < 1e-14, "fwd");
    y.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let g = x.grad().unwrap();
    let g = g.as_slice();
    for (i, want) in [24.0, 12.0, 8.0, 6.0].iter().enumerate() {
        assert!((g[i] - want).abs() < 1e-14, "dx[{i}]: {} vs {want}", g[i]);
    }

    // Adversarial zero: prod([2,0,3]) = 0; only the zero position has a
    // non-zero gradient (d/dx_1 = 2*3 = 6) — exact, not epsilon-fudged.
    let z = Var::new(
        Tensor::from_slice_on(vec![3], &[2.0f64, 0.0, 3.0], &backend),
        true,
    );
    let yz = coeus_autograd::prod(&z);
    assert_eq!(yz.tensor.as_slice()[0], 0.0, "fwd zero");
    yz.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let gz = z.grad().unwrap();
    let gz = gz.as_slice();
    for (i, want) in [0.0, 6.0, 0.0].iter().enumerate() {
        assert!(
            (gz[i] - want).abs() < 1e-14,
            "zero dx[{i}]: {} vs {want}",
            gz[i]
        );
    }

    // A non-unit seed scales the exact one-zero derivative.
    let seeded = Var::new(
        Tensor::from_slice_on(vec![3], &[2.0f64, 0.0, 3.0], &backend),
        true,
    );
    let seeded_product = coeus_autograd::prod(&seeded);
    seeded_product
        .backward_with_seed(Tensor::from_slice_on(vec![1], &[7.0], &backend))
        .expect("invariant: seeded product backward completes");
    let seeded_grad = seeded.grad().unwrap();
    for (i, want) in [0.0, 42.0, 0.0].iter().enumerate() {
        assert!(
            (seeded_grad.as_slice()[i] - want).abs() < 1e-14,
            "seeded zero dx[{i}]: {} vs {want}",
            seeded_grad.as_slice()[i]
        );
    }

    // Two zeros annihilate every partial product, so every derivative is zero.
    let multiple = Var::new(
        Tensor::from_slice_on(vec![5], &[2.0f64, 0.0, 3.0, 0.0, 5.0], &backend),
        true,
    );
    let multiple_product = coeus_autograd::prod(&multiple);
    multiple_product
        .backward()
        .expect("invariant: multi-zero product backward completes");
    assert_eq!(
        multiple.grad().unwrap().as_slice(),
        &[0.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn test_cumprod_backward_exact_at_zeros() {
    let backend = MoiraiBackend::new();

    // Zero-free regression: x = [1,2,3], out = [1,2,6], ones seed.
    // grad_i = Σ_{j≥i} ∏_{k≤j,k≠i} x_k: [1+2+6, 1+3, 2] = [9, 4, 2].
    let x = Var::new(
        Tensor::from_slice_on(vec![3], &[1.0f64, 2.0, 3.0], &backend),
        true,
    );
    let y = cumsum_free_cumprod(&x);
    y.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let g = x.grad().unwrap();
    for (i, want) in [9.0, 4.0, 2.0].iter().enumerate() {
        assert!(
            (g.as_slice()[i] - want).abs() < 1e-14,
            "zero-free dx[{i}]: {} vs {want}",
            g.as_slice()[i]
        );
    }

    // Single zero: x = [2,0,3], out = [2,0,0], ones seed.
    // dx0 = d out0/dx0 = 1 (later outs carry x1 = 0);
    // dx1 = x0 + x0·x2 = 2 + 6 = 8; dx2 = x0·x1 = 0.
    let x = Var::new(
        Tensor::from_slice_on(vec![3], &[2.0f64, 0.0, 3.0], &backend),
        true,
    );
    let y = cumsum_free_cumprod(&x);
    y.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let g = x.grad().unwrap();
    for (i, want) in [1.0, 8.0, 0.0].iter().enumerate() {
        assert!(
            (g.as_slice()[i] - want).abs() < 1e-14,
            "one-zero dx[{i}]: {} vs {want}",
            g.as_slice()[i]
        );
    }

    // Two zeros: x = [2,0,3,0,5] — the second zero kills every gradient at
    // and after it; the first zero's gradient sums only up to the second:
    // dx = [1, 2 + 2·3, 0, 0, 0] = [1, 8, 0, 0, 0].
    let x = Var::new(
        Tensor::from_slice_on(vec![5], &[2.0f64, 0.0, 3.0, 0.0, 5.0], &backend),
        true,
    );
    let y = cumsum_free_cumprod(&x);
    y.backward()
        .expect("invariant: valid autograd fixture completes backward");
    let g = x.grad().unwrap();
    for (i, want) in [1.0, 8.0, 0.0, 0.0, 0.0].iter().enumerate() {
        assert!(
            (g.as_slice()[i] - want).abs() < 1e-14,
            "two-zero dx[{i}]: {} vs {want}",
            g.as_slice()[i]
        );
    }
}

/// Sum the cumprod so backward seeds ones across all cumprod outputs.
fn cumsum_free_cumprod(x: &Var<f64, MoiraiBackend>) -> Var<f64, MoiraiBackend> {
    coeus_autograd::sum(&coeus_autograd::cumprod(x, 0))
}
