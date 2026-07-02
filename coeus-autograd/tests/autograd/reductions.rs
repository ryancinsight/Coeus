use coeus_autograd::{
    cumsum, log_sum_exp, max_axis, mean_axis, min_axis, norm_p, norm_p_axis, std_dev, sum_axis,
    var, var_mean, var_mean_axis, Var,
};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn test_sum_axis_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(
        vec![2, 3],
        &[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32],
        &backend,
    );
    let x = Var::new(x_val, true);

    let y = sum_axis(&x, 1);
    assert_eq!(y.tensor.shape(), &[2, 1]);
    let y_slice = y.tensor.as_slice();
    assert_eq!(y_slice[0], 6.0);
    assert_eq!(y_slice[1], 15.0);

    let grad_out = Tensor::from_slice_on(vec![2, 1], &[2.0f32, 3.0f32], &backend);
    y.backward_with_seed(grad_out);

    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    assert_eq!(gx_slice, &[2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
}

#[test]
fn test_mean_axis_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(
        vec![2, 3],
        &[1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32],
        &backend,
    );
    let x = Var::new(x_val, true);

    let y = mean_axis(&x, 1);
    assert_eq!(y.tensor.shape(), &[2, 1]);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 2.0).abs() < 1e-5);
    assert!((y_slice[1] - 5.0).abs() < 1e-5);

    let grad_out = Tensor::from_slice_on(vec![2, 1], &[3.0f32, 6.0f32], &backend);
    y.backward_with_seed(grad_out);

    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    assert_eq!(gx_slice, &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
}

#[test]
fn test_max_axis_autograd() {
    // x = [[1, 3, 2], [4, 1, 5]]  max along axis=1 → [[3], [5]]
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![2, 3], &[1.0f64, 3.0, 2.0, 4.0, 1.0, 5.0], &backend);
    let x = Var::new(x_val, true);

    let y = max_axis(&x, 1);
    let y_slice = y.tensor.as_slice();
    assert!(
        (y_slice[0] - 3.0).abs() < 1e-10,
        "max row 0: {}",
        y_slice[0]
    );
    assert!(
        (y_slice[1] - 5.0).abs() < 1e-10,
        "max row 1: {}",
        y_slice[1]
    );
    assert_eq!(y.tensor.shape(), &[2, 1]);

    let seed = Tensor::from_slice_on(vec![2, 1], &[1.0f64, 1.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!((gx_s[0] - 0.0).abs() < 1e-10, "[0,0]: {}", gx_s[0]);
    assert!((gx_s[1] - 1.0).abs() < 1e-10, "[0,1]: {}", gx_s[1]);
    assert!((gx_s[2] - 0.0).abs() < 1e-10, "[0,2]: {}", gx_s[2]);
    assert!((gx_s[3] - 0.0).abs() < 1e-10, "[1,0]: {}", gx_s[3]);
    assert!((gx_s[4] - 0.0).abs() < 1e-10, "[1,1]: {}", gx_s[4]);
    assert!((gx_s[5] - 1.0).abs() < 1e-10, "[1,2]: {}", gx_s[5]);
}

#[test]
fn test_max_axis_tie_normalisation() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![3], &[2.0f64, 2.0, 1.0], &backend);
    let x = Var::new(x_val, true);
    let y = max_axis(&x, 0);
    let seed = Tensor::from_slice_on(vec![1], &[1.0f64], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!((gx_s[0] - 0.5).abs() < 1e-8, "tie pos 0: {}", gx_s[0]);
    assert!((gx_s[1] - 0.5).abs() < 1e-8, "tie pos 1: {}", gx_s[1]);
    assert!((gx_s[2] - 0.0).abs() < 1e-10, "non-max: {}", gx_s[2]);
}

#[test]
fn test_min_axis_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![2, 3], &[1.0f64, 3.0, 2.0, 4.0, 1.0, 5.0], &backend);
    let x = Var::new(x_val, true);
    let y = min_axis(&x, 1);
    let y_slice = y.tensor.as_slice();
    assert!((y_slice[0] - 1.0).abs() < 1e-10);
    assert!((y_slice[1] - 1.0).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![2, 1], &[1.0f64, 1.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    assert!((gx_s[0] - 1.0).abs() < 1e-10, "[0,0]: {}", gx_s[0]);
    assert!((gx_s[1] - 0.0).abs() < 1e-10, "[0,1]: {}", gx_s[1]);
    assert!((gx_s[2] - 0.0).abs() < 1e-10, "[0,2]: {}", gx_s[2]);
    assert!((gx_s[3] - 0.0).abs() < 1e-10, "[1,0]: {}", gx_s[3]);
    assert!((gx_s[4] - 1.0).abs() < 1e-10, "[1,1]: {}", gx_s[4]);
    assert!((gx_s[5] - 0.0).abs() < 1e-10, "[1,2]: {}", gx_s[5]);
}

#[test]
fn test_norm_p_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![2], &[2.0f64, -4.0], &backend);
    let x = Var::new(x_val, true);

    let y = norm_p(&x, 3.0);
    let expected_y = 72.0f64.powf(1.0 / 3.0);
    assert_eq!(y.tensor.shape(), &[1]);
    assert!((y.tensor.as_slice()[0] - expected_y).abs() < 1e-10);

    let seed = Tensor::from_slice_on(vec![1], &[1.5f64], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    let denom = expected_y.powf(2.0);
    let expected = [1.5 * 4.0 / denom, -1.5 * 16.0 / denom];
    for (i, (&actual, &expected)) in gx_s.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-10,
            "norm_p grad[{i}]: {actual} vs {expected}"
        );
    }
}

#[test]
fn test_norm_p_axis_autograd() {
    let backend = MoiraiBackend::new();
    let x_val = Tensor::from_slice_on(vec![2, 2], &[3.0f64, 4.0, 5.0, 12.0], &backend);
    let x = Var::new(x_val, true);

    let y = norm_p_axis(&x, 2.0, 1);
    assert_eq!(y.tensor.shape(), &[2, 1]);
    assert_eq!(y.tensor.as_slice(), &[5.0, 13.0]);

    let seed = Tensor::from_slice_on(vec![2, 1], &[2.0f64, 3.0], &backend);
    y.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    let expected = [6.0 / 5.0, 8.0 / 5.0, 15.0 / 13.0, 36.0 / 13.0];
    for (i, (&actual, &expected)) in gx_s.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-10,
            "norm_p_axis grad[{i}]: {actual} vs {expected}"
        );
    }
}

#[test]
fn test_log_sum_exp_autograd() {
    let backend = MoiraiBackend::new();
    let vals = [1.0f64, 2.0, 3.0];
    let x_val = Tensor::from_slice_on(vec![3], &vals, &backend);
    let x = Var::new(x_val, true);

    let lse = log_sum_exp(&x, 0);
    let lse_val = lse.tensor.as_slice()[0];
    let e1 = 1.0f64.exp();
    let e2 = 2.0f64.exp();
    let e3 = 3.0f64.exp();
    let expected_lse = (e1 + e2 + e3).ln();
    assert!(
        (lse_val - expected_lse).abs() < 1e-9,
        "lse value: {} vs {}",
        lse_val,
        expected_lse
    );

    let seed = Tensor::from_slice_on(vec![1], &[1.0f64], &backend);
    lse.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    let sum_exp = e1 + e2 + e3;
    let sm0 = e1 / sum_exp;
    let sm1 = e2 / sum_exp;
    let sm2 = e3 / sum_exp;
    assert!(
        (gx_s[0] - sm0).abs() < 1e-9,
        "lse grad[0]: {} vs {}",
        gx_s[0],
        sm0
    );
    assert!(
        (gx_s[1] - sm1).abs() < 1e-9,
        "lse grad[1]: {} vs {}",
        gx_s[1],
        sm1
    );
    assert!(
        (gx_s[2] - sm2).abs() < 1e-9,
        "lse grad[2]: {} vs {}",
        gx_s[2],
        sm2
    );
    let grad_sum = gx_s[0] + gx_s[1] + gx_s[2];
    assert!(
        (grad_sum - 1.0).abs() < 1e-9,
        "softmax sums to {}",
        grad_sum
    );
}

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
    y.backward_with_seed(seed);
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
    v.backward();
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
    v.backward();
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
