use coeus_autograd::{
    cumsum, log_sum_exp, max_axis, mean_axis, min_axis, sum_axis, Var,
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
    assert!((y_slice[0] - 3.0).abs() < 1e-10, "max row 0: {}", y_slice[0]);
    assert!((y_slice[1] - 5.0).abs() < 1e-10, "max row 1: {}", y_slice[1]);
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
    assert!((lse_val - expected_lse).abs() < 1e-9, "lse value: {} vs {}", lse_val, expected_lse);

    let seed = Tensor::from_slice_on(vec![1], &[1.0f64], &backend);
    lse.backward_with_seed(seed);
    let gx = x.grad().unwrap();
    let gx_s = gx.as_slice();
    let sum_exp = e1 + e2 + e3;
    let sm0 = e1 / sum_exp;
    let sm1 = e2 / sum_exp;
    let sm2 = e3 / sum_exp;
    assert!((gx_s[0] - sm0).abs() < 1e-9, "lse grad[0]: {} vs {}", gx_s[0], sm0);
    assert!((gx_s[1] - sm1).abs() < 1e-9, "lse grad[1]: {} vs {}", gx_s[1], sm1);
    assert!((gx_s[2] - sm2).abs() < 1e-9, "lse grad[2]: {} vs {}", gx_s[2], sm2);
    let grad_sum = gx_s[0] + gx_s[1] + gx_s[2];
    assert!((grad_sum - 1.0).abs() < 1e-9, "softmax sums to {}", grad_sum);
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
    assert!((gx_s[0] - 10.0).abs() < 1e-10, "cumsum grad[0]: {}", gx_s[0]);
    assert!((gx_s[1] - 9.0).abs() < 1e-10, "cumsum grad[1]: {}", gx_s[1]);
    assert!((gx_s[2] - 7.0).abs() < 1e-10, "cumsum grad[2]: {}", gx_s[2]);
    assert!((gx_s[3] - 4.0).abs() < 1e-10, "cumsum grad[3]: {}", gx_s[3]);
}
