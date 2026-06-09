use coeus_autograd::{exp, log, Var};
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
    y.backward_with_seed(grad_out);

    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    assert!((gx_slice[0] - 1.0).abs() < 1e-5);
    assert!((gx_slice[1] - 5.4365636).abs() < 1e-5);
    assert!((gx_slice[2] - 22.167168).abs() < 1e-5);
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
    y.backward_with_seed(grad_out);

    let gx = x.grad().unwrap();
    let gx_slice = gx.as_slice();
    assert!((gx_slice[0] - 1.0).abs() < 1e-5);
    assert!((gx_slice[1] - 1.0).abs() < 1e-5);
    assert!((gx_slice[2] - 0.75).abs() < 1e-5);
}
