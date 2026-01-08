use anyhow::Result;
use autograd::tensor_ops;
use dtype::float::Float32;
use tensor::{CpuBackend, DenseStorage, Tensor};

#[test]
fn test_autograd_simple_mul_mean() -> Result<()> {
    // a = [2.0, 3.0]
    let a = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0), Float32::new(3.0)],
        &[2],
    )?
    .requires_grad_(true);

    // b = [4.0, 5.0]
    let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(4.0), Float32::new(5.0)],
        &[2],
    )?
    .requires_grad_(true);

    // c = a * b = [8.0, 15.0]
    let c = tensor_ops::mul(&a, &b)?;

    // d = mean(c) = (8 + 15) / 2 = 11.5
    let d = tensor_ops::mean(&c, None, false)?;

    // backward
    autograd::backward(&d, None, false, false)?;

    // Check gradients
    // d(mean)/dc = 0.5
    // dc/da = b
    // d(mean)/da = d(mean)/dc * dc/da = 0.5 * b
    // grad_a = [0.5 * 4, 0.5 * 5] = [2.0, 2.5]

    let grad_a = a.grad()?;
    let grad_a_data = grad_a.as_slice();
    assert!((grad_a_data[0].0 - 2.0).abs() < 1e-6);
    assert!((grad_a_data[1].0 - 2.5).abs() < 1e-6);

    // grad_b = 0.5 * a = [1.0, 1.5]
    let grad_b = b.grad()?;
    let grad_b_data = grad_b.as_slice();
    assert!((grad_b_data[0].0 - 1.0).abs() < 1e-6);
    assert!((grad_b_data[1].0 - 1.5).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_autograd_reuse_variable() -> Result<()> {
    // x = [2.0]
    let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0)],
        &[1],
    )?
    .requires_grad_(true);

    // y = x * x + x
    let x2 = tensor_ops::mul(&x, &x)?;
    let y = tensor_ops::add(&x2, &x)?;

    // backward
    autograd::backward(&y, None, false, false)?;

    // dy/dx = 2x + 1 = 2(2) + 1 = 5
    let grad_x = x.grad()?;
    let grad_x_data = grad_x.as_slice();
    assert!((grad_x_data[0].0 - 5.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_operator_overload_broadcast_add_scalar_vector_grad() -> Result<()> {
    let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0)],
        &[1],
    )?
    .requires_grad_(true);

    let y = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[3],
    )?
    .requires_grad_(true);

    let z = &x + &y;
    autograd::backward(&z, None, false, false)?;

    let grad_x = x.grad()?;
    let grad_y = y.grad()?;

    assert!((grad_x.as_slice()[0].0 - 3.0).abs() < 1e-6);
    for g in grad_y.as_slice() {
        assert!((g.0 - 1.0).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn test_hvp() -> Result<()> {
    // f(x) = sum(x^2)
    // grad(f) = 2x
    // Hessian(f) = 2I
    // For x = [1, 2], v = [1, 1], Hessian * v = [2, 2]

    let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0)],
        &[2],
    )?
    .requires_grad_(true);

    let v = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(1.0)],
        &[2],
    )?;

    let func = |inputs: &[Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>]| {
        let x = &inputs[0];
        let x2 = tensor_ops::mul(x, x)?;
        let res = tensor_ops::sum(&x2, None, false)?;
        Ok(res)
    };

    let hvp_res = autograd::hvp(func, &[x], &[v])?;
    let hvp_data = hvp_res[0].as_slice();

    assert!((hvp_data[0].0 - 2.0).abs() < 1e-6);
    assert!((hvp_data[1].0 - 2.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_jvp() -> Result<()> {
    // f(x) = x^2 (element-wise)
    // Jacobian(f) = diag(2x)
    // For x = [1, 2], v = [1, 1], Jacobian * v = [2*1*1, 2*2*1] = [2, 4]

    let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0)],
        &[2],
    )?
    .requires_grad_(true);

    let v = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(1.0)],
        &[2],
    )?;

    let func = |inputs: &[Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>]| {
        let x = &inputs[0];
        let x2 = tensor_ops::mul(x, x)?;
        Ok(x2)
    };

    let jvp_res = autograd::jvp(func, &[x], &[v])?;
    let jvp_data = jvp_res.as_slice();

    assert!((jvp_data[0].0 - 2.0).abs() < 1e-6);
    assert!((jvp_data[1].0 - 4.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_operator_overload_broadcast_mul_scalar_vector_grad() -> Result<()> {
    let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(2.0)],
        &[1],
    )?
    .requires_grad_(true);

    let y = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
        &[3],
    )?
    .requires_grad_(true);

    let z = &x * &y;
    autograd::backward(&z, None, false, false)?;

    let grad_x = x.grad()?;
    let grad_y = y.grad()?;

    assert!((grad_x.as_slice()[0].0 - 6.0).abs() < 1e-6);
    for g in grad_y.as_slice() {
        assert!((g.0 - 2.0).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn test_operator_overload_broadcast_add_leading_dims_grad() -> Result<()> {
    let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ],
        &[2, 1, 3],
    )?
    .requires_grad_(true);

    let y = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(10.0), Float32::new(20.0), Float32::new(30.0)],
        &[3],
    )?
    .requires_grad_(true);

    let z = &x + &y;
    autograd::backward(&z, None, false, false)?;

    let grad_x = x.grad()?;
    let grad_y = y.grad()?;

    for g in grad_x.as_slice() {
        assert!((g.0 - 1.0).abs() < 1e-6);
    }

    for g in grad_y.as_slice() {
        assert!((g.0 - 2.0).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn test_tensor_ops_broadcast_add_matrix_vector_grad() -> Result<()> {
    let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ],
        &[2, 3],
    )?
    .requires_grad_(true);

    let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(10.0), Float32::new(20.0), Float32::new(30.0)],
        &[3],
    )?
    .requires_grad_(true);

    let y = tensor_ops::add(&x, &b)?;
    autograd::backward(&y, None, false, false)?;

    let grad_x = x.grad()?;
    let grad_b = b.grad()?;

    for g in grad_x.as_slice() {
        assert!((g.0 - 1.0).abs() < 1e-6);
    }

    for g in grad_b.as_slice() {
        assert!((g.0 - 2.0).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn test_tensor_ops_broadcast_mul_matrix_vector_grad() -> Result<()> {
    let x = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
        ],
        &[2, 3],
    )?
    .requires_grad_(true);

    let b = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(10.0), Float32::new(20.0), Float32::new(30.0)],
        &[3],
    )?
    .requires_grad_(true);

    let y = tensor_ops::mul(&x, &b)?;
    autograd::backward(&y, None, false, false)?;

    let grad_x = x.grad()?;
    let grad_b = b.grad()?;

    let expected_grad_x = [10.0, 20.0, 30.0, 10.0, 20.0, 30.0];
    for (i, (g, expected)) in grad_x
        .as_slice()
        .iter()
        .zip(expected_grad_x.iter())
        .enumerate()
    {
        assert!(
            (g.0 - expected).abs() < 1e-6,
            "grad_x[{i}] = {}, expected {expected}",
            g.0
        );
    }

    let expected_grad_b = [5.0, 7.0, 9.0];
    for (i, (g, expected)) in grad_b
        .as_slice()
        .iter()
        .zip(expected_grad_b.iter())
        .enumerate()
    {
        assert!(
            (g.0 - expected).abs() < 1e-6,
            "grad_b[{i}] = {}, expected {expected}",
            g.0
        );
    }
    Ok(())
}
