use anyhow::Result;
use autograd::loss;
use dtype::float::Float32;
use tensor::{CpuBackend, DenseStorage, Tensor};

#[test]
fn test_mse_loss_forward() -> Result<()> {
    let backend = CpuBackend::<Float32>::new();

    // Pred: [1.0, 2.0, 3.0]
    let pred =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
            backend.clone(),
        )?;

    // Target: [1.5, 2.5, 3.5]
    let target =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
            vec![Float32::new(1.5), Float32::new(2.5), Float32::new(3.5)],
            &[3],
            backend.clone(),
        )?;

    // Diff: [-0.5, -0.5, -0.5]
    // Squared: [0.25, 0.25, 0.25]
    // Sum: 0.75
    // Mean: 0.75 / 3 = 0.25

    let loss = loss::mse_loss(&pred, &target)?;

    let loss_val = loss.as_slice()[0];
    assert!((loss_val.0 - 0.25).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_mse_loss_backward() -> Result<()> {
    let backend = CpuBackend::<Float32>::new();

    // Pred: [1.0, 2.0]
    let pred =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
            backend.clone(),
        )?
        .requires_grad_(true);

    // Target: [2.0, 3.0]
    let target =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
            vec![Float32::new(2.0), Float32::new(3.0)],
            &[2],
            backend.clone(),
        )?;

    // MSE = 1/2 * ((1-2)^2 + (2-3)^2) = 1/2 * (1 + 1) = 1.0

    let loss = loss::mse_loss(&pred, &target)?;

    autograd::backward(&loss, None, false, false)?;

    // d(MSE)/d(pred) = 2/N * (pred - target)
    // N = 2
    // d(MSE)/d(pred) = (pred - target)
    // pred - target = [-1.0, -1.0]

    let grad = pred.grad()?;
    let grad_data = grad.as_slice();

    assert!((grad_data[0].0 - (-1.0)).abs() < 1e-6);
    assert!((grad_data[1].0 - (-1.0)).abs() < 1e-6);
    Ok(())
}

#[test]
fn test_cross_entropy_loss_forward() -> Result<()> {
    let backend = CpuBackend::<Float32>::new();

    // Logits: [[1.0, 2.0]]
    let logits =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[1, 2],
            backend.clone(),
        )?;

    // Targets: [1.0] (Class index 1)
    let targets =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
            vec![Float32::new(1.0)],
            &[1],
            backend.clone(),
        )?;

    // Softmax(logits) = [e^1/(e^1+e^2), e^2/(e^1+e^2)]
    // = [0.2689, 0.7311]
    // LogSoftmax = [-1.3133, -0.3133]
    // Loss = -log_softmax[1] = 0.3133

    let loss = loss::cross_entropy_loss(&logits, &targets)?;

    let loss_val = loss.as_slice()[0];
    assert!((loss_val.0 - 0.3132617).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_cross_entropy_loss_backward() -> Result<()> {
    let backend = CpuBackend::<Float32>::new();

    // Logits: [[1.0, 2.0]]
    let logits =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[1, 2],
            backend.clone(),
        )?
        .requires_grad_(true);

    // Targets: [1.0] (Class 1)
    let targets =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
            vec![Float32::new(1.0)],
            &[1],
            backend.clone(),
        )?;

    let loss = loss::cross_entropy_loss(&logits, &targets)?;

    autograd::backward(&loss, None, false, false)?;

    // Gradient check
    // dL/dx_i = (softmax(x)_i - 1{i=target}) / N
    // N=1
    // softmax = [0.2689, 0.7311]
    // target = 1
    // grad[0] = 0.2689 - 0 = 0.2689
    // grad[1] = 0.7311 - 1 = -0.2689

    let grad = logits.grad()?;
    let grad_data = grad.as_slice();

    assert!((grad_data[0].0 - 0.26894).abs() < 1e-4);
    assert!((grad_data[1].0 - (-0.26894)).abs() < 1e-4);
    Ok(())
}
