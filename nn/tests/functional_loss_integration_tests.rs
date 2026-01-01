use backend::CpuBackend;
use dtype::float::Float32;
use nn::functional_loss;
use storage::DenseStorage;
use tensor::Tensor;

#[test]
fn test_functional_mse_loss_autograd_integration() {
    let backend = CpuBackend::<Float32>::new();

    // Pred: [1.0, 2.0]
    let pred =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
            backend.clone(),
        )
        .unwrap()
        .requires_grad_(true);

    // Target: [2.0, 3.0]
    let target =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
            vec![Float32::new(2.0), Float32::new(3.0)],
            &[2],
            backend.clone(),
        )
        .unwrap();

    // MSE = 1.0

    let loss = functional_loss::mse_loss(&pred, &target).unwrap();

    // Verify loss value
    let loss_val = loss.as_slice()[0];
    assert!((loss_val.0 - 1.0).abs() < 1e-6);

    // Verify gradient propagation (requires autograd feature)
    #[cfg(feature = "autograd")]
    {
        // This should compile and run if autograd is enabled for tests
        assert!(loss.requires_grad());

        autograd::backward(&loss, None, false, false).unwrap();

        let grad = pred.grad().unwrap();
        let grad_data = grad.as_slice();

        // d(MSE)/d(pred) = pred - target = [-1.0, -1.0]
        assert!((grad_data[0].0 - (-1.0)).abs() < 1e-6);
        assert!((grad_data[1].0 - (-1.0)).abs() < 1e-6);
    }
}

#[test]
fn test_functional_cross_entropy_loss_autograd_integration() {
    let backend = CpuBackend::<Float32>::new();

    // Logits: [[0.0, 0.0]]
    let logits =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
            vec![Float32::new(0.0), Float32::new(0.0)],
            &[1, 2],
            backend.clone(),
        )
        .unwrap()
        .requires_grad_(true);

    // Targets: [[1.0, 0.0]] (Class 0)
    let targets =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
            vec![Float32::new(1.0), Float32::new(0.0)],
            &[1, 2],
            backend.clone(),
        )
        .unwrap();

    let loss = functional_loss::cross_entropy(&logits, &targets).unwrap();

    // Loss = 0.6931
    let loss_val = loss.as_slice()[0];
    assert!((loss_val.0 - std::f32::consts::LN_2).abs() < 1e-4);

    #[cfg(feature = "autograd")]
    {
        assert!(loss.requires_grad());

        autograd::backward(&loss, None, false, false).unwrap();

        let grad = logits.grad().unwrap();
        let grad_data = grad.as_slice();

        // Grad = [-0.5, 0.5]
        assert!((grad_data[0].0 - (-0.5)).abs() < 1e-6);
        assert!((grad_data[1].0 - 0.5).abs() < 1e-6);
    }
}
