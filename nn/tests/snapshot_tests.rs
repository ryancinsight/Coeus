//! Snapshot tests for neural network operations.
//!
//! This module uses the `insta` crate to create snapshot tests that verify
//! neural network operations produce consistent, expected results.

use insta::assert_snapshot;
use serde::Serialize;

use backend::CpuBackend;
use dtype::float::Float32;
use nn::functional_api as functional;
use nn::functional_api as functional_loss;
use storage::DenseStorage;
use tensor::Tensor;

/// Helper to create a serializable snapshot of tensor data
#[derive(Serialize)]
struct TensorSnapshot {
    shape: Vec<usize>,
    data: Vec<f64>, // Use f64 for better precision in snapshots
}

impl TensorSnapshot {
    fn from_tensor(tensor: &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>) -> Self {
        Self {
            shape: tensor.shape().dims().to_vec(),
            data: tensor.as_slice().iter().map(|&x| x.get() as f64).collect(),
        }
    }
}

/// Test ReLU activation snapshot
#[test]
fn test_relu_snapshot() {
    let data = vec![
        Float32::new(-2.0),
        Float32::new(-1.0),
        Float32::new(0.0),
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
    ];
    let tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data, &[6])
            .unwrap();

    let result = functional::relu(&tensor).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "relu_basic",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test sigmoid activation snapshot
#[test]
fn test_sigmoid_snapshot() {
    let data = vec![
        Float32::new(-2.0),
        Float32::new(-1.0),
        Float32::new(0.0),
        Float32::new(1.0),
        Float32::new(2.0),
    ];
    let tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data, &[5])
            .unwrap();

    let result = functional::sigmoid(&tensor).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "sigmoid_basic",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test tanh activation snapshot
#[test]
fn test_tanh_snapshot() {
    let data = vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(1.0)];
    let tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data, &[3])
            .unwrap();

    let result = functional::tanh(&tensor).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "tanh_basic",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test GELU activation snapshot
#[test]
fn test_gelu_snapshot() {
    let data = vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(1.0)];
    let tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data, &[3])
            .unwrap();

    let result = functional::gelu(&tensor).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "gelu_basic",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test max pooling snapshot
#[test]
fn test_max_pool2d_snapshot() {
    // Create a simple 4x4 input with known values
    let data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
        Float32::new(5.0),
        Float32::new(6.0),
        Float32::new(7.0),
        Float32::new(8.0),
        Float32::new(9.0),
        Float32::new(10.0),
        Float32::new(11.0),
        Float32::new(12.0),
        Float32::new(13.0),
        Float32::new(14.0),
        Float32::new(15.0),
        Float32::new(16.0),
    ];
    let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        data,
        &[1, 1, 4, 4],
    )
    .unwrap();

    let result = functional::max_pool2d(&tensor, (2, 2), Some((2, 2)), (0, 0)).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "max_pool2d_basic",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test average pooling snapshot
#[test]
fn test_avg_pool2d_snapshot() {
    let data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
        Float32::new(4.0),
        Float32::new(3.0),
        Float32::new(2.0),
        Float32::new(1.0),
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
        Float32::new(4.0),
        Float32::new(3.0),
        Float32::new(2.0),
        Float32::new(1.0),
    ];
    let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        data,
        &[1, 1, 4, 4],
    )
    .unwrap();

    let result = functional::avg_pool2d(&tensor, (2, 2), Some((2, 2)), (0, 0)).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "avg_pool2d_basic",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test linear transformation snapshot
#[test]
fn test_linear_snapshot() {
    let input_data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 3],
    )
    .unwrap();

    let weight_data = vec![
        Float32::new(1.0),
        Float32::new(0.0),
        Float32::new(0.0),
        Float32::new(0.0),
        Float32::new(1.0),
        Float32::new(0.0),
    ];
    let weight = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        weight_data,
        &[2, 3],
    )
    .unwrap();

    let result = functional::linear(&input, &weight, None).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "linear_basic",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test linear with bias snapshot
#[test]
fn test_linear_with_bias_snapshot() {
    let input_data = vec![Float32::new(1.0), Float32::new(2.0)];
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 2],
    )
    .unwrap();

    let weight_data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
    ];
    let weight = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        weight_data,
        &[2, 2],
    )
    .unwrap();

    let bias_data = vec![Float32::new(0.5), Float32::new(1.5)];
    let bias =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(bias_data, &[2])
            .unwrap();

    let result = functional::linear(&input, &weight, Some(&bias)).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "linear_with_bias",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test convolution snapshot
#[test]
fn test_conv2d_snapshot() {
    let input_data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
        Float32::new(5.0),
        Float32::new(6.0),
        Float32::new(7.0),
        Float32::new(8.0),
        Float32::new(9.0),
    ];
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 1, 3, 3],
    )
    .unwrap();

    let weight_data = vec![
        Float32::new(1.0),
        Float32::new(0.0),
        Float32::new(0.0),
        Float32::new(1.0),
    ];
    let weight = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        weight_data,
        &[1, 1, 2, 2],
    )
    .unwrap();

    let result = functional::conv2d(&input, &weight, None, Some((1, 1)), Some((0, 0))).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "conv2d_basic",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test softmax snapshot
#[test]
fn test_softmax_snapshot() {
    let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
    let tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data, &[3])
            .unwrap();

    let result = functional::softmax(&tensor).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "softmax_basic",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test MSE loss snapshot
#[test]
fn test_mse_loss_snapshot() {
    let pred_data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
    let pred =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(pred_data, &[3])
            .unwrap();

    let target_data = vec![Float32::new(1.5), Float32::new(2.5), Float32::new(3.5)];
    let target =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(target_data, &[3])
            .unwrap();

    let result = functional_loss::mse_loss(&pred, &target).unwrap();

    #[derive(Serialize)]
    struct ScalarSnapshot {
        value: f64,
    }

    let snapshot = ScalarSnapshot {
        value: result.as_slice()[0].get() as f64,
    };
    assert_snapshot!(
        "mse_loss_basic",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test layer normalization snapshot
#[test]
fn test_layer_norm_snapshot() {
    let data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
        Float32::new(5.0),
        Float32::new(6.0),
    ];
    let tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data, &[2, 3])
            .unwrap();

    let result = functional::layer_norm(&tensor, &[3], None, None, 1e-5).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "layer_norm_basic",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test layer normalization with affine parameters
#[test]
fn test_layer_norm_with_affine_snapshot() {
    let data = vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)];
    let tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data, &[1, 3])
            .unwrap();

    let weight_data = vec![Float32::new(2.0), Float32::new(1.0), Float32::new(0.5)];
    let weight =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(weight_data, &[3])
            .unwrap();

    let bias_data = vec![Float32::new(0.1), Float32::new(0.2), Float32::new(0.3)];
    let bias =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(bias_data, &[3])
            .unwrap();

    let result = functional::layer_norm(&tensor, &[3], Some(&weight), Some(&bias), 1e-5).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "layer_norm_affine",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test SiLU activation snapshot
#[test]
fn test_silu_snapshot() {
    let data = vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(1.0)];
    let tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data, &[3])
            .unwrap();

    let result = functional::silu(&tensor).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "silu_basic",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test Leaky ReLU activation snapshot
#[test]
fn test_leaky_relu_snapshot() {
    let data = vec![
        Float32::new(-2.0),
        Float32::new(-1.0),
        Float32::new(0.0),
        Float32::new(1.0),
    ];
    let tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data, &[4])
            .unwrap();

    let result = functional::leaky_relu(&tensor, Float32::new(0.1)).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "leaky_relu_basic",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}

/// Test ELU activation snapshot
#[test]
fn test_elu_snapshot() {
    let data = vec![
        Float32::new(-2.0),
        Float32::new(-1.0),
        Float32::new(0.0),
        Float32::new(1.0),
    ];
    let tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data, &[4])
            .unwrap();

    let result = functional::elu(&tensor, Float32::new(1.0)).unwrap();
    let snapshot = TensorSnapshot::from_tensor(&result);

    assert_snapshot!(
        "elu_basic",
        serde_json::to_string_pretty(&snapshot).unwrap()
    );
}
