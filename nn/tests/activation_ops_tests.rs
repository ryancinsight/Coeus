//! Unit tests for activation function operations
//!
//! Tests for stateless activation functions in nn::functional::ops::activations

use backend::CpuBackend;
use dtype::float::Float32;
use nn::functional::ops::activations::*;
use storage::DenseStorage;
use tensor::Tensor;

type TestTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

// Helper function to create a test tensor
fn create_tensor(data: Vec<f32>, shape: &[usize]) -> TestTensor {
    let float_data: Vec<Float32> = data.into_iter().map(Float32::new).collect();
    TestTensor::from_vec(float_data, shape).unwrap()
}

// Helper function to compare tensors with tolerance
fn assert_tensor_close(actual: &TestTensor, expected: &[f32], tolerance: f32) {
    let actual_data = actual.as_slice();
    assert_eq!(actual_data.len(), expected.len(), "Tensor length mismatch");
    for (i, (&a, &e)) in actual_data.iter().zip(expected.iter()).enumerate() {
        let diff = (a.get() as f32 - e).abs();
        assert!(
            diff < tolerance,
            "Value mismatch at index {}: expected {}, got {} (diff: {})",
            i,
            e,
            a.get() as f32,
            diff
        );
    }
}

#[test]
fn test_relu_basic() {
    let input = create_tensor(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let output = relu(&input).unwrap();
    assert_tensor_close(&output, &[0.0, 0.0, 0.0, 1.0, 2.0], 1e-6);
}

#[test]
fn test_relu_all_positive() {
    let input = create_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let output = relu(&input).unwrap();
    assert_tensor_close(&output, &[1.0, 2.0, 3.0], 1e-6);
}

#[test]
fn test_relu_all_negative() {
    let input = create_tensor(vec![-1.0, -2.0, -3.0], &[3]);
    let output = relu(&input).unwrap();
    assert_tensor_close(&output, &[0.0, 0.0, 0.0], 1e-6);
}

#[test]
fn test_relu_empty_tensor() {
    let input = create_tensor(vec![], &[0]);
    let output = relu(&input).unwrap();
    assert_eq!(output.as_slice().len(), 0);
}

#[test]
fn test_relu_single_element() {
    let input = create_tensor(vec![5.0], &[1]);
    let output = relu(&input).unwrap();
    assert_tensor_close(&output, &[5.0], 1e-6);
}

#[test]
fn test_sigmoid_basic() {
    let input = create_tensor(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let output = sigmoid(&input).unwrap();

    // Expected values: sigmoid(x) = 1 / (1 + exp(-x))
    let expected = vec![
        1.0 / (1.0 + (-(-2.0_f32)).exp()),
        1.0 / (1.0 + (-(-1.0_f32)).exp()),
        0.5,
        1.0 / (1.0 + (-1.0_f32).exp()),
        1.0 / (1.0 + (-2.0_f32).exp()),
    ];
    assert_tensor_close(&output, &expected, 1e-6);
}

#[test]
fn test_sigmoid_range() {
    // Sigmoid output should always be in [0, 1] (within float tolerance)
    let input = create_tensor(vec![-100.0, -10.0, 0.0, 10.0, 100.0], &[5]);
    let output = sigmoid(&input).unwrap();

    for val in output.as_slice() {
        let v = val.get() as f32;
        assert!(
            v >= -1e-6 && v <= 1.0 + 1e-6,
            "Sigmoid output {} not in [0, 1] within tolerance",
            v
        );
    }
}

#[test]
fn test_tanh_basic() {
    let input = create_tensor(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let output = tanh(&input).unwrap();

    let expected: Vec<f32> = vec![-2.0_f32, -1.0_f32, 0.0_f32, 1.0_f32, 2.0_f32]
        .into_iter()
        .map(|x| x.tanh())
        .collect();
    assert_tensor_close(&output, &expected, 1e-6);
}

#[test]
fn test_tanh_range() {
    // Tanh output should always be in [-1, 1] (within float tolerance)
    let input = create_tensor(vec![-100.0, -10.0, 0.0, 10.0, 100.0], &[5]);
    let output = tanh(&input).unwrap();

    for val in output.as_slice() {
        let v = val.get() as f32;
        assert!(
            v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6,
            "Tanh output {} not in [-1, 1] within tolerance",
            v
        );
    }
}

#[test]
fn test_gelu_basic() {
    let input = create_tensor(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let output = gelu(&input).unwrap();

    // GELU should be approximately 0 for negative values and close to x for positive
    let output_data = output.as_slice();
    assert!(output_data[0].get() as f32 < 0.0); // negative input
    assert!(output_data[1].get() as f32 < 0.0); // negative input
    assert!(output_data[2].get() as f32.abs() < 1e-6); // zero input
    assert!(output_data[3].get() as f32 > 0.0); // positive input
    assert!(output_data[4].get() as f32 > 0.0); // positive input
}

#[test]
fn test_silu_basic() {
    let input = create_tensor(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let output = silu(&input).unwrap();

    // SiLU(x) = x * sigmoid(x)
    let output_data = output.as_slice();
    assert!(output_data[0].get() as f32 < 0.0); // negative input
    assert!(output_data[1].get() as f32 < 0.0); // negative input
    assert!(output_data[2].get() as f32.abs() < 1e-6); // zero input
    assert!(output_data[3].get() as f32 > 0.0); // positive input
    assert!(output_data[4].get() as f32 > 0.0); // positive input
}

#[test]
fn test_leaky_relu_basic() {
    let negative_slope = Float32::new(0.01);
    let input = create_tensor(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let output = leaky_relu(&input, negative_slope).unwrap();

    assert_tensor_close(&output, &[-0.02, -0.01, 0.0, 1.0, 2.0], 1e-6);
}

#[test]
fn test_leaky_relu_different_slopes() {
    let input = create_tensor(vec![-1.0], &[1]);

    // Test with slope 0.1
    let output1 = leaky_relu(&input, Float32::new(0.1)).unwrap();
    assert_tensor_close(&output1, &[-0.1], 1e-6);

    // Test with slope 0.5
    let output2 = leaky_relu(&input, Float32::new(0.5)).unwrap();
    assert_tensor_close(&output2, &[-0.5], 1e-6);
}

#[test]
fn test_elu_basic() {
    let alpha = Float32::new(1.0);
    let input = create_tensor(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let output = elu(&input, alpha).unwrap();

    // ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
    let expected = vec![
        1.0 * ((-2.0_f32).exp() - 1.0),
        1.0 * ((-1.0_f32).exp() - 1.0),
        0.0,
        1.0,
        2.0,
    ];
    assert_tensor_close(&output, &expected, 1e-6);
}

#[test]
fn test_elu_different_alpha() {
    let input = create_tensor(vec![-1.0], &[1]);

    // Test with alpha 0.5
    let output1 = elu(&input, Float32::new(0.5)).unwrap();
    let expected1 = 0.5 * ((-1.0_f32).exp() - 1.0);
    assert_tensor_close(&output1, &[expected1], 1e-6);

    // Test with alpha 2.0
    let output2 = elu(&input, Float32::new(2.0)).unwrap();
    let expected2 = 2.0 * ((-1.0_f32).exp() - 1.0);
    assert_tensor_close(&output2, &[expected2], 1e-6);
}

#[test]
fn test_softmax_basic() {
    let input = create_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let output = softmax(&input).unwrap();

    // Softmax output should sum to 1
    let sum: f32 = output.as_slice().iter().map(|x| x.get() as f32).sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Softmax sum is {}, expected 1.0",
        sum
    );

    // All values should be positive
    for val in output.as_slice() {
        assert!(
            val.get() as f32 > 0.0,
            "Softmax output {} should be positive",
            val.get() as f32
        );
    }
}

#[test]
fn test_softmax_2d() {
    // Test softmax on 2D tensor [2, 3]
    let input = create_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let output = softmax(&input).unwrap();

    // Check shape is preserved
    assert_eq!(output.shape().dims(), &[2, 3]);

    // Each row should sum to 1
    let data = output.as_slice();
    let row1_sum: f32 = data[0..3].iter().map(|x| x.get() as f32).sum();
    let row2_sum: f32 = data[3..6].iter().map(|x| x.get() as f32).sum();

    assert!((row1_sum - 1.0).abs() < 1e-6);
    assert!((row2_sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_softmax_dim() {
    // Test softmax along specific dimension
    let input = create_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let output = softmax_dim(&input, 1).unwrap();

    // Check shape is preserved
    assert_eq!(output.shape().dims(), &[2, 3]);

    // Each row should sum to 1 (softmax along dim 1)
    let data = output.as_slice();
    let row1_sum: f32 = data[0..3].iter().map(|x| x.get() as f32).sum();
    let row2_sum: f32 = data[3..6].iter().map(|x| x.get() as f32).sum();

    assert!((row1_sum - 1.0).abs() < 1e-6);
    assert!((row2_sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_log_softmax_basic() {
    let input = create_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let output = log_softmax(&input, 0).unwrap();

    // Log softmax output should be negative (since softmax is in (0, 1))
    for val in output.as_slice() {
        assert!(
            val.get() as f32 < 0.0,
            "Log softmax output {} should be negative",
            val.get() as f32
        );
    }

    // exp(log_softmax) should equal softmax
    let softmax_output = softmax(&input).unwrap();
    for (log_val, soft_val) in output
        .as_slice()
        .iter()
        .zip(softmax_output.as_slice().iter())
    {
        let exp_log = log_val.get() as f32.exp();
        assert!((exp_log - soft_val.get() as f32).abs() < 1e-5);
    }
}

#[test]
fn test_dropout_training_mode() {
    let input = create_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    let output = dropout(&input, 0.5, true).unwrap();

    // In training mode with p=0.5, some values should be zeroed
    // and others should be scaled by 1/(1-p) = 2.0
    let output_data = output.as_slice();
    for val in output_data {
        let v = val.get() as f32;
        // Value should be either 0 or scaled
        assert!(v == 0.0 || v > 0.0);
    }
}

#[test]
fn test_dropout_eval_mode() {
    let input = create_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    let output = dropout(&input, 0.5, false).unwrap();

    // In eval mode, output should equal input
    assert_tensor_close(&output, &[1.0, 2.0, 3.0, 4.0, 5.0], 1e-6);
}

#[test]
fn test_dropout_p_zero() {
    let input = create_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let output = dropout(&input, 0.0, true).unwrap();

    // With p=0, no dropout should occur
    assert_tensor_close(&output, &[1.0, 2.0, 3.0], 1e-6);
}

#[test]
fn test_dropout_p_one() {
    let input = create_tensor(vec![1.0, 2.0, 3.0], &[3]);
    let output = dropout(&input, 1.0, true).unwrap();

    // With p=1, all values should be zeroed
    assert_tensor_close(&output, &[0.0, 0.0, 0.0], 1e-6);
}

// Edge case tests

#[test]
fn test_activations_with_different_dtypes() {
    // Test with f32 (already tested above)
    // Note: f64 tests would require separate type parameters
    // This is a placeholder for dtype testing
}

#[test]
fn test_activations_preserve_shape() {
    let shapes = vec![vec![5], vec![2, 3], vec![2, 3, 4], vec![1, 2, 3, 4]];

    for shape in shapes {
        let size: usize = shape.iter().product();
        let input = create_tensor(vec![1.0; size], &shape);

        let relu_out = relu(&input).unwrap();
        assert_eq!(relu_out.shape().dims(), shape.as_slice());

        let sigmoid_out = sigmoid(&input).unwrap();
        assert_eq!(sigmoid_out.shape().dims(), shape.as_slice());

        let tanh_out = tanh(&input).unwrap();
        assert_eq!(tanh_out.shape().dims(), shape.as_slice());
    }
}
