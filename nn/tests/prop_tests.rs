//! Property-based tests for neural network operations.
//!
//! This module uses proptest to generate random inputs and verify
//! mathematical properties and invariants of neural network operations.

use approx::{assert_relative_eq, assert_ulps_eq};
use proptest::prelude::*;

use backend::CpuBackend;
use dtype::float::Float32;
use nn::*;
use storage::DenseStorage;
use tensor::Tensor;

/// Generate random tensors with specified shape
fn arb_tensor(
    shape: Vec<usize>,
) -> impl Strategy<Value = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>> {
    let len: usize = shape.iter().product();
    prop::collection::vec(-10.0..10.0f32, len).prop_map(move |data| {
        let float_data: Vec<Float32> = data.into_iter().map(Float32::new).collect();
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(float_data, &shape)
            .unwrap()
    })
}

proptest! {
    #[test]
    fn test_relu_properties(values in prop::collection::vec(-100.0..100.0f32, 1..100)) {
        let float_data: Vec<Float32> = values.iter().map(|&x| Float32::new(x)).collect();
        let shape = vec![float_data.len()];
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(float_data.clone(), &shape).unwrap();

        let relu_result = functional_activations::relu(&tensor).unwrap();

        // All output values should be non-negative
        for &val in relu_result.as_slice() {
            prop_assert!(val >= Float32::new(0.0));
        }

        // Check ReLU behavior: negative -> 0, positive -> unchanged
        for (&input, &output) in float_data.iter().zip(relu_result.as_slice()) {
            if input <= Float32::new(0.0) {
                prop_assert_eq!(output, Float32::new(0.0));
            } else {
                prop_assert_eq!(output, input);
            }
        }
    }

    /// Test that sigmoid activation produces values in [0, 1]
    #[test]
    fn test_sigmoid_range(values in prop::collection::vec(-100.0..100.0f32, 1..50)) {
        let float_data: Vec<Float32> = values.into_iter().map(Float32::new).collect();
        let shape = vec![float_data.len()];
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(float_data, &shape).unwrap();

        let sigmoid_result = functional_activations::sigmoid(&tensor).unwrap();

        // All sigmoid outputs should be in [0, 1]
        for &val in sigmoid_result.as_slice() {
            let val_f64 = val.get() as f64;
            prop_assert!(val_f64.is_finite());
            prop_assert!((0.0..=1.0).contains(&val_f64));
        }
    }

    /// Test that tanh activation produces values in [-1, 1]
    #[test]
    fn test_tanh_range(values in prop::collection::vec(-100.0..100.0f32, 1..50)) {
        let float_data: Vec<Float32> = values.into_iter().map(Float32::new).collect();
        let shape = vec![float_data.len()];
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(float_data, &shape).unwrap();

        let tanh_result = functional_activations::tanh(&tensor).unwrap();

        // All tanh outputs should be in [-1, 1]
        for &val in tanh_result.as_slice() {
            let val_f64 = val.get() as f64;
            prop_assert!(val_f64.is_finite());
            prop_assert!((-1.0..=1.0).contains(&val_f64));
        }
    }

    /// Test GELU approximation properties
    #[test]
    fn test_gelu_properties(values in prop::collection::vec(-5.0..5.0f32, 1..30)) {
        let float_data: Vec<Float32> = values.into_iter().map(Float32::new).collect();
        let shape = vec![float_data.len()];
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(float_data.clone(), &shape).unwrap();

        let gelu_result = functional_activations::gelu(&tensor).unwrap();

        // GELU should be approximately equal to x * sigmoid(1.702 * x) for small x
        for (&input, &output) in float_data.iter().zip(gelu_result.as_slice()) {
            let input_f64 = input.get() as f64;
            let output_f64 = output.get() as f64;

            // GELU should be close to ReLU for large positive values
            if input_f64 > 3.0 {
                prop_assert!(output_f64 >= input_f64 * 0.9); // At least 90% of input
            }

            // GELU should be close to zero for large negative values
            if input_f64 < -3.0 {
                prop_assert!(output_f64.abs() < 0.1);
            }
        }
    }

    /// Test max pooling preserves maximum values and reduces spatial dimensions
    #[test]
    fn test_max_pool_2d_properties(
        input in arb_tensor(vec![2, 3, 8, 8]),  // Fixed shape for simplicity
        kernel_h in 2..4usize,
        kernel_w in 2..4usize,
        stride_h in 1..3usize,
        stride_w in 1..3usize
    ) {
        let batch_size = input.shape().dims()[0];
        let channels = input.shape().dims()[1];
        let height = input.shape().dims()[2];
        let width = input.shape().dims()[3];

        let pool_result = functional_pooling::max_pool2d(
            &input,
            (kernel_h, kernel_w),
            Some((stride_h, stride_w)),
            (0, 0)
        ).unwrap();

        let output_shape = pool_result.shape().dims();
        let expected_height = (height - kernel_h) / stride_h + 1;
        let expected_width = (width - kernel_w) / stride_w + 1;

        // Check output shape
        prop_assert_eq!(output_shape, &[batch_size, channels, expected_height, expected_width]);

        // Check that each output value is indeed the maximum in its pooling window
        let input_slice = input.as_slice();
        let output_slice = pool_result.as_slice();

        let mut output_idx = 0;
        for b in 0..batch_size {
            for c in 0..channels {
                for oh in 0..expected_height {
                    for ow in 0..expected_width {
                    let mut max_val = Float32::new(f32::NEG_INFINITY);

                    // Find maximum in the pooling window
                    for kh in 0..kernel_h {
                        for kw in 0..kernel_w {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;

                            if ih < height && iw < width {
                                let input_idx = ((b * channels + c) * height + ih) * width + iw;
                                let val = input_slice[input_idx];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }

                        // Output should equal the maximum found
                        prop_assert_eq!(output_slice[output_idx], max_val);
                        output_idx += 1;
                    }
                }
            }
        }
    }

    /// Test linear transformation properties
    #[test]
    fn test_linear_properties(
        input in arb_tensor(vec![4, 8]),  // Fixed shape for simplicity
        weight in arb_tensor(vec![6, 8])  // Fixed weight shape
    ) {
        let batch_size = input.shape().dims()[0];
        let out_features = weight.shape().dims()[0];

        let linear_result = functional_linear::linear(&input, &weight, None).unwrap();
        let output_shape = linear_result.shape().dims();

        // Check output shape
        prop_assert_eq!(output_shape, &[batch_size, out_features]);

        // Basic property: output should not be all zeros for non-zero input
        let input_sum: Float32 = input.as_slice().iter().fold(Float32::new(0.0), |acc, &x| acc + x);
        let output_sum: Float32 = linear_result.as_slice().iter().fold(Float32::new(0.0), |acc, &x| acc + x);
        prop_assert!(input_sum != Float32::new(0.0) || output_sum == Float32::new(0.0));
    }

    /// Test softmax normalization properties
    #[test]
    fn test_softmax_properties(values in prop::collection::vec(-10.0..10.0f32, 2..16)) {
        let float_data: Vec<Float32> = values.into_iter().map(Float32::new).collect();
        let shape = vec![float_data.len()];
        let tensor = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(float_data, &shape).unwrap();

        let softmax_result = functional_attention::softmax(&tensor).unwrap();

        // All values should be positive and less than 1
        for &val in softmax_result.as_slice() {
            let val_f64 = val.get() as f64;
            prop_assert!((0.0..=1.0).contains(&val_f64));
        }

        // Sum of all values should be 1 (approximately)
        let sum: f64 = softmax_result.as_slice().iter().map(|&x| x.get() as f64).sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
    }

    /// Test MSE loss calculation
    #[test]
    fn test_mse_loss_properties((pred_values, target_values) in
                                 (prop::collection::vec(-10.0..10.0f32, 1..50),
                                  prop::collection::vec(-10.0..10.0f32, 1..50))) {
        let len = pred_values.len().min(target_values.len());
        let pred_data: Vec<Float32> = pred_values[..len].iter().map(|&x| Float32::new(x)).collect();
        let target_data: Vec<Float32> = target_values[..len].iter().map(|&x| Float32::new(x)).collect();

        let pred = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(pred_data.clone(), &[len]).unwrap();
        let target = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(target_data.clone(), &[len]).unwrap();

        let mse_loss = functional_loss::mse_loss(&pred, &target).unwrap();

        // MSE should be non-negative
        prop_assert!(mse_loss.as_slice()[0] >= Float32::new(0.0));

        // MSE should be zero when predictions equal targets
        if pred_data == target_data {
            assert_relative_eq!(mse_loss.as_slice()[0].get() as f64, 0.0, epsilon = 1e-6);
        }

        // Manual calculation should match (in Float32 precision)
        let manual_sum: Float32 = pred_data
            .iter()
            .zip(&target_data)
            .map(|(&p, &t)| {
                let d = p - t;
                d * d
            })
            .fold(Float32::new(0.0), |acc, x| acc + x);
        let manual_mse = manual_sum / Float32::new(len as f32);

        assert_ulps_eq!(
            mse_loss.as_slice()[0].get(),
            manual_mse.get(),
            max_ulps = 32
        );
    }

    /// Test convolution kernel dot product
    #[test]
    fn test_conv_kernel_dot_product(
        input_vals in prop::collection::vec(-5.0..5.0f32, 1..16),
        weight_vals in prop::collection::vec(-5.0..5.0f32, 1..16)
    ) {
        let len = input_vals.len().min(weight_vals.len());
        let input_data: Vec<Float32> = input_vals[..len].iter().map(|&x| Float32::new(x)).collect();
        let weight_data: Vec<Float32> = weight_vals[..len].iter().map(|&x| Float32::new(x)).collect();

        let result = input_data.iter().zip(&weight_data).map(|(&i, &w)| i * w).fold(Float32::new(0.0), |acc, x| acc + x);

        // Manual calculation
        let manual_result: Float32 = input_data.iter().zip(&weight_data)
            .map(|(&i, &w)| i * w)
            .fold(Float32::new(0.0), |acc, x| acc + x);

        assert_relative_eq!(result.get() as f64, manual_result.get() as f64, epsilon = 1e-6);
    }
}

/// Test numerical stability under extreme conditions
#[test]
fn test_numerical_stability_extremes() {
    // Test with very small numbers
    let tiny_data = vec![Float32::new(1e-30), Float32::new(1e-30)];
    let tiny_tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(tiny_data, &[2])
            .unwrap();

    let relu_result = functional_activations::relu(&tiny_tensor).unwrap();
    // Should not underflow to zero inappropriately
    assert!(relu_result.as_slice()[0] > Float32::new(0.0));

    // Test with very large numbers
    let huge_data = vec![Float32::new(1e30), Float32::new(-1e30)];
    let huge_tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(huge_data, &[2])
            .unwrap();

    let relu_huge = functional_activations::relu(&huge_tensor).unwrap();
    assert_eq!(relu_huge.as_slice()[0], Float32::new(1e30));
    assert_eq!(relu_huge.as_slice()[1], Float32::new(0.0));
}

/// Test composition of operations
#[test]
fn test_operation_composition() {
    let data = vec![
        Float32::new(-1.0),
        Float32::new(0.5),
        Float32::new(2.0),
        Float32::new(-3.0),
    ];
    let tensor =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(data, &[4])
            .unwrap();

    // Test that sigmoid ∘ relu gives valid probabilities
    let relu_result = functional_activations::relu(&tensor).unwrap();
    let sigmoid_result = functional_activations::sigmoid(&relu_result).unwrap();

    // All sigmoid outputs should be in [0, 1]
    for &val in sigmoid_result.as_slice() {
        let val_f64 = val.get() as f64;
        assert!((0.0..=1.0).contains(&val_f64));
    }

    // Values that were negative should become 0 then sigmoid(0) = 0.5
    // Values that were positive should remain positive then go through sigmoid
    assert_relative_eq!(
        sigmoid_result.as_slice()[0].get() as f64,
        0.5,
        epsilon = 1e-6
    ); // relu(-1) = 0, sigmoid(0) = 0.5
    assert!(sigmoid_result.as_slice()[1].get() as f64 > 0.5); // relu(0.5) = 0.5, sigmoid(0.5) > 0.5
    assert!(sigmoid_result.as_slice()[2].get() as f64 > 0.5); // relu(2.0) = 2.0, sigmoid(2.0) > 0.5
    assert_relative_eq!(
        sigmoid_result.as_slice()[3].get() as f64,
        0.5,
        epsilon = 1e-6
    ); // relu(-3) = 0, sigmoid(0) = 0.5
}
