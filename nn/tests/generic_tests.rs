//! Comprehensive tests for B<S<T>> generic functionality
//!
//! These tests validate that neural network components work correctly
//! with the generic Backend-Storage-DataType architecture.

use approx::assert_relative_eq;
use num_traits::Zero;
use proptest::prelude::*;

use backend::CpuBackend;
use dtype::float::Float32;
use nn::{BatchNorm2d, Linear, Module};
use storage::DenseStorage;
use tensor::Tensor;

/// Type alias for our test tensor type
type TestTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

/// Generate random tensor data
fn arb_tensor_data() -> impl Strategy<Value = Vec<Float32>> {
    prop::collection::vec(prop::num::f32::NORMAL.prop_map(Float32::new), 1..=1000)
}

/// Generate small tensor shapes suitable for neural network operations
fn arb_nn_shape() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1..=100usize, 2..=4).prop_filter("Valid NN shapes", |s| {
        s.iter().all(|&x| x > 0) && s.len() >= 2 && s.len() <= 4
    })
}

/// Generate compatible input/output shapes for linear layers
fn arb_linear_shapes() -> impl Strategy<Value = (usize, usize)> {
    (1..=100usize, 1..=100usize)
}

/// Generate BatchNorm2D compatible shapes [N, C, H, W]
fn arb_batchnorm2d_shapes() -> impl Strategy<Value = Vec<usize>> {
    (1..=10usize, 1..=64usize, 1..=32usize, 1..=32usize).prop_map(|(n, c, h, w)| vec![n, c, h, w])
}

proptest! {
    #[test]
    fn prop_linear_generic_forward(
        in_features in 1..=50usize,
        out_features in 1..=50usize,
        batch_size in 1..=10usize,
        data in arb_tensor_data()
    ) {
        // Create a linear layer with the generic B<S<T>> signature
        let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_features,
            out_features
        ).unwrap();

        // Create input tensor with compatible shape
        let input_size = batch_size * in_features;
        prop_assume!(data.len() >= input_size);
        let input_data = &data[..input_size];
        let input = Tensor::from_vec(input_data.to_vec(), &[batch_size, in_features]).unwrap();

        // Forward pass
        let output = linear.forward(&input).unwrap();

        // Verify output shape
        prop_assert_eq!(output.shape().dims(), &[batch_size, out_features]);

        // Verify output is not all zeros (weights are initialized)
        let output_data = output.as_slice();
        let has_non_zero = output_data.iter().any(|&x| x != Float32::zero());
        prop_assert!(has_non_zero, "Output should not be all zeros");
    }

    #[test]
    fn prop_batchnorm2d_generic_forward(
        num_features in 1..=32usize,
        batch_size in 1..=5usize,
        height in 1..=16usize,
        width in 1..=16usize,
        data in arb_tensor_data()
    ) {
        // Create BatchNorm2d with generic B<S<T>> signature
        let batchnorm = BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            num_features, 1e-5, 0.1
        ).unwrap();

        // Create input tensor [N, C, H, W]
        let input_size = batch_size * num_features * height * width;
        prop_assume!(data.len() >= input_size);
        let input_data = &data[..input_size];
        let input = Tensor::from_vec(
            input_data.to_vec(),
            &[batch_size, num_features, height, width]
        ).unwrap();

        // Forward pass
        let output = batchnorm.forward(&input).unwrap();

        // Verify output shape matches input
        prop_assert_eq!(output.shape().dims(), &[batch_size, num_features, height, width]);

        // During training, output should be normalized (mean close to 0 for large batches)
        if batch_size > 1 {
            // Check that output has reasonable statistics (not identical to input)
            let output_data = output.as_slice();
            let input_slice = input.as_slice();

            // Output should be different from input (due to normalization)
            let has_difference = output_data.iter().zip(input_slice.iter())
                .any(|(&o, &i)| (o.0 - i.0).abs() > 1e-6);
            prop_assert!(has_difference, "BatchNorm should modify input during training");
        }
    }

    #[test]
    fn prop_linear_parameters_generic(
        in_features in 1..=20usize,
        out_features in 1..=20usize
    ) {
        let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_features,
            out_features
        ).unwrap();

        // Get parameters using the generic Module trait
        let params = linear.parameters();

        // Should have weight and bias parameters
        prop_assert_eq!(params.len(), 2);

        // Weight should be [out_features, in_features]
        let weight_shape = params[0].data().shape().dims();
        prop_assert_eq!(weight_shape, &[out_features, in_features]);

        // Bias should be [out_features]
        let bias_shape = params[1].data().shape().dims();
        prop_assert_eq!(bias_shape, &[out_features]);
    }

    #[test]
    fn prop_batchnorm2d_parameters_generic(
        num_features in 1..=32usize
    ) {
        let batchnorm = BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            num_features, 1e-5, 0.1
        ).unwrap();

        let params = batchnorm.parameters();

        // Should have weight and bias parameters
        prop_assert_eq!(params.len(), 2);

        // Both weight and bias should be [num_features]
        for param in &params {
            prop_assert_eq!(param.data().shape().dims(), &[num_features]);
        }
    }

    #[test]
    fn prop_linear_modules_generic(
        in_features in 1..=20usize,
        out_features in 1..=20usize
    ) {
        let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_features,
            out_features
        ).unwrap();

        // Linear layer should have no sub-modules
        let modules = linear.modules();
        prop_assert_eq!(modules.len(), 0);
    }

    #[test]
    fn prop_batchnorm2d_modules_generic(
        num_features in 1..=32usize
    ) {
        let batchnorm = BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            num_features, 1e-5, 0.1
        ).unwrap();

        // BatchNorm should have no sub-modules
        let modules = batchnorm.modules();
        prop_assert_eq!(modules.len(), 0);
    }

    #[test]
    fn prop_linear_zero_grad_generic(
        in_features in 1..=20usize,
        out_features in 1..=20usize
    ) {
        let mut linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_features,
            out_features
        ).unwrap();

        // Test that zero_grad doesn't panic
        linear.zero_grad();

        // Basic functionality test - parameters should still exist
        let params = linear.parameters();
        prop_assert_eq!(params.len(), 2); // weight and bias
    }

    #[test]
    fn prop_batchnorm2d_zero_grad_generic(
        num_features in 1..=32usize
    ) {
        let mut batchnorm = BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            num_features,
            1e-5,
            0.1
        ).unwrap();

        // Test that zero_grad doesn't panic
        batchnorm.zero_grad();

        // Basic functionality test - parameters should still exist
        let params = batchnorm.parameters();
        prop_assert_eq!(params.len(), 2); // weight and bias
    }

    #[test]
    fn prop_linear_train_mode_generic(
        in_features in 1..=20usize,
        out_features in 1..=20usize
    ) {
        let mut linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_features,
            out_features
        ).unwrap();

        // Test training mode toggle
        linear.train(true);
        // Note: Linear doesn't currently track training mode, but the method should exist

        linear.train(false);
        // Note: Linear doesn't currently track training mode, but the method should exist
    }

    #[test]
    fn prop_batchnorm2d_train_mode_generic(
        num_features in 1..=32usize
    ) {
        let mut batchnorm = BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            num_features,
            1e-5,
            0.1
        ).unwrap();

        // Test training mode toggle
        batchnorm.train(true);
        // BatchNorm should track training mode for running stats
        prop_assert!(batchnorm.training, "Should be in training mode");

        batchnorm.train(false);
        prop_assert!(!batchnorm.training, "Should be in evaluation mode");
    }

    #[test]
    fn prop_linear_name_generic(
        in_features in 1..=20usize,
        out_features in 1..=20usize
    ) {
        let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_features,
            out_features
        ).unwrap();

        prop_assert_eq!(linear.name(), "Linear");
    }

    #[test]
    fn prop_batchnorm2d_name_generic(
        num_features in 1..=32usize
    ) {
        let batchnorm = BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            num_features, 1e-5, 0.1
        ).unwrap();

        prop_assert_eq!(batchnorm.name(), "BatchNorm2d");
    }

    #[test]
    fn prop_linear_consistent_output(
        in_features in 1..=20usize,
        out_features in 1..=20usize,
        batch_size in 1..=5usize,
        data in arb_tensor_data()
    ) {
        let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_features,
            out_features
        ).unwrap();

        // Create input
        let input_size = batch_size * in_features;
        prop_assume!(data.len() >= input_size);
        let input_data = &data[..input_size];
        let input = Tensor::from_vec(input_data.to_vec(), &[batch_size, in_features]).unwrap();

        // Get output twice - should be identical
        let output1 = linear.forward(&input).unwrap();
        let output2 = linear.forward(&input).unwrap();

        // Outputs should be identical (deterministic)
        let data1 = output1.as_slice();
        let data2 = output2.as_slice();
        prop_assert_eq!(data1, data2, "Linear forward should be deterministic");
    }

    #[test]
    fn prop_batchnorm2d_running_stats_update(
        num_features in 1..=16usize,
        batch_size in 2..=5usize,
        height in 1..=8usize,
        width in 1..=8usize
    ) {
        let mut batchnorm = BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            num_features,
            1e-5,
            0.1
        ).unwrap();

        // Start in training mode
        batchnorm.train(true);

        // Create input with known values
        let input_data = vec![Float32::new(1.0); batch_size * num_features * height * width];
        let input = Tensor::from_vec(
            input_data,
            &[batch_size, num_features, height, width]
        ).unwrap();

        // Forward pass - should update running statistics
        let _output = batchnorm.forward(&input).unwrap();

        // Running mean should be updated (input mean is 1.0)
        let running_mean = batchnorm.running_mean();
        let mean_data = running_mean.as_slice();
        for &mean_val in mean_data {
            prop_assert!(mean_val > Float32::zero(), "Running mean should be updated");
        }
    }
}
