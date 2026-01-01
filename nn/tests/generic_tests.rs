//! Comprehensive tests for B<S<T>> generic functionality
//!
//! These tests validate that neural network components work correctly
//! with the generic Backend-Storage-DataType architecture.

use num_traits::Zero;
use proptest::prelude::*;

use backend::CpuBackend;
use dtype::float::Float32;
use nn::{BatchNorm2d, Linear, Module};
use storage::DenseStorage;
use tensor::Tensor;

fn arb_tensor_data_with_len(len: usize) -> impl Strategy<Value = Vec<Float32>> {
    prop::collection::vec(prop::num::f32::NORMAL.prop_map(Float32::new), len..=len)
}

/// Generate small tensor shapes suitable for neural network operations
fn arb_linear_input() -> impl Strategy<Value = (usize, usize, usize, Vec<Float32>)> {
    (1..=50usize, 1..=50usize, 1..=10usize).prop_flat_map(
        |(in_features, out_features, batch_size)| {
            let input_size = batch_size * in_features;
            (
                Just(in_features),
                Just(out_features),
                Just(batch_size),
                arb_tensor_data_with_len(input_size),
            )
        },
    )
}

fn arb_linear_consistent_input() -> impl Strategy<Value = (usize, usize, usize, Vec<Float32>)> {
    (1..=20usize, 1..=20usize, 1..=5usize).prop_flat_map(
        |(in_features, out_features, batch_size)| {
            let input_size = batch_size * in_features;
            (
                Just(in_features),
                Just(out_features),
                Just(batch_size),
                arb_tensor_data_with_len(input_size),
            )
        },
    )
}

fn arb_batchnorm2d_input() -> impl Strategy<Value = (usize, usize, usize, usize, Vec<Float32>)> {
    (1..=32usize, 1..=5usize, 1..=16usize, 1..=16usize).prop_flat_map(
        |(num_features, batch_size, height, width)| {
            let input_size = batch_size * num_features * height * width;
            (
                Just(num_features),
                Just(batch_size),
                Just(height),
                Just(width),
                arb_tensor_data_with_len(input_size),
            )
        },
    )
}

proptest! {
    #[test]
    fn prop_linear_generic_forward(
        (in_features, out_features, batch_size, data) in arb_linear_input()
    ) {
        // Create a linear layer with the generic B<S<T>> signature
        let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_features,
            out_features
        ).unwrap();

        let input = Tensor::from_vec(data, &[batch_size, in_features]).unwrap();

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
        (num_features, batch_size, height, width, data) in arb_batchnorm2d_input()
    ) {
        // Create BatchNorm2d with generic B<S<T>> signature
        let batchnorm = BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            num_features, 1e-5, 0.1
        ).unwrap();

        // Create input tensor [N, C, H, W]
        let input = Tensor::from_vec(
            data,
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
        (in_features, out_features, batch_size, data) in arb_linear_consistent_input()
    ) {
        let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_features,
            out_features
        ).unwrap();

        let input = Tensor::from_vec(data, &[batch_size, in_features]).unwrap();

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
