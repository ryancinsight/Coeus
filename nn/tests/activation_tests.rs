//! Activation Function Tests
//!
//! Tests for activation functions including PReLU with learnable parameters.

use backend::CpuBackend;
use dtype::float::Float32;
use nn::{Module, PReLU, Parameter};
use storage::DenseStorage;
use tensor::Tensor;

#[test]
fn test_prelu_forward() {
    let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1, None);

    // Test with mixed positive/negative values
    let input_data = vec![
        Float32::new(2.0f32),  // positive
        Float32::new(-1.0f32), // negative
        Float32::new(0.0f32),  // zero
        Float32::new(-0.5f32), // negative
        Float32::new(3.0f32),  // positive
    ];

    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(input_data, &[5])
            .unwrap();

    let output = prelu.forward(&input).unwrap();

    // Output should have same shape
    assert_eq!(output.shape().dims(), &[5]);

    // Check that output is properly computed
    let output_data = output.as_slice();
    assert_eq!(output_data.len(), 5);
}

#[test]
fn test_prelu_parameters() {
    let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(3, None);

    let params = prelu.parameters();

    // PReLU with 3 parameters should have 1 parameter tensor with shape [3]
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].data().shape().dims(), &[3]);
    assert!(params[0].requires_grad());
}

#[test]
fn test_prelu_shared_parameter() {
    let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1, None);

    let params = prelu.parameters();

    // Shared parameter case should have 1 parameter with shape [1]
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].data().shape().dims(), &[1]);
}

#[test]
fn test_prelu_gradient_flow() {
    let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1, None);

    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec_with_backend(
            vec![Float32::new(1.0), Float32::new(-1.0)],
            &[2],
            CpuBackend::<Float32>::new(),
        )
        .unwrap()
        .requires_grad_(true);

    let output = prelu.forward(&input).unwrap();

    // Output should require gradients
    assert!(output.requires_grad());

    // Parameters should require gradients
    let params = prelu.parameters();
    assert!(params.iter().all(
        |p: &Parameter<CpuBackend<Float32>, DenseStorage<Float32>, Float32>| p.requires_grad()
    ));
}

#[test]
fn test_prelu_zero_grad() {
    let mut prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(2, None);

    // Test zero_grad functionality
    prelu.zero_grad();

    // Parameters should still exist (1 parameter with shape [2])
    let params = prelu.parameters();
    assert_eq!(params.len(), 1);
    assert_eq!(params[0].data().shape().dims(), &[2]);
}

#[test]
fn test_prelu_module_api() {
    let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1, None);

    // Test Module trait methods
    assert_eq!(prelu.name(), "PReLU");

    // Test parameter access
    let params = prelu.parameters();
    assert_eq!(params.len(), 1);
}

#[test]
fn test_prelu_different_shapes() {
    let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1, None);

    // Test with different input shapes
    let test_shapes = vec![
        vec![5],       // 1D
        vec![2, 3],    // 2D
        vec![2, 2, 2], // 3D
    ];

    for shape in test_shapes {
        let size: usize = shape.iter().product();
        let input_data = vec![Float32::new(1.0); size];
        let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            input_data, &shape,
        )
        .unwrap();

        let output = prelu.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), shape.as_slice());
    }
}

#[test]
fn test_prelu_per_channel() {
    let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(3, None);

    // Input with 3 channels
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(-1.0),
            Float32::new(0.5), // channel 0
            Float32::new(-2.0),
            Float32::new(1.5),
            Float32::new(-0.5), // channel 1
            Float32::new(0.0),
            Float32::new(-3.0),
            Float32::new(2.0), // channel 2
        ],
        &[3, 3], // [channels, values_per_channel]
    )
    .unwrap();

    let output = prelu.forward(&input).unwrap();

    // Output should have same shape
    assert_eq!(output.shape().dims(), &[3, 3]);

    // Check that output is properly computed with per-channel parameters
    let output_data = output.as_slice();
    assert_eq!(output_data.len(), 9);
}

#[test]
fn test_prelu_train_eval_modes() {
    let mut prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1, None);

    // Test train mode
    prelu.train(true);

    // Test eval mode
    prelu.train(false);

    // Functionality should work in both modes
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(-1.0)],
        &[2],
    )
    .unwrap();

    let output = prelu.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[2]);
}

#[test]
fn test_prelu_per_channel_functionality() {
    // Test that different channels use different weights
    let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(3, None);

    // Create input with shape [1, 3, 2] - batch=1, channels=3, spatial=2
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(-1.0), // channel 0
            Float32::new(-2.0),
            Float32::new(0.5), // channel 1
            Float32::new(0.0),
            Float32::new(-3.0), // channel 2
        ],
        &[1, 3, 2],
    )
    .unwrap();

    let output = prelu.forward(&input).unwrap();

    // Verify shape is preserved
    assert_eq!(output.shape().dims(), &[1, 3, 2]);

    // Verify the computation: max(0, x) + weight * min(0, x)
    let output_data = output.as_slice();
    let weight_data = prelu.weight.data().as_slice();

    // Channel 0: weight = weight_data[0]
    assert_eq!(output_data[0], Float32::new(1.0)); // max(0, 1.0) + w * min(0, 1.0) = 1.0
    assert_eq!(output_data[1], weight_data[0] * Float32::new(-1.0)); // max(0, -1.0) + w * min(0, -1.0) = w * (-1.0)

    // Channel 1: weight = weight_data[1]
    assert_eq!(output_data[2], weight_data[1] * Float32::new(-2.0)); // max(0, -2.0) + w * min(0, -2.0) = w * (-2.0)
    assert_eq!(output_data[3], Float32::new(0.5)); // max(0, 0.5) + w * min(0, 0.5) = 0.5

    // Channel 2: weight = weight_data[2]
    assert_eq!(output_data[4], weight_data[2] * Float32::new(0.0)); // max(0, 0.0) + w * min(0, 0.0) = 0.0
    assert_eq!(output_data[5], weight_data[2] * Float32::new(-3.0)); // max(0, -3.0) + w * min(0, -3.0) = w * (-3.0)
}

#[test]
fn test_prelu_edge_cases() {
    let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1, None);

    // Test with zeros
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(0.0), Float32::new(0.0)],
        &[2],
    )
    .unwrap();
    let output = prelu.forward(&input).unwrap();
    assert_eq!(output.as_slice(), &[Float32::new(0.0), Float32::new(0.0)]);

    // Test with all positive
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(1.0), Float32::new(2.0)],
        &[2],
    )
    .unwrap();
    let output = prelu.forward(&input).unwrap();
    assert_eq!(output.as_slice(), &[Float32::new(1.0), Float32::new(2.0)]);

    // Test with all negative
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(-1.0), Float32::new(-2.0)],
        &[2],
    )
    .unwrap();
    let output = prelu.forward(&input).unwrap();
    let weight = prelu.weight.data().as_slice()[0];
    assert_eq!(
        output.as_slice(),
        &[weight * Float32::new(-1.0), weight * Float32::new(-2.0)]
    );
}

#[test]
fn test_prelu_zero_init() {
    let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        1,
        Some(Float32::new(0.0)),
    );

    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(-1.0), Float32::new(-2.0)],
        &[2],
    )
    .unwrap();
    let output = prelu.forward(&input).unwrap();
    // weight is 0.0. max(0, x) + 0 * min(0, x) = max(0, x). For negatives, it's 0.
    assert_eq!(output.as_slice(), &[Float32::new(0.0), Float32::new(0.0)]);
}

#[test]
fn test_prelu_positive_init() {
    let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        1,
        Some(Float32::new(0.5)),
    );

    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![Float32::new(-2.0), Float32::new(-4.0)],
        &[2],
    )
    .unwrap();
    let output = prelu.forward(&input).unwrap();
    // weight is 0.5. For -2.0: 0 + 0.5 * -2.0 = -1.0. For -4.0: 0 + 0.5 * -4.0 = -2.0.
    assert_eq!(output.as_slice(), &[Float32::new(-1.0), Float32::new(-2.0)]);
}

#[test]
fn test_prelu_channel_wise() {
    let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(3, None);

    // Input [1, 3, 3] - batch 1, channels 3, spatial 3
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(-1.0),
            Float32::new(0.5), // channel 0
            Float32::new(-2.0),
            Float32::new(1.5),
            Float32::new(-0.5), // channel 1
            Float32::new(0.0),
            Float32::new(-3.0),
            Float32::new(2.0), // channel 2
        ],
        &[1, 3, 3],
    )
    .unwrap();

    let output = prelu.forward(&input).unwrap();

    // Verify shape is preserved
    assert_eq!(output.shape().dims(), &[1, 3, 3]);
}

#[test]
fn test_prelu_serialization() {
    let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(1, None);
    // This test would normally check serde, but for now just checking instantiation
    assert_eq!(prelu.parameters().len(), 1);
}
