//! Comprehensive unit tests for all functional operations
//!
//! Tests for stateless operations in nn::functional::ops::*
//! Validates Requirements 15.1

use backend::CpuBackend;
use dtype::float::Float32;
use nn::functional::ops::{conv::*, linear::*, normalization::*, pooling::*};
use storage::DenseStorage;
use tensor::Tensor;

type TestTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

fn create_tensor(data: Vec<f32>, shape: &[usize]) -> TestTensor {
    let float_data: Vec<Float32> = data.into_iter().map(Float32::new).collect();
    TestTensor::from_vec(float_data, shape).unwrap()
}

fn assert_close(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() < tolerance,
        "Expected {}, got {}, diff: {}",
        expected,
        actual,
        (actual - expected).abs()
    );
}

// ============================================================================
// Convolution Tests
// ============================================================================

#[test]
fn test_conv1d_basic() {
    // Input: [batch=1, channels=1, length=5]
    let input = create_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
    // Weight: [out_channels=1, in_channels=1, kernel_size=3]
    let weight = create_tensor(vec![1.0, 0.0, -1.0], &[1, 1, 3]);

    let output = conv1d(&input, &weight, None, 1, 0).unwrap();

    // Output shape: [1, 1, 3] (5 - 3 + 1 = 3)
    assert_eq!(output.shape().dims(), &[1, 1, 3]);
}

#[test]
fn test_conv1d_with_bias() {
    let input = create_tensor(vec![1.0; 10], &[1, 1, 10]);
    let weight = create_tensor(vec![0.5; 3], &[1, 1, 3]);
    let bias = create_tensor(vec![0.1], &[1]);

    let output = conv1d(&input, &weight, Some(&bias), 1, 0).unwrap();

    // Each output should be sum of kernel * input + bias
    // = (0.5 * 1.0 * 3) + 0.1 = 1.6
    for &val in output.as_slice() {
        assert_close(val.get(), 1.6, 1e-5);
    }
}

#[test]
fn test_conv1d_with_stride() {
    let input = create_tensor(vec![1.0; 10], &[1, 1, 10]);
    let weight = create_tensor(vec![1.0; 3], &[1, 1, 3]);

    // Stride = 2
    let output = conv1d(&input, &weight, None, 2, 0).unwrap();

    // Output length = (10 - 3) / 2 + 1 = 4
    assert_eq!(output.shape().dims(), &[1, 1, 4]);
}

#[test]
fn test_conv1d_with_padding() {
    let input = create_tensor(vec![1.0; 5], &[1, 1, 5]);
    let weight = create_tensor(vec![1.0; 3], &[1, 1, 3]);

    // Padding = 1
    let output = conv1d(&input, &weight, None, 1, 1).unwrap();

    // Output length = (5 + 2*1 - 3) / 1 + 1 = 5
    assert_eq!(output.shape().dims(), &[1, 1, 5]);
}

#[test]
fn test_conv2d_basic() {
    // Input: [batch=1, channels=1, height=4, width=4]
    let input = create_tensor(vec![1.0; 16], &[1, 1, 4, 4]);
    // Weight: [out_channels=1, in_channels=1, kernel_h=3, kernel_w=3]
    let weight = create_tensor(vec![1.0; 9], &[1, 1, 3, 3]);

    let output = conv2d(&input, &weight, None, (1, 1), (0, 0)).unwrap();

    // Output shape: [1, 1, 2, 2] (4-3+1=2 for both dimensions)
    assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
}

#[test]
fn test_conv2d_with_bias() {
    let input = create_tensor(vec![1.0; 16], &[1, 1, 4, 4]);
    let weight = create_tensor(vec![0.5; 9], &[1, 1, 3, 3]);
    let bias = create_tensor(vec![0.1], &[1]);

    let output = conv2d(&input, &weight, Some(&bias), (1, 1), (0, 0)).unwrap();

    // Each output should be sum of kernel * input + bias
    // = (0.5 * 1.0 * 9) + 0.1 = 4.6
    for &val in output.as_slice() {
        assert_close(val.get(), 4.6, 1e-5);
    }
}

#[test]
fn test_conv2d_multiple_channels() {
    // Input: [batch=1, channels=2, height=4, width=4]
    let input = create_tensor(vec![1.0; 32], &[1, 2, 4, 4]);
    // Weight: [out_channels=3, in_channels=2, kernel_h=3, kernel_w=3]
    let weight = create_tensor(vec![0.5; 54], &[3, 2, 3, 3]);

    let output = conv2d(&input, &weight, None, (1, 1), (0, 0)).unwrap();

    // Output shape: [1, 3, 2, 2]
    assert_eq!(output.shape().dims(), &[1, 3, 2, 2]);
}

#[test]
fn test_conv3d_basic() {
    // Input: [batch=1, channels=1, depth=3, height=3, width=3]
    let input = create_tensor(vec![1.0; 27], &[1, 1, 3, 3, 3]);
    // Weight: [out_channels=1, in_channels=1, kernel_d=2, kernel_h=2, kernel_w=2]
    let weight = create_tensor(vec![1.0; 8], &[1, 1, 2, 2, 2]);

    let output = conv3d(&input, &weight, None, (1, 1, 1), (0, 0, 0)).unwrap();

    // Output shape: [1, 1, 2, 2, 2] (3-2+1=2 for all dimensions)
    assert_eq!(output.shape().dims(), &[1, 1, 2, 2, 2]);
}

#[test]
fn test_pad_2d() {
    let input = create_tensor(vec![1.0; 12], &[1, 1, 3, 4]);
    let padded = pad_2d(&input, 1, 2).unwrap();

    // Padded shape: [1, 1, 5, 8] (3+2*1=5, 4+2*2=8)
    assert_eq!(padded.shape().dims(), &[1, 1, 5, 8]);
}

#[test]
fn test_pad_3d() {
    let input = create_tensor(vec![1.0; 24], &[1, 1, 2, 3, 4]);
    let padded = pad_3d(&input, 1, 1, 1).unwrap();

    // Padded shape: [1, 1, 4, 5, 6] (2+2*1=4, 3+2*1=5, 4+2*1=6)
    assert_eq!(padded.shape().dims(), &[1, 1, 4, 5, 6]);
}

// ============================================================================
// Linear Tests
// ============================================================================

#[test]
fn test_linear_basic() {
    // Input: [batch=2, in_features=3]
    let input = create_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    // Weight: [out_features=2, in_features=3]
    let weight = create_tensor(vec![1.0, 0.0, -1.0, 0.5, 0.5, 0.5], &[2, 3]);

    let output = linear(&input, &weight, None).unwrap();

    // Output shape: [2, 2]
    assert_eq!(output.shape().dims(), &[2, 2]);
}

#[test]
fn test_linear_with_bias() {
    let input = create_tensor(vec![1.0, 2.0], &[1, 2]);
    let weight = create_tensor(vec![1.0, 1.0], &[1, 2]);
    let bias = create_tensor(vec![0.5], &[1]);

    let output = linear(&input, &weight, Some(&bias)).unwrap();

    // Output = (1*1 + 2*1) + 0.5 = 3.5
    assert_close(output.as_slice()[0].get(), 3.5, 1e-6);
}

#[test]
fn test_linear_multidimensional_input() {
    // Input: [batch=2, seq=3, features=4]
    let input = create_tensor(vec![1.0; 24], &[2, 3, 4]);
    // Weight: [out_features=5, in_features=4]
    let weight = create_tensor(vec![0.5; 20], &[5, 4]);

    let output = linear(&input, &weight, None).unwrap();

    // Output shape: [2, 3, 5]
    assert_eq!(output.shape().dims(), &[2, 3, 5]);
}

#[test]
fn test_sparse_linear_basic() {
    let input = create_tensor(vec![1.0, 2.0], &[1, 2]);

    // Sparse weight: only (0, 0) = 1.0 and (0, 1) = 2.0
    let weight_data = vec![Float32::new(1.0), Float32::new(2.0)];
    let weight_indices = vec![(0, 0), (0, 1)];
    let weight_shape = (1, 2);

    let output = sparse_linear(&input, &weight_data, &weight_indices, weight_shape, None).unwrap();

    // Output = 1*1 + 2*2 = 5
    assert_close(output.as_slice()[0].get(), 5.0, 1e-6);
}

// ============================================================================
// Normalization Tests
// ============================================================================

#[test]
fn test_layer_norm_basic() {
    // Input: [batch=2, features=3]
    let input = create_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

    let output = layer_norm(&input, &[3], None, None, 1e-5).unwrap();

    // Output shape should match input
    assert_eq!(output.shape().dims(), &[2, 3]);

    // Each sample should be normalized (mean ≈ 0, std ≈ 1)
    let data = output.as_slice();

    // First sample: [1, 2, 3] normalized
    let sample1_mean = (data[0].get() + data[1].get() + data[2].get()) / 3.0;
    assert_close(sample1_mean, 0.0, 1e-5);
}

#[test]
fn test_layer_norm_with_weight_and_bias() {
    let input = create_tensor(vec![1.0, 2.0, 3.0], &[1, 3]);
    let weight = create_tensor(vec![2.0, 2.0, 2.0], &[3]);
    let bias = create_tensor(vec![0.5, 0.5, 0.5], &[3]);

    let output = layer_norm(&input, &[3], Some(&weight), Some(&bias), 1e-5).unwrap();

    // Output should be: normalized * weight + bias
    assert_eq!(output.shape().dims(), &[1, 3]);
}

#[test]
fn test_batch_norm_basic() {
    // Input: [batch=2, channels=2, spatial=3]
    let input = create_tensor(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 2, 3],
    );

    let output = batch_norm(&input, None, None, 1e-5).unwrap();

    // Output shape should match input
    assert_eq!(output.shape().dims(), &[2, 2, 3]);
}

#[test]
fn test_batch_norm_with_weight_and_bias() {
    let input = create_tensor(vec![1.0; 12], &[2, 2, 3]);
    let weight = create_tensor(vec![2.0, 2.0], &[2]);
    let bias = create_tensor(vec![0.5, 0.5], &[2]);

    let output = batch_norm(&input, Some(&weight), Some(&bias), 1e-5).unwrap();

    assert_eq!(output.shape().dims(), &[2, 2, 3]);
}

// ============================================================================
// Pooling Tests
// ============================================================================

#[test]
fn test_max_pool2d_basic() {
    // Input: [batch=1, channels=1, height=4, width=4]
    let input = create_tensor(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[1, 1, 4, 4],
    );

    // Kernel size 2x2, stride 2x2
    let output = max_pool2d(&input, (2, 2), Some((2, 2)), (0, 0)).unwrap();

    // Output shape: [1, 1, 2, 2]
    assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);

    // Check max values
    let data = output.as_slice();
    assert_close(data[0].get(), 6.0, 1e-6); // max of [1,2,5,6]
    assert_close(data[1].get(), 8.0, 1e-6); // max of [3,4,7,8]
    assert_close(data[2].get(), 14.0, 1e-6); // max of [9,10,13,14]
    assert_close(data[3].get(), 16.0, 1e-6); // max of [11,12,15,16]
}

#[test]
fn test_max_pool2d_with_padding() {
    let input = create_tensor(vec![1.0; 16], &[1, 1, 4, 4]);

    // With padding (1, 1)
    let output = max_pool2d(&input, (2, 2), Some((2, 2)), (1, 1)).unwrap();

    // Output shape: [1, 1, 3, 3] ((4+2*1-2)/2+1 = 3)
    assert_eq!(output.shape().dims(), &[1, 1, 3, 3]);
}

#[test]
fn test_avg_pool2d_basic() {
    // Input: [batch=1, channels=1, height=4, width=4]
    let input = create_tensor(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
        &[1, 1, 4, 4],
    );

    // Kernel size 2x2, stride 2x2
    let output = avg_pool2d(&input, (2, 2), Some((2, 2)), (0, 0)).unwrap();

    // Output shape: [1, 1, 2, 2]
    assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);

    // Check average values
    let data = output.as_slice();
    assert_close(data[0].get(), 3.5, 1e-6); // avg of [1,2,5,6]
    assert_close(data[1].get(), 5.5, 1e-6); // avg of [3,4,7,8]
    assert_close(data[2].get(), 11.5, 1e-6); // avg of [9,10,13,14]
    assert_close(data[3].get(), 13.5, 1e-6); // avg of [11,12,15,16]
}

#[test]
fn test_avg_pool2d_with_padding() {
    let input = create_tensor(vec![1.0; 16], &[1, 1, 4, 4]);

    // With padding (1, 1)
    let output = avg_pool2d(&input, (2, 2), Some((2, 2)), (1, 1)).unwrap();

    // Output shape: [1, 1, 3, 3]
    assert_eq!(output.shape().dims(), &[1, 1, 3, 3]);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_conv2d_shape_mismatch() {
    let input = create_tensor(vec![1.0; 16], &[1, 1, 4, 4]);
    // Wrong number of input channels
    let weight = create_tensor(vec![1.0; 18], &[1, 2, 3, 3]);

    let result = conv2d(&input, &weight, None, (1, 1), (0, 0));
    assert!(result.is_err());
}

#[test]
fn test_linear_shape_mismatch() {
    let input = create_tensor(vec![1.0, 2.0], &[1, 2]);
    // Wrong number of input features
    let weight = create_tensor(vec![1.0, 1.0, 1.0], &[1, 3]);

    let result = linear(&input, &weight, None);
    assert!(result.is_err());
}

#[test]
fn test_layer_norm_shape_mismatch() {
    let input = create_tensor(vec![1.0, 2.0, 3.0], &[1, 3]);
    // Wrong normalized shape
    let weight = create_tensor(vec![1.0, 1.0], &[2]);

    let result = layer_norm(&input, &[3], Some(&weight), None, 1e-5);
    assert!(result.is_err());
}

#[test]
fn test_pooling_stride_zero() {
    let input = create_tensor(vec![1.0; 16], &[1, 1, 4, 4]);

    // Stride of 0 should error
    let result = max_pool2d(&input, (2, 2), Some((0, 0)), (0, 0));
    assert!(result.is_err());
}

// ============================================================================
// Output Size Calculation Tests
// ============================================================================

#[test]
fn test_conv1d_output_size() {
    assert_eq!(conv1d_output_size(10, 3, 1, 0), 8);
    assert_eq!(conv1d_output_size(10, 3, 2, 0), 4);
    assert_eq!(conv1d_output_size(10, 3, 1, 1), 10);
}

#[test]
fn test_conv2d_output_size() {
    assert_eq!(conv2d_output_size(10, 10, 3, 3, 1, 1, 0, 0), (8, 8));
    assert_eq!(conv2d_output_size(10, 10, 3, 3, 2, 2, 0, 0), (4, 4));
    assert_eq!(conv2d_output_size(10, 10, 3, 3, 1, 1, 1, 1), (10, 10));
}

#[test]
fn test_conv3d_output_size() {
    assert_eq!(
        conv3d_output_size((10, 10, 10), (3, 3, 3), (1, 1, 1), (0, 0, 0)),
        (8, 8, 8)
    );
    assert_eq!(
        conv3d_output_size((10, 10, 10), (3, 3, 3), (2, 2, 2), (0, 0, 0)),
        (4, 4, 4)
    );
    assert_eq!(
        conv3d_output_size((10, 10, 10), (3, 3, 3), (1, 1, 1), (1, 1, 1)),
        (10, 10, 10)
    );
}
