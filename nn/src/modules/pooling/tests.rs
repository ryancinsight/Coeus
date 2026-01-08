use super::*;
use backend::CpuBackend;
use dtype::float::Float32;
use dtype::traits::FloatExt;
use num_traits::ToPrimitive;
use storage::DenseStorage;
use tensor::Tensor;

use crate::core::module::Module; // Adjust import path

type TestBackend = CpuBackend<Float32>;
type TestStorage = DenseStorage<Float32>;
type TestDataType = Float32;

#[test]
fn test_maxpool2d_constructor() {
    let pool = MaxPool2d::new((2, 2), Some((2, 2)), (0, 0));
    assert_eq!(pool.kernel_size, (2, 2));
    assert_eq!(pool.stride, Some((2, 2)));
    assert_eq!(pool.padding, (0, 0));
}

#[test]
fn test_maxpool2d_forward_shape() {
    let pool = MaxPool2d::new((2, 2), Some((2, 2)), (0, 0));

    // Input: [batch_size=2, channels=3, height=4, width=4]
    let input_data: Vec<Float32> = (0..96).map(|i| Float32::new(i as f32)).collect();
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[2, 3, 4, 4],
    )
    .unwrap();

    let output =
        <MaxPool2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();

    // Output shape should be [2, 3, 2, 2] (downsampled by 2x)
    assert_eq!(output.shape().dims(), &[2, 3, 2, 2]);
}

#[test]
fn test_maxpool2d_forward_correctness() {
    let pool = MaxPool2d::new((2, 2), Some((2, 2)), (0, 0));

    // Input: [1, 1, 4, 4] with known values
    let input_data: Vec<Float32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]
    .iter()
    .map(|&x| Float32::new(x))
    .collect();
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 1, 4, 4],
    )
    .unwrap();

    let output =
        <MaxPool2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();
    let output_data = output.as_slice();

    // Expected output: max of each 2x2 window
    // Top-left: max(1,2,5,6) = 6
    // Top-right: max(3,4,7,8) = 8
    // Bottom-left: max(9,10,13,14) = 14
    // Bottom-right: max(11,12,15,16) = 16
    assert_eq!(output_data[0].to_f64().unwrap(), 6.0);
    assert_eq!(output_data[1].to_f64().unwrap(), 8.0);
    assert_eq!(output_data[2].to_f64().unwrap(), 14.0);
    assert_eq!(output_data[3].to_f64().unwrap(), 16.0);
}

#[test]
fn test_maxpool2d_stride_default() {
    // When stride is None, it should default to kernel_size
    let pool = MaxPool2d::new((2, 2), None, (0, 0));

    let input_data: Vec<Float32> = (0..16).map(|i| Float32::new(i as f32)).collect();
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 1, 4, 4],
    )
    .unwrap();

    let output =
        <MaxPool2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();

    // Output shape should be [1, 1, 2, 2] (stride defaults to kernel_size)
    assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
}

#[test]
#[should_panic(expected = "kernel_size must be > 0")]
fn test_maxpool2d_invalid_kernel_size() {
    let _pool = MaxPool2d::new((0, 2), Some((2, 2)), (0, 0));
}

#[test]
fn test_avgpool2d_constructor() {
    let pool = AvgPool2d::new((2, 2), Some((2, 2)), (0, 0));
    assert_eq!(pool.kernel_size, (2, 2));
    assert_eq!(pool.stride, Some((2, 2)));
    assert_eq!(pool.padding, (0, 0));
}

#[test]
fn test_avgpool2d_forward_shape() {
    let pool = AvgPool2d::new((2, 2), Some((2, 2)), (0, 0));

    // Input: [batch_size=2, channels=3, height=4, width=4]
    let input_data: Vec<Float32> = (0..96).map(|i| Float32::new(i as f32)).collect();
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[2, 3, 4, 4],
    )
    .unwrap();

    let output =
        <AvgPool2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();

    // Output shape should be [2, 3, 2, 2] (downsampled by 2x)
    assert_eq!(output.shape().dims(), &[2, 3, 2, 2]);
}

#[test]
fn test_avgpool2d_forward_correctness() {
    let pool = AvgPool2d::new((2, 2), Some((2, 2)), (0, 0));

    // Input: [1, 1, 4, 4] with known values
    let input_data: Vec<Float32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]
    .iter()
    .map(|&x| Float32::new(x))
    .collect();
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 1, 4, 4],
    )
    .unwrap();

    let output =
        <AvgPool2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();
    let output_data = output.as_slice();

    // Expected output: average of each 2x2 window
    // Top-left: avg(1,2,5,6) = 3.5
    // Top-right: avg(3,4,7,8) = 5.5
    // Bottom-left: avg(9,10,13,14) = 11.5
    // Bottom-right: avg(11,12,15,16) = 13.5
    assert!((output_data[0].to_f64().unwrap() - 3.5).abs() < 1e-6);
    assert!((output_data[1].to_f64().unwrap() - 5.5).abs() < 1e-6);
    assert!((output_data[2].to_f64().unwrap() - 11.5).abs() < 1e-6);
    assert!((output_data[3].to_f64().unwrap() - 13.5).abs() < 1e-6);
}

#[test]
fn test_avgpool2d_stride_default() {
    // When stride is None, it should default to kernel_size
    let pool = AvgPool2d::new((2, 2), None, (0, 0));

    let input_data: Vec<Float32> = (0..16).map(|i| Float32::new(i as f32)).collect();
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 1, 4, 4],
    )
    .unwrap();

    let output =
        <AvgPool2d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();

    // Output shape should be [1, 1, 2, 2] (stride defaults to kernel_size)
    assert_eq!(output.shape().dims(), &[1, 1, 2, 2]);
}

#[test]
#[should_panic(expected = "kernel_size must be > 0")]
fn test_avgpool2d_invalid_kernel_size() {
    let _pool = AvgPool2d::new((0, 2), Some((2, 2)), (0, 0));
}

#[test]
fn test_adaptive_avgpool2d_constructor() {
    let pool: AdaptiveAvgPool2d = AdaptiveAvgPool2d::new((7, 7));
    assert_eq!(pool.output_size, (7, 7));
}

#[test]
fn test_adaptive_avgpool2d_forward_shape() {
    let pool = AdaptiveAvgPool2d::new((7, 7));
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 3, 14, 14])
            .unwrap();
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[2, 3, 7, 7]);
}

#[test]
fn test_adaptive_avgpool2d_global_pooling() {
    let pool = AdaptiveAvgPool2d::new((1, 1));
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(6.0),
            Float32::new(7.0),
            Float32::new(8.0),
            Float32::new(9.0),
        ],
        &[1, 1, 3, 3],
    )
    .unwrap();

    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 1, 1, 1]);

    // Global average should be (1+2+3+4+5+6+7+8+9)/9 = 5.0
    let expected = 5.0;
    assert!((output.as_slice()[0].get() - expected).abs() < 1e-5);
}

#[test]
fn test_adaptive_avgpool2d_3d_input() {
    let pool = AdaptiveAvgPool2d::new((2, 2));
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[3, 4, 4]).unwrap();
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[3, 2, 2]);
}

#[test]
fn test_adaptive_maxpool2d_constructor() {
    let pool: AdaptiveMaxPool2d = AdaptiveMaxPool2d::new((7, 7));
    assert_eq!(pool.output_size, (7, 7));
}

#[test]
fn test_adaptive_maxpool2d_forward_shape() {
    let pool = AdaptiveMaxPool2d::new((7, 7));
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 3, 14, 14])
            .unwrap();
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[2, 3, 7, 7]);
}

#[test]
fn test_adaptive_maxpool2d_global_pooling() {
    let pool = AdaptiveMaxPool2d::new((1, 1));
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
            Float32::new(5.0),
            Float32::new(9.0),
            Float32::new(7.0),
            Float32::new(8.0),
            Float32::new(6.0),
        ],
        &[1, 1, 3, 3],
    )
    .unwrap();

    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 1, 1, 1]);

    // Global max should be 9.0
    let expected = 9.0;
    assert!((output.as_slice()[0].get() - expected).abs() < 1e-5);
}

#[test]
fn test_adaptive_maxpool2d_3d_input() {
    let pool = AdaptiveMaxPool2d::new((2, 2));
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[3, 4, 4]).unwrap();
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[3, 2, 2]);
}

#[test]
fn test_maxpool1d_creation() {
    let pool = MaxPool1d::new(2, Some(2), 0);
    assert_eq!(pool.kernel_size, 2);
}

#[test]
fn test_maxpool1d_forward_basic() {
    let pool = MaxPool1d::new(2, Some(2), 0);
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 64, 50]);
}

#[test]
fn test_maxpool1d_forward_with_stride() {
    let pool = MaxPool1d::new(3, Some(2), 0);
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 32, 100]).unwrap();
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 32, 49]);
}

#[test]
fn test_maxpool1d_forward_computation() {
    let pool = MaxPool1d::new(2, Some(2), 0);
    let input_data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
    ];
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 1, 4],
    )
    .unwrap();
    let output = pool.forward(&input).unwrap();

    // Expected: max([1, 2]) = 2, max([3, 4]) = 4
    assert_eq!(output.shape().dims(), &[1, 1, 2]);
    assert_eq!(output.as_slice()[0].get(), 2.0);
    assert_eq!(output.as_slice()[1].get(), 4.0);
}

#[test]
fn test_avgpool1d_creation() {
    let pool = AvgPool1d::new(2, Some(2), 0);
    assert_eq!(pool.kernel_size, 2);
}

#[test]
fn test_avgpool1d_forward_basic() {
    let pool = AvgPool1d::new(2, Some(2), 0);
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 64, 50]);
}

#[test]
fn test_avgpool1d_forward_computation() {
    let pool = AvgPool1d::new(2, Some(2), 0);
    let input_data = vec![
        Float32::new(1.0),
        Float32::new(3.0),
        Float32::new(2.0),
        Float32::new(4.0),
    ];
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 1, 4],
    )
    .unwrap();
    let output = pool.forward(&input).unwrap();

    // Expected: avg([1, 3]) = 2, avg([2, 4]) = 3
    assert_eq!(output.shape().dims(), &[1, 1, 2]);
    assert_eq!(output.as_slice()[0].get(), 2.0);
    assert_eq!(output.as_slice()[1].get(), 3.0);
}

#[test]
fn test_adaptive_avgpool1d_creation() {
    let pool = AdaptiveAvgPool1d::new(10);
    assert_eq!(pool.output_size, 10);
}

#[test]
fn test_adaptive_avgpool1d_forward_basic() {
    let pool = AdaptiveAvgPool1d::new(10);
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 100]).unwrap();
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 64, 10]);
}

#[test]
fn test_adaptive_avgpool1d_forward_upsampling() {
    let pool = AdaptiveAvgPool1d::new(20);
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 32, 10]).unwrap();
    let output = pool.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 32, 20]);
}

#[test]
fn test_adaptive_avgpool1d_forward_computation() {
    let pool = AdaptiveAvgPool1d::new(2);
    let input_data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
    ];
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 1, 4],
    )
    .unwrap();
    let output = pool.forward(&input).unwrap();

    // Expected: avg([1, 2]) = 1.5, avg([3, 4]) = 3.5
    assert_eq!(output.shape().dims(), &[1, 1, 2]);
    assert_eq!(output.as_slice()[0].get(), 1.5);
    assert_eq!(output.as_slice()[1].get(), 3.5);
}

#[test]
fn test_maxpool3d_constructor() {
    let pool = MaxPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));
    assert_eq!(pool.kernel_size, (2, 2, 2));
    assert_eq!(pool.stride, Some((2, 2, 2)));
    assert_eq!(pool.padding, (0, 0, 0));
}

#[test]
fn test_maxpool3d_forward_shape() {
    let pool = MaxPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

    // Input: [batch_size=1, channels=3, depth=4, height=4, width=4]
    let input_data: Vec<Float32> = (0..192).map(|i| Float32::new(i as f32)).collect();
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 3, 4, 4, 4],
    )
    .unwrap();

    let output =
        <MaxPool3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();

    // Output shape should be [1, 3, 2, 2, 2] (downsampled by 2x in all dimensions)
    assert_eq!(output.shape().dims(), &[1, 3, 2, 2, 2]);
}

#[test]
fn test_maxpool3d_forward_computation() {
    let pool = MaxPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

    // Simple 2x2x2 input with known values
    let input_data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
        Float32::new(5.0),
        Float32::new(6.0),
        Float32::new(7.0),
        Float32::new(8.0),
    ];
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 1, 2, 2, 2],
    )
    .unwrap();
    let output =
        <MaxPool3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();

    // Expected: max of all 8 values = 8.0
    assert_eq!(output.shape().dims(), &[1, 1, 1, 1, 1]);
    assert_eq!(output.as_slice()[0].get(), 8.0);
}

#[test]
fn test_maxpool3d_with_stride() {
    let pool = MaxPool3d::new((2, 2, 2), Some((1, 1, 1)), (0, 0, 0));

    // Input: [1, 1, 3, 3, 3]
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 1, 3, 3, 3])
            .unwrap();
    let output =
        <MaxPool3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();

    // Output: (3 - 2) / 1 + 1 = 2
    assert_eq!(output.shape().dims(), &[1, 1, 2, 2, 2]);
}

#[test]
fn test_maxpool3d_batch_processing() {
    let pool = MaxPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

    // Input: [batch_size=4, channels=2, depth=4, height=4, width=4]
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[4, 2, 4, 4, 4])
            .unwrap();
    let output =
        <MaxPool3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();

    // Output shape should be [4, 2, 2, 2, 2]
    assert_eq!(output.shape().dims(), &[4, 2, 2, 2, 2]);
}

#[test]
fn test_maxpool3d_video_classification() {
    let pool = MaxPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

    // Video input: [1, 64, 16, 112, 112] (16 frames, 64 channels, 112x112 resolution)
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 16, 112, 112])
            .unwrap();
    let output =
        <MaxPool3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();

    // Output: [1, 64, 8, 56, 56] (downsampled by 2x)
    assert_eq!(output.shape().dims(), &[1, 64, 8, 56, 56]);
}

#[test]
fn test_avgpool3d_constructor() {
    let pool = AvgPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));
    assert_eq!(pool.kernel_size, (2, 2, 2));
    assert_eq!(pool.stride, Some((2, 2, 2)));
    assert_eq!(pool.padding, (0, 0, 0));
}

#[test]
fn test_avgpool3d_forward_shape() {
    let pool = AvgPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

    // Input: [batch_size=1, channels=3, depth=4, height=4, width=4]
    let input_data: Vec<Float32> = (0..192).map(|i| Float32::new(i as f32)).collect();
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 3, 4, 4, 4],
    )
    .unwrap();

    let output =
        <AvgPool3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();

    // Output shape should be [1, 3, 2, 2, 2] (downsampled by 2x in all dimensions)
    assert_eq!(output.shape().dims(), &[1, 3, 2, 2, 2]);
}

#[test]
fn test_avgpool3d_forward_computation() {
    let pool = AvgPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

    // Simple 2x2x2 input with known values
    let input_data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
        Float32::new(5.0),
        Float32::new(6.0),
        Float32::new(7.0),
        Float32::new(8.0),
    ];
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 1, 2, 2, 2],
    )
    .unwrap();
    let output =
        <AvgPool3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();

    // Expected: avg of all 8 values = (1+2+3+4+5+6+7+8)/8 = 36/8 = 4.5
    assert_eq!(output.shape().dims(), &[1, 1, 1, 1, 1]);
    assert_eq!(output.as_slice()[0].get(), 4.5);
}

#[test]
fn test_avgpool3d_with_stride() {
    let pool = AvgPool3d::new((2, 2, 2), Some((1, 1, 1)), (0, 0, 0));

    // Input: [1, 1, 3, 3, 3]
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 1, 3, 3, 3])
            .unwrap();
    let output =
        <AvgPool3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();

    // Output: (3 - 2) / 1 + 1 = 2
    assert_eq!(output.shape().dims(), &[1, 1, 2, 2, 2]);
    // All values should be 1.0 (average of 1.0s)
    assert_eq!(output.as_slice()[0].get(), 1.0);
}

#[test]
fn test_avgpool3d_batch_processing() {
    let pool = AvgPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

    // Input: [batch_size=4, channels=2, depth=4, height=4, width=4]
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[4, 2, 4, 4, 4])
            .unwrap();
    let output =
        <AvgPool3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();

    // Output shape should be [4, 2, 2, 2, 2]
    assert_eq!(output.shape().dims(), &[4, 2, 2, 2, 2]);
}

#[test]
fn test_avgpool3d_video_classification() {
    let pool = AvgPool3d::new((2, 2, 2), Some((2, 2, 2)), (0, 0, 0));

    // Video input: [1, 64, 16, 112, 112] (16 frames, 64 channels, 112x112 resolution)
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 64, 16, 112, 112])
            .unwrap();
    let output =
        <AvgPool3d as Module<CpuBackend<Float32>, DenseStorage<Float32>, Float32>>::forward(
            &pool, &input,
        )
        .unwrap();

    // Output: [1, 64, 8, 56, 56] (downsampled by 2x)
    assert_eq!(output.shape().dims(), &[1, 64, 8, 56, 56]);
}
