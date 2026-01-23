use super::core::Conv3D;
use super::transpose::ConvTranspose3d;
use crate::core::module::Module;
use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

type TestTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

#[test]
fn test_conv3d_creation() {
    let conv3d = Conv3D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        3,
        64,
        (3, 3, 3),
        Some((1, 1, 1)),
        Some((1, 1, 1)),
        Some(true),
    )
    .unwrap();
    assert_eq!(conv3d.in_channels, 3);
    let weight_shape = conv3d.weight().data().shape().dims();
    assert_eq!(weight_shape[0], 64); // out_channels
    assert_eq!(conv3d.kernel_depth, 3);
    assert!(conv3d.bias().is_some());
}

#[test]
fn test_conv3d_forward() {
    let conv3d = Conv3D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        1,
        2,
        (3, 3, 3),
        Some((1, 1, 1)),
        Some((1, 1, 1)),
        Some(false),
    )
    .unwrap();
    let input_data = vec![Float32::new(1.0); 5 * 5 * 5];
    let input = TestTensor::from_vec(input_data, &[1, 1, 5, 5, 5]).unwrap();
    let output = conv3d.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 2, 5, 5, 5]);
}

#[test]
fn test_conv3d_output_size() {
    let conv3d = Conv3D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        3,
        64,
        (3, 3, 3),
        Some((1, 1, 1)),
        Some((1, 1, 1)),
        Some(true),
    )
    .unwrap();
    assert_eq!(conv3d.output_size(8, 8, 8), (8, 8, 8));
}

#[test]
fn test_conv_transpose_3d_creation() {
    let conv = ConvTranspose3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        64,
        32,
        (4, 4, 4),
        Some((2, 2, 2)),
        Some((1, 1, 1)),
        Some((0, 0, 0)),
        Some(true),
    )
    .unwrap();

    assert_eq!(conv.in_channels, 64);
    assert_eq!(conv.out_channels, 32);
    assert!(conv.bias().is_some());
}

#[test]
fn test_conv_transpose_3d_forward() {
    let conv = ConvTranspose3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        2,
        3,
        (3, 3, 3),
        Some((2, 2, 2)),
        Some((1, 1, 1)),
        Some((0, 0, 0)),
        Some(false),
    )
    .unwrap();

    // Input: [batch=1, channels=2, depth=4, height=4, width=4] = 128 elements
    let input_data = vec![Float32::new(1.0); 128];
    let input = TestTensor::from_vec(input_data, &[1, 2, 4, 4, 4]).unwrap();

    let output = conv.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 3, 7, 7, 7]);
}

#[test]
fn test_conv_transpose_3d_output_size() {
    let conv = ConvTranspose3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        64,
        32,
        (4, 4, 4),
        Some((2, 2, 2)),
        Some((1, 1, 1)),
        Some((0, 0, 0)),
        Some(true),
    )
    .unwrap();

    // (4 - 1) * 2 - 2 * 1 + 4 + 0 = 6 - 2 + 4 = 8
    assert_eq!(conv.output_size(4, 4, 4), (8, 8, 8));
}
