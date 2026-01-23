use super::core::Conv1D;
use super::transpose::ConvTranspose1d;
use crate::core::module::Module;
use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

#[test]
fn test_conv1d_creation() {
    let conv = Conv1D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        3,
        64,
        5,
        Some(1),
        Some(2),
        Some(true),
    )
    .unwrap();

    assert_eq!(conv.in_channels, 3);
    let weight_shape = conv.weight().data().shape().dims();
    assert_eq!(weight_shape[0], 64); // out_channels
    assert_eq!(weight_shape[2], 5); // kernel_size
    assert_eq!(conv.stride, 1);
    assert_eq!(conv.padding, 2);
    assert!(conv.bias().is_some());

    let params = conv.parameters();
    assert_eq!(params.len(), 2); // weight + bias
    assert_eq!(params[0].name(), "weight");
    assert_eq!(params[1].name(), "bias");
}

#[test]
fn test_conv1d_forward() {
    let conv = Conv1D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        1,
        2,
        3,
        Some(1),
        Some(1),
        Some(false),
    )
    .unwrap();

    // Input: [batch_size=1, channels=1, length=5]
    let input_data = vec![
        Float32::new(1.0),
        Float32::new(2.0),
        Float32::new(3.0),
        Float32::new(4.0),
        Float32::new(5.0),
    ];
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 1, 5],
    )
    .unwrap();

    let output = conv.forward(&input).unwrap();
    let output_shape = output.shape().dims();

    // Expected: [batch_size=1, channels=2, length=5] (with stride=1, padding=1, kernel=3)
    // Output length = (5 + 2*1 - 3) / 1 + 1 = 5
    assert_eq!(output_shape, &[1, 2, 5]);
}

#[test]
fn test_conv_transpose_1d_creation() {
    let conv = ConvTranspose1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        64,
        1,
        4,
        Some(2),
        Some(1),
        Some(0),
        Some(true),
    )
    .unwrap();

    assert_eq!(conv.in_channels, 64);
    let weight_shape = conv.weight().data().shape().dims();
    assert_eq!(weight_shape[1], 1); // out_channels
    assert_eq!(conv.kernel_size, 4);
    assert_eq!(conv.stride, 2);
    assert_eq!(conv.padding, 1);
    assert_eq!(conv.output_padding, 0);
    assert!(conv.bias().is_some());
}

#[test]
fn test_conv_transpose_1d_forward() {
    let conv = ConvTranspose1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        2,
        3,
        3,
        Some(2),
        Some(1),
        Some(0),
        Some(false),
    )
    .unwrap();

    // Input: [batch_size=1, channels=2, length=4]
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
        &[1, 2, 4],
    )
    .unwrap();

    let output = conv.forward(&input).unwrap();
    let output_shape = output.shape().dims();

    // Expected output length: (4 - 1) * 2 - 2 * 1 + 3 + 0 = 7
    assert_eq!(output_shape, &[1, 3, 7]);
}

#[test]
fn test_conv_transpose_1d_output_size() {
    let conv = ConvTranspose1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        1,
        1,
        4,
        Some(2),
        Some(1),
        Some(0),
        Some(true),
    )
    .unwrap();
    assert_eq!(conv.output_size(100), 200); // (100 - 1) * 2 - 2 * 1 + 4 + 0 = 200
}
