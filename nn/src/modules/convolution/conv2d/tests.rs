use super::core::Conv2D;
use super::transpose::ConvTranspose2d;
use crate::core::module::Module;
use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

type TestTensor = Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>;

#[test]
fn test_conv2d_creation() {
    let conv = Conv2D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        3,
        64,
        (3, 3),
        Some((1, 1)),
        Some((1, 1)),
        Some(true),
    )
    .unwrap();
    assert_eq!(conv.in_channels, 3);
    assert_eq!(conv.out_channels, 64);
    assert_eq!(conv.kernel_height, 3);
    assert_eq!(conv.kernel_width, 3);
    assert_eq!(conv.stride_h, 1);
    assert_eq!(conv.stride_w, 1);
    assert_eq!(conv.padding_h, 1);
    assert_eq!(conv.padding_w, 1);
    assert!(conv.bias().is_some());
    let params = conv.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_conv2d_forward() {
    let conv = Conv2D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        1,
        2,
        (3, 3),
        Some((1, 1)),
        Some((1, 1)),
        Some(false),
    )
    .unwrap();
    let input_data = vec![Float32::new(1.0); 25];
    let input = TestTensor::from_vec(input_data, &[1, 1, 5, 5]).unwrap();
    let output = conv.forward(&input).unwrap();
    let output_shape = output.shape().dims();
    assert_eq!(output_shape, &[1, 2, 5, 5]);
}

#[test]
fn test_conv2d_output_size() {
    let conv = Conv2D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        3,
        64,
        (3, 3),
        Some((1, 1)),
        Some((1, 1)),
        Some(true),
    )
    .unwrap();
    assert_eq!(conv.output_size(32, 32), (32, 32));
    let conv2 = Conv2D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        3,
        64,
        (3, 3),
        Some((2, 2)),
        Some((0, 0)),
        Some(true),
    )
    .unwrap();
    assert_eq!(conv2.output_size(28, 28), (13, 13));
}

#[test]
fn test_conv2d_backward_basic() {
    let conv = Conv2D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        1,
        1,
        (3, 3),
        None,
        None,
        Some(true),
    )
    .unwrap();
    let input = TestTensor::from_vec(vec![Float32::new(1.0); 25], &[1, 1, 5, 5]).unwrap();
    let grad_output = TestTensor::from_vec(vec![Float32::new(1.0); 9], &[1, 1, 3, 3]).unwrap();
    let (input_grad, weight_grad, bias_grad) = conv.backward(&grad_output, &input).unwrap();
    assert_eq!(input_grad.shape().dims(), &[1, 1, 5, 5]);
    assert_eq!(weight_grad.shape().dims(), &[1, 1, 3, 3]);
    assert_eq!(bias_grad.as_ref().unwrap().shape().dims(), &[1]);
}

#[test]
fn test_conv2d_backward_no_bias() {
    let conv = Conv2D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        2,
        3,
        (2, 2),
        Some((1, 1)),
        Some((1, 1)),
        Some(false),
    )
    .unwrap();
    let input = TestTensor::from_vec(vec![Float32::new(0.5); 64], &[2, 2, 4, 4]).unwrap();
    // With padding=1, kernel=2, stride=1, output size = (4 + 2*1 - 2)/1 + 1 = 5
    let grad_output = TestTensor::from_vec(vec![Float32::new(1.0); 150], &[2, 3, 5, 5]).unwrap();
    let (input_grad, weight_grad, bias_grad) = conv.backward(&grad_output, &input).unwrap();
    assert_eq!(input_grad.shape().dims(), &[2, 2, 4, 4]);
    assert_eq!(weight_grad.shape().dims(), &[3, 2, 2, 2]);
    assert!(bias_grad.is_none());
}

#[test]
fn test_conv_transpose_2d_creation() {
    let conv = ConvTranspose2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        64,
        32,
        (4, 4),
        Some((2, 2)),
        Some((1, 1)),
        Some((0, 0)),
        Some(true),
    )
    .unwrap();

    assert_eq!(conv.in_channels, 64);
    assert_eq!(conv.out_channels, 32);
    assert_eq!(conv.kernel_height, 4);
    assert_eq!(conv.kernel_width, 4);
    assert_eq!(conv.stride_h, 2);
    assert_eq!(conv.stride_w, 2);
    assert_eq!(conv.padding_h, 1);
    assert_eq!(conv.padding_w, 1);
    assert!(conv.bias().is_some());

    let params = conv.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_conv_transpose_2d_output_size() {
    let conv = ConvTranspose2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        64,
        32,
        (4, 4),
        Some((2, 2)),
        Some((1, 1)),
        Some((0, 0)),
        Some(true),
    )
    .unwrap();

    // (8 - 1) * 2 - 2 * 1 + 4 + 0 = 7 * 2 - 2 + 4 = 14 - 2 + 4 = 16
    assert_eq!(conv.output_size(8, 8), (16, 16));
}

#[test]
fn test_conv_transpose_2d_forward() {
    let conv = ConvTranspose2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        2,
        3,
        (3, 3),
        Some((2, 2)),
        Some((1, 1)),
        Some((0, 0)),
        Some(false),
    )
    .unwrap();

    // Input: [batch=1, channels=2, height=4, width=4]
    let input_data = vec![Float32::new(1.0); 32];
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[1, 2, 4, 4],
    )
    .unwrap();

    let output = conv.forward(&input).unwrap();
    let output_shape = output.shape().dims();

    // Expected output: (4 - 1) * 2 - 2 * 1 + 3 + 0 = 6 - 2 + 3 = 7
    assert_eq!(output_shape, &[1, 3, 7, 7]);
}
