//! LSTM unit tests.

use super::LSTM;
use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

#[test]
fn test_lstm_new() {
    let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        10, 20, 1, true, false, false,
    )
    .unwrap();
    assert_eq!(lstm.input_size, 10);
    assert_eq!(lstm.hidden_size, 20);
}

#[test]
fn test_lstm_forward() {
    let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        10, 20, 1, true, false, false,
    )
    .unwrap();
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[5, 3, 10]).unwrap();
    let (output, (h_n, c_n)) = lstm
        .forward(
            &input,
            None::<(
                &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
                &Tensor<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
            )>,
        )
        .unwrap();
    assert_eq!(output.shape().dims(), &[5, 3, 20]);
    assert_eq!(h_n.shape().dims(), &[1, 3, 20]);
    assert_eq!(c_n.shape().dims(), &[1, 3, 20]);
}

#[test]
fn test_lstm_bidirectional() {
    let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        10, 20, 2, true, false, true,
    )
    .unwrap();
    assert!(lstm.bidirectional);
}

#[test]
fn test_reverse_sequence() {
    let input_data = (1..=18).map(|x| Float32::new(x as f32)).collect::<Vec<_>>();
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        input_data,
        &[3, 2, 3],
    )
    .unwrap();
    let reversed = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::reverse_sequence(
        &input, 3, 2, 3,
    )
    .unwrap();
    let reversed_data = reversed.as_slice().to_vec();
    assert_eq!(reversed_data[0], Float32::new(13.0));
}
