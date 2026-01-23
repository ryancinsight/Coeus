//! Basic RNN unit tests.

use super::RNN;
use crate::core::module::Module;
use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

#[test]
fn test_rnn_creation() {
    let rnn = RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        10, 20, 1, true, false, false,
    )
    .unwrap();
    assert_eq!(rnn.input_size, 10);
    assert_eq!(rnn.hidden_size, 20);
}

#[test]
fn test_rnn_forward_shape() {
    let rnn = RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        10, 20, 1, true, false, false,
    )
    .unwrap();
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[5, 3, 10]).unwrap();
    let output = rnn.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[5, 3, 20]);
}
