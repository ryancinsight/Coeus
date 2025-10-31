//! RNN/LSTM/GRU Tests
//!
//! Comprehensive tests for recurrent neural network functionality.

use backend::CpuBackend;
use dtype::float::Float32;
use nn::{Module, GRU, LSTM, RNN};
use storage::DenseStorage;
use tensor::Tensor;

#[test]
fn test_rnn_basic_forward() {
    let rnn = RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        4, 2, 1, true, false, false,
    )
    .unwrap();

    // Input: [seq_len=3, batch_size=1, input_size=4]
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[3, 1, 4]).unwrap();

    let output = rnn.forward(&input).unwrap();

    // Output: [seq_len=3, batch_size=1, hidden_size=2]
    assert_eq!(output.shape().dims(), &[3, 1, 2]);
}

#[test]
fn test_rnn_parameters() {
    let rnn = RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        3, 2, 1, true, false, false,
    )
    .unwrap();

    let params = rnn.parameters();

    // RNN has weight_ih and weight_hh (and bias if enabled)
    // weight_ih: [hidden_size, input_size] = [2, 3]
    // weight_hh: [hidden_size, hidden_size] = [2, 2]
    // bias_ih: [hidden_size] = [2]
    // bias_hh: [hidden_size] = [2]
    assert_eq!(params.len(), 4);

    assert_eq!(params[0].data().shape().dims(), &[2, 3]); // weight_ih
    assert_eq!(params[1].data().shape().dims(), &[2, 2]); // weight_hh
    assert_eq!(params[2].data().shape().dims(), &[2]); // bias_ih
    assert_eq!(params[3].data().shape().dims(), &[2]); // bias_hh
}

#[test]
fn test_lstm_forward() {
    let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        4, 2, 1, true, false, false,
    )
    .unwrap();

    // Input: [seq_len=2, batch_size=1, input_size=4]
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 1, 4]).unwrap();

    let output = lstm.forward(&input).unwrap();

    // Output: [seq_len=2, batch_size=1, hidden_size=2]
    assert_eq!(output.shape().dims(), &[2, 1, 2]);
}

#[test]
fn test_lstm_parameters() {
    let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        3, 2, 1, true, false, false,
    )
    .unwrap();

    let params = lstm.parameters();

    // LSTM has 4 gates, each with weight_ih and weight_hh (and biases)
    // Each weight_ih: [hidden_size, input_size] = [2, 3]
    // Each weight_hh: [hidden_size, hidden_size] = [2, 2]
    // Each bias_ih/bias_hh: [hidden_size] = [2]
    // Total: 4 gates × 4 parameters = 16 parameters
    assert_eq!(params.len(), 16);
}

#[test]
fn test_gru_forward() {
    let gru = GRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        4, 2, 1, true, false, false,
    )
    .unwrap();

    // Input: [seq_len=2, batch_size=1, input_size=4]
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 1, 4]).unwrap();

    let output = gru.forward(&input).unwrap();

    // Output: [seq_len=2, batch_size=1, hidden_size=2]
    assert_eq!(output.shape().dims(), &[2, 1, 2]);
}

#[test]
fn test_gru_parameters() {
    let gru = GRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        3, 2, 1, true, false, false,
    )
    .unwrap();

    let params = gru.parameters();

    // GRU has 3 gates (reset, update, new), each with weight_ih and weight_hh (and biases)
    // Each weight_ih: [hidden_size, input_size] = [2, 3]
    // Each weight_hh: [hidden_size, hidden_size] = [2, 2]
    // Each bias_ih/bias_hh: [hidden_size] = [2]
    // Total: 3 gates × 4 parameters = 12 parameters
    assert_eq!(params.len(), 12);
}

#[test]
fn test_rnn_bidirectional() {
    let rnn =
        RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(4, 2, 1, true, false, true)
            .unwrap();

    // Input: [seq_len=2, batch_size=1, input_size=4]
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 1, 4]).unwrap();

    let output = rnn.forward(&input).unwrap();

    // Bidirectional: output_size = hidden_size * 2 = 4
    // Output: [seq_len=2, batch_size=1, hidden_size * num_directions=4]
    assert_eq!(output.shape().dims(), &[2, 1, 4]);
}

#[test]
fn test_rnn_multiple_layers() {
    let rnn = RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        4, 2, 2, true, false, false,
    )
    .unwrap();

    // Input: [seq_len=2, batch_size=1, input_size=4]
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[2, 1, 4]).unwrap();

    let output = rnn.forward(&input).unwrap();

    // Multi-layer: output from final layer
    // Output: [seq_len=2, batch_size=1, hidden_size=2]
    assert_eq!(output.shape().dims(), &[2, 1, 2]);
}

#[test]
fn test_rnn_batch_first() {
    let rnn =
        RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(4, 2, 1, true, true, false)
            .unwrap();

    // Input: [batch_size=1, seq_len=2, input_size=4] (batch_first=True)
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 2, 4]).unwrap();

    let output = rnn.forward(&input).unwrap();

    // Output: [batch_size=1, seq_len=2, hidden_size=2] (batch_first=True)
    assert_eq!(output.shape().dims(), &[1, 2, 2]);
}

#[test]
fn test_rnn_gradient_flow() {
    let rnn = RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        2, 1, 1, true, false, false,
    )
    .unwrap();

    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
        vec![
            Float32::new(1.0),
            Float32::new(2.0),
            Float32::new(3.0),
            Float32::new(4.0),
        ],
        &[2, 1, 2], // [seq_len=2, batch_size=1, input_size=2]
    )
    .unwrap()
    .requires_grad_(true);

    let output = rnn.forward(&input).unwrap();

    // Output should require gradients
    assert!(output.requires_grad());

    // Parameters should require gradients
    let params = rnn.parameters();
    assert!(params.iter().all(|p| p.requires_grad()));
}
