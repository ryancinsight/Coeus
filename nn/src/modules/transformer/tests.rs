//! Transformer unit tests.

use super::TransformerDecoder;
use super::TransformerEncoder;
use crate::core::module::Module;
use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

#[test]
fn test_transformer_encoder_creation() {
    let encoder = TransformerEncoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        512, 8, 2048, 0.1,
    )
    .unwrap();
    assert_eq!(encoder.d_model, 512);
}

#[test]
fn test_transformer_encoder_forward_shape() {
    let encoder = TransformerEncoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        512, 8, 2048, 0.1,
    )
    .unwrap();
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[1, 10, 512])
        .unwrap();
    let output = encoder.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 10, 512]);
}

#[test]
fn test_transformer_decoder_creation() {
    let decoder = TransformerDecoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        512, 8, 2048, 0.1,
    )
    .unwrap();
    assert_eq!(decoder.d_model, 512);
}

#[test]
fn test_transformer_decoder_forward_shape() {
    let decoder = TransformerDecoder::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
        512, 8, 2048, 0.1,
    )
    .unwrap();
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::zeros(&[1, 10, 512])
        .unwrap();
    let output = decoder.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 10, 512]);
}
