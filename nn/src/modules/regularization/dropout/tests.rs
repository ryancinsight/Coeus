use super::spatial::spatial2d::Dropout2d;
use super::standard::Dropout;
use crate::core::module::Module;
use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use tensor::Tensor;

#[test]
fn test_dropout_eval_mode() {
    let mut dropout = Dropout::new(0.5);
    dropout.train(false);
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[4]).unwrap();
    let output = dropout.forward(&input).unwrap();
    assert_eq!(input.as_slice(), output.as_slice());
}

#[test]
fn test_dropout_training_mode() {
    let mut dropout = Dropout::new(0.5);
    dropout.train(true);
    let input =
        Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[100]).unwrap();
    let output = dropout.forward(&input).unwrap();
    let zeros = output.as_slice().iter().filter(|&x| x.get() == 0.0).count();
    assert!((30..=70).contains(&zeros));
}

#[test]
fn test_dropout2d_training() {
    let mut dropout = Dropout2d::new(0.5);
    dropout.train(true);
    let input = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::ones(&[1, 10, 8, 8])
        .unwrap();
    let output = dropout.forward(&input).unwrap();
    assert_eq!(output.shape().dims(), &[1, 10, 8, 8]);
}
