use burn::backend::ndarray::NdArray;
use burn::tensor::{Tensor as BurnTensor, TensorData};
use coeus_autograd::Var;
use coeus_core::SequentialBackend;
use coeus_nn::{cross_entropy_loss, softmax};
use coeus_tensor::Tensor as CoeusTensor;

type BurnBackend = NdArray<f32>;

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch: actual = {}, expected = {}",
        actual.len(),
        expected.len()
    );

    for (index, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
        let tolerance = 512.0 * f32::EPSILON * (1.0 + want.abs());
        let diff = (got - want).abs();
        assert!(
            diff <= tolerance,
            "{label}[{index}]: actual = {got}, expected = {want}, diff = {diff}, tolerance = {tolerance}"
        );
    }
}

#[test]
fn softmax_matches_burn_ndarray_reference() {
    let logits = [1.5_f32, 0.5, -0.5, -1.0, 2.0, 0.0];
    let coeus_logits = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &logits),
        false,
    );

    let coeus = softmax(&coeus_logits, 1);

    let device = Default::default();
    let burn_logits =
        BurnTensor::<BurnBackend, 2>::from_data(TensorData::new(logits.to_vec(), [2, 3]), &device);
    let burn = burn::tensor::activation::softmax(burn_logits, 1);
    let burn_values = burn.into_data().to_vec::<f32>().unwrap();

    assert_close("softmax", coeus.tensor.as_slice(), &burn_values);
}

#[test]
fn cross_entropy_loss_matches_burn_ndarray_reference() {
    let logits = [1.5_f32, 0.5, -0.5, -1.0, 2.0, 0.0];
    let targets = [0_usize, 1_usize];

    let coeus_logits = Var::new(
        CoeusTensor::<f32, SequentialBackend>::from_slice(vec![2, 3], &logits),
        true,
    );
    let coeus = cross_entropy_loss(&coeus_logits, &targets);

    let device = Default::default();
    let burn_logits =
        BurnTensor::<BurnBackend, 2>::from_data(TensorData::new(logits.to_vec(), [2, 3]), &device);
    let burn_softmax = burn::tensor::activation::softmax(burn_logits, 1);
    let burn_values = burn_softmax.into_data().to_vec::<f32>().unwrap();
    let burn_loss = targets
        .iter()
        .enumerate()
        .map(|(row, &target)| -burn_values[row * 3 + target].ln())
        .sum::<f32>()
        / targets.len() as f32;

    assert_close("cross_entropy_loss", coeus.tensor.as_slice(), &[burn_loss]);
}
