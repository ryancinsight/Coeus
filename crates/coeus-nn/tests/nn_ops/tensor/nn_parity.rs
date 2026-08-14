#[path = "nn_parity/attention.rs"]
mod attention;
#[path = "nn_parity/convolution.rs"]
mod convolution;
#[path = "nn_parity/embedding.rs"]
mod embedding;
#[path = "nn_parity/linear_norm.rs"]
mod linear_norm;
#[path = "nn_parity/losses.rs"]
mod losses;
#[path = "nn_parity/regularization.rs"]
mod regularization;

use coeus_tensor::Tensor as CoeusTensor;

fn assert_tensor_eq_data<B: coeus_core::ComputeBackend>(
    coeus: &CoeusTensor<f32, B>,
    expected: &[f32],
    tol: f32,
) where
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    let coeus_slice = coeus.as_slice();
    assert_eq!(coeus_slice.len(), expected.len());
    for (i, (&c, &b)) in coeus_slice.iter().zip(expected.iter()).enumerate() {
        let diff = (c - b).abs();
        assert!(
            diff < tol,
            "Mismatch at index {i}: coeus = {c}, expected = {b} (diff = {diff}, tolerance = {tol})"
        );
    }
}
