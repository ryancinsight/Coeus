#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
use coeus_autograd::{embedding, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn test_embedding_autograd() {
    let backend = MoiraiBackend::new();

    // Weight matrix of shape [3, 2]
    let w_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_tensor = Tensor::from_slice_on(vec![3, 2], &w_data, &backend);
    let weight = Var::new(w_tensor, true);

    // Indices of shape [2, 2]
    let idx_data = vec![0i32, 2, 1, 0];
    let indices = Tensor::from_slice_on(vec![2, 2], &idx_data, &backend);

    let y = embedding(&weight, &indices);
    assert_eq!(y.tensor.shape(), &[2, 2, 2]);
    let y_slice = y.tensor.as_slice();
    assert_eq!(y_slice, &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);

    let grad_out_data = vec![1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
    let grad_out = Tensor::from_slice_on(vec![2, 2, 2], &grad_out_data, &backend);
    y.backward_with_seed(grad_out)
        .expect("invariant: valid autograd fixture completes backward");

    let gw = weight.grad().unwrap();
    let gw_slice = gw.as_slice();
    assert_eq!(gw_slice, &[5.0, 5.0, 3.0, 3.0, 2.0, 2.0]);
}
