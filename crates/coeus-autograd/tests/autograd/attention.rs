use coeus_autograd::{sdp_attention, NullMask, Var};
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

#[test]
fn keep_mask_is_not_a_differentiable_attention_input() {
    let backend = MoiraiBackend::new();
    let constant =
        |values: &[f64]| Var::new(Tensor::from_slice_on([1, 2, 2], values, &backend), false);
    let query = constant(&[1.0, 0.0, 0.0, 1.0]);
    let key = constant(&[1.0, 0.0, 0.0, 1.0]);
    let value = constant(&[2.0, 3.0, 5.0, 7.0]);
    let mask = Var::new(Tensor::from_slice_on([1, 2], &[1.0, 0.0], &backend), true);

    let (output, _) =
        sdp_attention::<f64, MoiraiBackend, NullMask>(&query, &key, &value, Some(&mask), 1.0)
            .expect("invariant: binary keep mask is valid");

    assert_eq!(output.tensor.as_slice(), &[2.0, 3.0, 2.0, 3.0]);
    assert!(
        output.creator.is_none(),
        "a nondifferentiable mask alone must not construct an autograd node"
    );
}
