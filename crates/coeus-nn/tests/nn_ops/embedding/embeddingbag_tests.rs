use coeus_nn::{EmbeddingBag, EmbeddingBagMode};
use coeus_tensor::Tensor;

fn seeded_embedding_bag(mode: EmbeddingBagMode) -> EmbeddingBag<f64> {
    let mut bag = EmbeddingBag::<f64>::new(4, 2, mode);
    bag.weight.tensor = Tensor::from_slice(
        vec![4, 2],
        &[
            1.0, 2.0, // row 0
            3.0, 4.0, // row 1
            5.0, 6.0, // row 2
            7.0, 8.0, // row 3
        ],
    );
    bag
}

#[test]
fn embeddingbag_sum_with_offsets_matches_expected() {
    let bag = seeded_embedding_bag(EmbeddingBagMode::Sum);
    let out = bag.forward_with_offsets(&[0, 1, 2, 3], Some(&[0, 2]));
    assert_eq!(out.tensor.shape(), &[2, 2]);
    assert_eq!(out.tensor.as_slice(), &[4.0, 6.0, 12.0, 14.0]);
}

#[test]
fn embeddingbag_mean_with_offsets_matches_expected() {
    let bag = seeded_embedding_bag(EmbeddingBagMode::Mean);
    let out = bag.forward_with_offsets(&[1, 3], Some(&[0]));
    assert_eq!(out.tensor.shape(), &[1, 2]);
    assert_eq!(out.tensor.as_slice(), &[5.0, 6.0]);
}

#[test]
fn embeddingbag_max_with_offsets_matches_expected() {
    let bag = seeded_embedding_bag(EmbeddingBagMode::Max);
    let out = bag.forward_with_offsets(&[0, 2, 1], Some(&[0]));
    assert_eq!(out.tensor.shape(), &[1, 2]);
    assert_eq!(out.tensor.as_slice(), &[5.0, 6.0]);
}

#[test]
fn embeddingbag_empty_bag_emits_zeros() {
    let bag = seeded_embedding_bag(EmbeddingBagMode::Sum);
    let out = bag.forward_with_offsets(&[1, 2], Some(&[0, 0, 2]));
    assert_eq!(out.tensor.shape(), &[3, 2]);
    assert_eq!(out.tensor.as_slice(), &[0.0, 0.0, 8.0, 10.0, 0.0, 0.0]);
}

#[test]
fn embeddingbag_sum_backward_accumulates_weight_grads() {
    let bag = seeded_embedding_bag(EmbeddingBagMode::Sum);
    let out = bag.forward_with_offsets(&[0, 1, 1, 2], Some(&[0, 2]));
    out.backward()
        .expect("invariant: valid autograd fixture completes backward");

    let grad = bag.weight.grad().expect("weight grad must exist");
    assert_eq!(grad.shape(), &[4, 2]);
    assert_eq!(
        grad.as_slice(),
        &[
            1.0, 1.0, // row 0 appears once
            2.0, 2.0, // row 1 appears twice
            1.0, 1.0, // row 2 appears once
            0.0, 0.0, // row 3 absent
        ]
    );
}
