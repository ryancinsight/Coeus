use coeus_autograd::Var;
use coeus_nn::{EmbeddingBag, EmbeddingBagMode, Module, ModuleError};
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
    let out = bag
        .forward_with_offsets(&[0, 1, 2, 3], Some(&[0, 2]))
        .expect("valid EmbeddingBag indices and offsets");
    assert_eq!(out.tensor.shape(), &[2, 2]);
    assert_eq!(out.tensor.as_slice(), &[4.0, 6.0, 12.0, 14.0]);
}

#[test]
fn embeddingbag_mean_with_offsets_matches_expected() {
    let bag = seeded_embedding_bag(EmbeddingBagMode::Mean);
    let out = bag
        .forward_with_offsets(&[1, 3], Some(&[0]))
        .expect("valid EmbeddingBag indices and offsets");
    assert_eq!(out.tensor.shape(), &[1, 2]);
    assert_eq!(out.tensor.as_slice(), &[5.0, 6.0]);
}

#[test]
fn embeddingbag_max_with_offsets_matches_expected() {
    let bag = seeded_embedding_bag(EmbeddingBagMode::Max);
    let out = bag
        .forward_with_offsets(&[0, 2, 1], Some(&[0]))
        .expect("valid EmbeddingBag indices and offsets");
    assert_eq!(out.tensor.shape(), &[1, 2]);
    assert_eq!(out.tensor.as_slice(), &[5.0, 6.0]);
}

#[test]
fn embeddingbag_empty_bag_emits_zeros() {
    let bag = seeded_embedding_bag(EmbeddingBagMode::Sum);
    let out = bag
        .forward_with_offsets(&[1, 2], Some(&[0, 0, 2]))
        .expect("valid EmbeddingBag indices and offsets");
    assert_eq!(out.tensor.shape(), &[3, 2]);
    assert_eq!(out.tensor.as_slice(), &[0.0, 0.0, 8.0, 10.0, 0.0, 0.0]);
}

#[test]
fn embeddingbag_sum_backward_accumulates_weight_grads() {
    let bag = seeded_embedding_bag(EmbeddingBagMode::Sum);
    let out = bag
        .forward_with_offsets(&[0, 1, 1, 2], Some(&[0, 2]))
        .expect("valid EmbeddingBag indices and offsets");
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

#[test]
fn embeddingbag_rejects_invalid_float_indices() {
    let bag = seeded_embedding_bag(EmbeddingBagMode::Sum);
    for invalid in [f64::NAN, f64::INFINITY, -1.0, 1.5, 4.0] {
        let input = Var::new(Tensor::from_slice([1], &[invalid]), false);
        let error = bag
            .forward(&input)
            .err()
            .expect("invalid EmbeddingBag index must be rejected");
        match error {
            ModuleError::ShapeMismatch {
                module,
                parameter,
                expected,
                actual,
            } => {
                assert_eq!(module, "EmbeddingBag");
                assert_eq!(
                    parameter,
                    "indices must be finite integers within the embedding vocabulary"
                );
                assert_eq!(expected, vec![4]);
                assert_eq!(actual, vec![0]);
            }
            other => panic!("expected typed EmbeddingBag index error, got {other:?}"),
        }
    }
}

#[test]
fn embeddingbag_rejects_invalid_offsets_and_integer_indices() {
    let bag = seeded_embedding_bag(EmbeddingBagMode::Sum);

    let index_error = bag
        .forward_with_offsets(&[0, 4], None)
        .err()
        .expect("out-of-range EmbeddingBag index must be rejected");
    match index_error {
        ModuleError::ShapeMismatch {
            module,
            expected,
            actual,
            ..
        } => {
            assert_eq!(module, "EmbeddingBag");
            assert_eq!(expected, vec![4]);
            assert_eq!(actual, vec![1]);
        }
        other => panic!("expected typed EmbeddingBag index error, got {other:?}"),
    }

    let offset_error = bag
        .forward_with_offsets(&[0, 1], Some(&[1, 0]))
        .err()
        .expect("unordered EmbeddingBag offsets must be rejected");
    match offset_error {
        ModuleError::ShapeMismatch {
            module,
            parameter,
            actual,
            ..
        } => {
            assert_eq!(module, "EmbeddingBag");
            assert_eq!(
                parameter,
                "offsets must be non-empty, ordered, and within indices"
            );
            assert_eq!(actual, vec![1, 0]);
        }
        other => panic!("expected typed EmbeddingBag offset error, got {other:?}"),
    }
}
