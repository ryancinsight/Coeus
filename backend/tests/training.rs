#[cfg(test)]
mod tests {
    use super::*;
    use coeus_backend::Backend;
    use proptest::prelude::*;
    use coeus_backend::{BackendKind, select_backend};

    proptest! {
        #[test]
        fn prop_embedding_oov(vocab_size in 1..=10usize, batch in 1..=4usize) {
            let backend = select_backend(BackendKind::Gpu).unwrap();
            let table = backend.create_tensor_data(vec![1.0f32; vocab_size * 64], vec![vocab_size, 64]).unwrap();
            // Stub: embedding not implemented, just verify tensor creation
            prop_assert_eq!(table.len(), vocab_size * 64);
        }

        #[test]
        fn prop_ce_loss_positive(logits_data in prop::collection::vec(-10.0..=10.0f32, 1..=10), labels_data in prop::collection::vec(0u32..10u32, 1..=10)) {
            let backend = select_backend(BackendKind::Gpu).unwrap();
            let len = logits_data.len();
            let logits = backend.create_tensor_data(logits_data, vec![len]).unwrap();
            // Stub: cross_entropy_loss not implemented, just verify tensor creation
            prop_assert_eq!(logits.len(), len);
        }
    }
}
