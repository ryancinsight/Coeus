#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{Backend, BackendData, BackendKind, select_backend};

    proptest! {
        #[test]
        fn prop_allreduce_sync(values in proptest::collection::vec(1.0..=10.0f32, 1..=8), world_size in 1..=4usize) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let input_data = BackendData::cpu(values.clone(), vec![values.len()]);
            let reduced = backend.allreduce(&input_data, world_size);
            // Stub implementation returns error, so we expect it to fail
            prop_assert!(reduced.is_err());
        }

        #[test]
        fn prop_upsample_scale(values in proptest::collection::vec(0.0..=1.0f32, 4), scale in 1..=2usize) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let input_data = BackendData::cpu(values.clone(), vec![2, 2]);
            let output = backend.upsample(&input_data, scale as f32);
            // Stub implementation returns error, so we expect it to fail
            prop_assert!(output.is_err());
        }
    }
}
