#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{dispatch::{BackendKind, select_backend}, Backend};
    use std::f32;

    proptest! {
        #[test]
        fn prop_batchnorm_stability(values in proptest::collection::vec(-10.0..=10.0f32, 1..=100), mean in -1.0..=1.0f32, var in 0.0..=1.0f32) {
            let backend = select_backend(BackendKind::Gpu).unwrap();
            let input = backend.create_tensor_data(values.clone(), vec![values.len()]).unwrap();
            // Stub: batchnorm not implemented yet, just verify tensor creation
            prop_assert_eq!(input.len(), values.len());
        }

        #[test]
        fn prop_softmax_stable(values in proptest::collection::vec(prop::num::f32::ANY, 3..=10)) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let input = backend.create_tensor_data(values.clone(), vec![values.len()]).unwrap();
            // Stub: softmax not implemented yet, just verify tensor creation
            prop_assert_eq!(input.len(), values.len());
        }
    }
}
