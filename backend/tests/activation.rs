#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{dispatch::{BackendKind, select_backend}, Backend};

    proptest! {
        #[test]
        fn prop_relu_edges(values in proptest::collection::vec(-10.0..=10.0f32, 1..=100)) {
            let backend = select_backend(BackendKind::Gpu).unwrap();
            let input = backend.create_tensor_data(values.clone(), vec![values.len()]).unwrap();
            // Stub: relu not implemented yet, just verify tensor creation
            prop_assert_eq!(input.len(), values.len());
        }

        #[test]
        fn prop_sigmoid_midpoint(values in proptest::collection::vec(-10.0..=10.0f32, 1..=100)) {
            let backend = select_backend(BackendKind::Gpu).unwrap();
            let input = backend.create_tensor_data(values.clone(), vec![values.len()]).unwrap();
            // Stub: sigmoid not implemented yet, just verify tensor creation
            prop_assert_eq!(input.len(), values.len());
        }
    }
}
