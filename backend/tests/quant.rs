#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{CpuBackend, Backend};

    proptest! {
        #[test]
        fn prop_quant_roundtrip(n in 1..=100usize) {
            let data = vec![1.0f32; n];
            let backend = CpuBackend::default();
            let tensor_data = backend.create_tensor_data(data.clone(), vec![n]).unwrap();
            // Stub: quantization not implemented yet, just verify tensor creation
            prop_assert_eq!(tensor_data.len(), n);
            prop_assert!(tensor_data.data().iter().all(|&x| x == 1.0));
        }
    }
}
