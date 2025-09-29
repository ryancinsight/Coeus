#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{Backend, BackendData, CpuBackend};

    proptest! {
        #[test]
        fn prop_quantized_infer_not_implemented(n in 1..=10usize) {
            let cpu_backend = CpuBackend::new();
            let input = BackendData::cpu(vec![1i8; n], vec![n]); // Quantized
            let result = cpu_backend.quantized_infer(&input); // CPU quant impl
            prop_assert!(result.is_err()); // Should return NotImplemented
        }
    }
}
