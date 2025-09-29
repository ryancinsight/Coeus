#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{Backend, BackendData, CpuBackend};

    proptest! {
        #[test]
        fn prop_matmul_not_implemented(n in 1..=16usize) {
            let cpu_backend = CpuBackend::new();
            let a = BackendData::cpu(vec![1.0f32; n * n], vec![n, n]);
            let b = BackendData::cpu(vec![1.0f32; n * n], vec![n, n]);
            let result = cpu_backend.matmul(&a, &b);
            prop_assert!(result.is_err()); // Should return NotImplemented
        }
    }
}
