#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{Backend, BackendData, BackendKind, select_backend};

    proptest! {
        #[test]
        fn prop_fused_batchnorm_not_implemented(values in proptest::collection::vec(-10.0..=10.0f32, 1..=100), mean in -1.0..=1.0f32, var in 0.0..=1.0f32) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let input = BackendData::cpu(values.clone(), vec![values.len()]);
            let result = backend.fused_batchnorm(&input, mean, var, 1.0, 0.0, 1e-8);
            prop_assert!(result.is_err()); // Should return NotImplemented
        }

        #[test]
        fn prop_fused_adam_not_implemented(grad in proptest::collection::vec(0.0..=1.0f32, 1..=10), lr in 0.001..=0.1f32) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let m = BackendData::cpu(vec![0.0f32; grad.len()], vec![grad.len()]);
            let v = BackendData::cpu(vec![0.0f32; grad.len()], vec![grad.len()]);
            let grad_data = BackendData::cpu(grad.clone(), vec![grad.len()]);
            let result = backend.fused_adam(&m, &v, &grad_data, lr, 0.9, 0.999, 1e-8, 1.0);
            prop_assert!(result.is_err()); // Should return NotImplemented
        }
    }
}
