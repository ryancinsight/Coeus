#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{Backend, BackendData, BackendKind, select_backend};

    proptest! {
        #[test]
        fn prop_layernorm_stability(values in proptest::collection::vec(-10.0..=10.0f32, 1..=100), mean in -1.0..=1.0f32, var in 0.0..=1.0f32) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let input = BackendData::cpu(values.clone(), vec![values.len()]);
            let output = backend.layernorm(&input, Some(mean), Some(var), Some(1.0), Some(0.0), 1e-8).unwrap();
            let sum: f32 = output.data().iter().sum();
            let mean_out = sum / output.len() as f32;
            prop_assert!((mean_out - 0.0).abs() < 1e-4); // Normalized mean ≈ 0
        }

        #[test]
        fn prop_dropout_prob(values in proptest::collection::vec(0.0..=1.0f32, 1..=100), p in 0.0..=1.0f32) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let input = BackendData::cpu(values.clone(), vec![values.len()]);
            let output = backend.dropout(&input, p).unwrap();
            let active_ratio = output.data().iter().filter(|&x| *x != 0.0).count() as f32 / input.len() as f32;
            prop_assert!((active_ratio - (1.0 - p)).abs() < 0.1); // Approx retention rate
        }
    }
}
