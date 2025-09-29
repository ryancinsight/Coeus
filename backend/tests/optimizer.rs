#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{Backend, BackendData, BackendKind, select_backend};

    proptest! {
        #[test]
        fn prop_gelu_approx(x in -10.0..=10.0f32) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let input = BackendData::cpu(vec![x], vec![1]);
            let output = backend.gelu(&input).unwrap();
            let expected = 0.5 * x * (1.0 + (0.79788456 * x * (1.0 + 0.044715 * x * x)).tanh());
            prop_assert!((output.data()[0] - expected).abs() < 1e-4); // Approx check
        }

        #[test]
        fn prop_adam_zero_grad(steps in 1..=5usize) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let m = BackendData::cpu(vec![0.0f32; 1], vec![1]);
            let v = BackendData::cpu(vec![0.0f32; 1], vec![1]);
            let grad = BackendData::cpu(vec![0.0f32; 1], vec![1]);
            let lr = 0.001;
            let beta1 = 0.9;
            let beta2 = 0.999;
            let eps = 1e-8;
            for t in 1..=steps {
                let update = backend.adam_step(&m, &v, &grad, lr, beta1, beta2, eps, t as f32).unwrap();
                prop_assert!(update.data()[0].abs() < 1e-6); // Zero grad → no update
            }
        }
    }
}
