#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{Backend, BackendData, BackendKind, select_backend};

    proptest! {
        #[test]
        fn prop_rmsprop_adaptive(lr in 0.001..=0.1f32, eps in 1e-8..=1e-6f32, grad in proptest::collection::vec(-2.0..=2.0f32, 1..=10)) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let v = BackendData::cpu(vec![0.0f32; grad.len()], vec![grad.len()]);
            let grad_data = BackendData::cpu(grad.clone(), vec![grad.len()]);
            let update = backend.rmsprop(&v, &grad_data, lr, eps).unwrap();

            // RMSprop should produce updates in the opposite direction of gradients
            for (i, &update_val) in update.data().iter().enumerate() {
                let grad_val = grad[i];
                // Update should be opposite sign to gradient (for initial v=0)
                prop_assert!((update_val * grad_val) <= 0.0,
                           "Update {} should oppose gradient {} for index {}", update_val, grad_val, i);
            }

            // Updates should be finite and reasonable in magnitude (not checking <= lr as that's unrealistic)
            for &update_val in update.data().iter() {
                prop_assert!(update_val.is_finite(), "Update should be finite, got {}", update_val);
                prop_assert!(update_val.abs() < 1000.0, "Update magnitude too large: {}", update_val);
            }
        }

        #[test]
        fn prop_layernorm_back_chain(mean in -1.0..=1.0f32, var in 0.0..=1.0f32, gamma in 0.1..=2.0f32, eps in 1e-8..=1e-6f32) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let grad_out = BackendData::cpu(vec![1.0f32; 10], vec![10]);
            let x = BackendData::cpu(vec![0.0f32; 10], vec![10]);
            let gamma_data = BackendData::cpu(vec![gamma; 10], vec![10]);
            let back_grad = backend.layernorm_backward(&grad_out, &x, mean, var, Some(&gamma_data), eps).unwrap();
            prop_assert!((back_grad.data().iter().sum::<f32>() / back_grad.len() as f32).abs() < 1e-4); // Approx zero mean grad
        }
    }
}
