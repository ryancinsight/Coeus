#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{Backend, BackendData, dispatch::{BackendKind, select_backend}};

    proptest! {
        #[test]
        fn prop_l2_norm_not_implemented(values in proptest::collection::vec(0.0..=10.0f32, 1..=100)) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let input = BackendData::cpu(values.clone(), vec![values.len()]);
            let result = backend.l2_norm(&input);
            prop_assert!(result.is_err()); // Should return NotImplemented
        }

        #[test]
        fn prop_attention_back_zero_grad(dim in 1..=8usize) {
            let backend = select_backend(BackendKind::Cpu).unwrap();
            let grad_out = BackendData::cpu(vec![0.0f32; dim * dim], vec![dim, dim]);
            let q = BackendData::cpu(vec![1.0f32; dim * dim], vec![dim, dim]);
            let k = BackendData::cpu(vec![1.0f32; dim * dim], vec![dim, dim]);
            let v = BackendData::cpu(vec![1.0f32; dim * dim], vec![dim, dim]);
            let (dq, dk, dv) = backend.attention_backward(&grad_out, &q, &k, &v).unwrap();
            prop_assert!(dq.data().iter().all(|&x| x == 0.0)); // Zero grad → zero backprop
        }
    }
}
