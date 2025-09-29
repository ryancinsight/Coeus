#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{Backend, BackendKind, select_backend};

    proptest! {
        #[test]
        fn prop_matmul_broadcast(m in 1..=32usize, k in 1..=32usize, n in 1..=32usize) {
            let backend = select_backend(BackendKind::Gpu).unwrap();
            let a_data = backend.create_tensor_data(vec![1.0f32; m * k], vec![m, k]).unwrap();
            let b_data = backend.create_tensor_data(vec![1.0f32; k * n], vec![k, n]).unwrap();
            let result_data = backend.matmul(&a_data, &b_data).unwrap();
            let expected_sum = (k as f32) * (m as f32) * (n as f32);
            let actual_sum: f32 = result_data.data().iter().sum();
            prop_assert!((actual_sum - expected_sum).abs() < 1e-6);
        }

        #[test]
        fn prop_matmul_large(n in 64..=128usize) {
            let backend = select_backend(BackendKind::Gpu).unwrap();
            let a_data = backend.create_tensor_data(vec![1.0f32; n * n], vec![n, n]).unwrap();
            let b_data = backend.create_tensor_data(vec![1.0f32; n * n], vec![n, n]).unwrap();
            let result_data = backend.matmul(&a_data, &b_data).unwrap();
            let expected_sum = (n as f32).powi(3);
            let actual_sum: f32 = result_data.data().iter().sum();
            prop_assert!((actual_sum - expected_sum).abs() < 1e-6);
        }
    }
}
