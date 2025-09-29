#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use coeus_backend::{CpuBackend, Backend};
    use std::f32;

    proptest! {
        #[test]
        fn prop_mul_edges(a in prop::num::f32::ANY,
                          b in prop::num::f32::ANY) {
            let backend = CpuBackend::default();
            let data_a = backend.create_tensor_data(vec![a; 64], vec![64]).unwrap();
            let data_b = backend.create_tensor_data(vec![b; 64], vec![64]).unwrap();
            let result = backend.mul(&data_a, &data_b).unwrap();
            prop_assert_eq!(result.data()[0], a * b); // Edge: NaN * 0 = NaN
        }

        #[test]
        fn prop_matmul_shapes(n in 1..=32u32) {
            let backend = CpuBackend::default();
            let a_data = backend.create_tensor_data(vec![1.0f32; (n * n) as usize], vec![n as usize, n as usize]).unwrap(); // n x n
            let b_data = backend.create_tensor_data(vec![1.0f32; (n * n) as usize], vec![n as usize, n as usize]).unwrap(); // n x n
            let result = backend.matmul(&a_data, &b_data).unwrap(); // n x n
            prop_assert_eq!(result.data().iter().sum::<f32>(), (n as f32).powi(3)); // Sum check
        }

        #[test]
        fn prop_add_edges(a in prop::num::f32::ANY, b in prop::num::f32::ANY) {
            let backend = CpuBackend::default();
            let a_data = vec![a];
            let b_data = vec![b];
            let a_tensor = backend.create_tensor_data(a_data, vec![1]).unwrap();
            let b_tensor = backend.create_tensor_data(b_data, vec![1]).unwrap();
            let result = backend.add(&a_tensor, &b_tensor).unwrap();
            prop_assert_eq!(result.data()[0], a + b);
        }
    }
}
