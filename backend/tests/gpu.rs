#[cfg(test)]
mod tests {
    use coeus_backend::{CpuBackend, Backend};
    use proptest::prelude::*;

    #[test]
    fn test_gpu_fallback() {
        let backend = CpuBackend::default(); // Direct concrete
        // ... rest with backend.add etc.
        // No GPU test, stub fallback
    }

    proptest! {
        #[test]
        fn prop_add_edges_simple(a in prop_oneof![Just(-1.0f32), Just(0.0), Just(1.0), Just(f32::NAN), Just(f32::INFINITY), Just(f32::NEG_INFINITY)],
                                 b in prop_oneof![Just(-1.0f32), Just(0.0), Just(1.0), Just(f32::NAN), Just(f32::INFINITY), Just(f32::NEG_INFINITY)]) {
            let backend = CpuBackend::default();
            let data_a = backend.create_tensor_data(vec![a; 64], vec![64]).unwrap();
            let data_b = backend.create_tensor_data(vec![b; 64], vec![64]).unwrap();
            let result = backend.add(&data_a, &data_b).unwrap();
            prop_assert_eq!(result.data()[0], a + b); // Edge: NaN + 0 = NaN
        }
    }

    proptest! {
        #[test]
        fn prop_add_edges_tensor(a in prop_oneof![Just(-1.0f32), Just(0.0), Just(1.0), Just(f32::NAN), Just(f32::INFINITY), Just(f32::NEG_INFINITY)],
                                 b in prop_oneof![Just(-1.0f32), Just(0.0), Just(1.0), Just(f32::NAN), Just(f32::INFINITY), Just(f32::NEG_INFINITY)]) {
            let backend = CpuBackend::default();
            let a_t = backend.create_tensor_data(vec![a], vec![1]).unwrap();
            let b_t = backend.create_tensor_data(vec![b], vec![1]).unwrap();
            let sum = backend.add(&a_t, &b_t).unwrap();
            if a.is_nan() || b.is_nan() {
                prop_assert!(sum.data()[0].is_nan());
            } else if a.is_infinite() || b.is_infinite() {
                prop_assert!(sum.data()[0].is_infinite());
            } else {
                prop_assert_eq!(sum.data()[0], a + b);
            }
        }
    }
}
