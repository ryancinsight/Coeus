use proptest::prelude::*;
use coeus_tensor::{Tensor, CpuBackend, ops::indexing::{Slice, Indexing}};

proptest! {
    #[test]
    fn prop_slice_equiv(
        data in proptest::collection::vec(-10.0f32..10.0, 10..100),
        shape in proptest::collection::vec(2usize..5, 1..4)
    ) {
        let backend = CpuBackend::default();
        let shape_len = shape.iter().product();
        if data.len() >= shape_len {
            let t = Tensor::from_vec(backend, data[..shape_len].to_vec(), shape).unwrap();
            let s = t.slice(&[Slice::Range(0, 2)]).unwrap();
            // Assert view equiv (temp copy, future exact)
            prop_assert_eq!(s.data().len(), 2);
            prop_assert!((s.data()[0] - t.data()[0]).abs() < 1e-6);
            // Edges
            let err = t.slice(&[Slice::Range(100, 101)]); // Out-bounds
            prop_assert!(err.is_err());
        }
    }

    #[test]
    fn prop_gather_equiv(indices in proptest::collection::vec(0usize..10, 5..20)) {
        let backend = CpuBackend::default();
        let data = (0..20).map(|i| i as f32).collect();
        let t = Tensor::from_vec(backend, data, vec![20]).unwrap();
        let g = t.gather(0, &indices.iter().map(|&x| x as i64).collect::<Vec<_>>()).unwrap();
        prop_assert_eq!(g.data().len(), indices.len());
        for (i, &idx) in indices.iter().enumerate() {
            prop_assert_eq!(g.data()[i], t.data()[idx]);
        }
        // Edge out-bounds
        let err = t.gather(0, &[25i64]);
        prop_assert!(err.is_err());
    }
}
