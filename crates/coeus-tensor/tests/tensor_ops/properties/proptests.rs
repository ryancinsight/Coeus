#![allow(clippy::needless_range_loop)]
use coeus_core::{MoiraiBackend, SequentialBackend};
use coeus_tensor::{Tensor, Transpose};
use proptest::prelude::*;

// Strategies for shapes
fn shape_strategy() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(1..=8usize, 1..=3)
}

fn shape2d_strategy() -> impl Strategy<Value = (usize, usize)> {
    (1..=16usize, 1..=16usize)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn test_transpose_invariant((m, n) in shape2d_strategy()) {
        let size = m * n;
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let a = Tensor::<f64, SequentialBackend>::from_slice(vec![m, n], &data);

        let a_t = a.transpose();

        prop_assert_eq!(a_t.shape(), &[n, m]);
        for r in 0..m {
            for c in 0..n {
                prop_assert_eq!(a_t.get(&[c, r]), a.get(&[r, c]));
            }
        }
    }

    #[test]
    fn test_reshape_invariant((m, n) in shape2d_strategy()) {
        let size = m * n;
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let a = Tensor::<f64, SequentialBackend>::from_slice(vec![m, n], &data);

        let a_reshaped = a.reshape(vec![size]);

        prop_assert_eq!(a_reshaped.shape(), &[size]);
        prop_assert!(a_reshaped.is_contiguous());
        for i in 0..size {
            prop_assert_eq!(a_reshaped.get(&[i]), data[i]);
        }
    }

    #[test]
    fn test_broadcasting_add_parity((m, n) in shape2d_strategy()) {
        let backend = SequentialBackend::new();
        let data_a: Vec<f64> = (0..m).map(|i| i as f64).collect();
        let data_b: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();

        let a = Tensor::<f64, SequentialBackend>::from_slice(vec![m, 1], &data_a);
        let b = Tensor::<f64, SequentialBackend>::from_slice(vec![1, n], &data_b);

        let c = coeus_ops::add(&a, &b, &backend);

        prop_assert_eq!(c.shape(), &[m, n]);
        for r in 0..m {
            for col in 0..n {
                let expected = data_a[r] + data_b[col];
                prop_assert_eq!(c.get(&[r, col]), expected);
            }
        }
    }

    #[test]
    fn test_moirai_backend_parity(shape in shape_strategy()) {
        let size: usize = shape.iter().product();
        let data_a: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let data_b: Vec<f64> = (0..size).map(|i| (i * 3) as f64).collect();

        let a_seq = Tensor::<f64, SequentialBackend>::from_slice(shape.clone(), &data_a);
        let b_seq = Tensor::<f64, SequentialBackend>::from_slice(shape.clone(), &data_b);
        let seq_backend = SequentialBackend::new();
        let c_seq = coeus_ops::add(&a_seq, &b_seq, &seq_backend);

        let a_moirai = Tensor::<f64, MoiraiBackend>::from_slice(shape.clone(), &data_a);
        let b_moirai = Tensor::<f64, MoiraiBackend>::from_slice(shape.clone(), &data_b);
        let moirai_backend = MoiraiBackend::new();
        let c_moirai = coeus_ops::add(&a_moirai, &b_moirai, &moirai_backend);

        prop_assert_eq!(c_moirai.shape(), c_seq.shape());
        let seq_slice = c_seq.as_slice();
        let moirai_slice = c_moirai.as_slice();
        for i in 0..size {
            let diff = (seq_slice[i] - moirai_slice[i]).abs();
            prop_assert!(diff < 1e-7, "Value mismatch at index {i}: seq={}, moirai={}", seq_slice[i], moirai_slice[i]);
        }
    }
}
