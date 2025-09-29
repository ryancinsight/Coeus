// Create if missing or add to existing tests mod

use crate::{Tensor, Backend, Dtype, Ops};
use crate::ops::arithmetic::add; // post-split
use proptest::prelude::*;
use approx::assert_relative_eq;

#[cfg(test)]
mod ops_tests {
    use super::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]
        fn prop_add_dispatch_equivalence_post_cleanup(
            data1 in vec![-10.0f32..10.0; 0..100],
            shape1 in vec![1..5; 1..4],
            data2 in vec![-10.0f32..10.0; 0..100],
            shape2 in vec![1..5; 1..4],
        ) {
            let backend = CpuBackend::default();
            let t1 = backend.create_tensor(data1.clone(), shape1.clone()).unwrap();
            let t2 = backend.create_tensor(data2.clone(), shape2.clone()).unwrap();

            let old_add = /* old if exists */ t1.clone() + t2.clone(); // assume
            let new_dispatch = Ops::Add(AddOp { lhs: t1.clone(), rhs: t2.clone() }).execute(&t1, &t2).unwrap();

            prop_assert_eq!(new_dispatch.shape(), old_add.shape());
            prop_assert!(new_dispatch.data().iter().zip(old_add.data().iter()).all(|(new, old)| (new - old).abs() < 1e-6));
        }

        // Edges: mul x=-1 y=10=-10 exact
        fn prop_mul_edge_neg() {
            let backend = CpuBackend::default();
            let a = backend.create_tensor(vec![-1.0f32], vec![1]).unwrap();
            let b = backend.create_tensor(vec![10.0f32], vec![1]).unwrap();

            let result = Ops::Mul(MulOp { lhs: a.clone(), rhs: b.clone() }).execute(&a, &b).unwrap();
            prop_assert_eq!(result.as_scalar().unwrap(), -10.0); // exact
        }

        // Overflow Err
        fn prop_overflow() {
            let backend = CpuBackend::default();
            let a = backend.create_tensor(vec![i32::MAX as f32], vec![1]).unwrap(); // assume i32 test
            let b = backend.create_tensor(vec![2.0f32], vec![1]).unwrap();
            let result = Ops::Mul(MulOp { lhs: a.clone(), rhs: b.clone() }).execute(&a, &b);
            prop_assert!(matches!(result, Err(TensorError::Overflow { .. })));
        }

        // Underflow/precision
        fn prop_underflow_precision() {
            let backend = CpuBackend::default();
            let a = backend.create_tensor(vec![1e-10f32], vec![1]).unwrap();
            let b = backend.create_tensor(vec![1.0f32], vec![1]).unwrap();
            let result = Ops::Add(AddOp { lhs: a.clone(), rhs: b.clone() }).execute(&a, &b).unwrap();
            prop_assert!((result.as_scalar().unwrap() - 1.00000001).abs() < 1e-6);
        }

        // Similar for sub/div/indexing select (post-split), matrix matmul equivalence, reduction sum/mean
    }
}

// Add to existing property_tests.rs or ops_tests if separate
