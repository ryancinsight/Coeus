use super::support::layout;
use super::{batched_matmul_accumulate_into, elementwise_add_into, matmul_accumulate_into};

#[test]
fn rank_beyond_dispatch_bound_is_rejected() {
    let a = vec![0.0f64; 128];
    let la = layout(&[2, 2, 2, 2, 2, 2, 2]); // rank 7 > MAX_DISPATCH_RANK
    let mut out = vec![0.0f64; 128];
    assert!(elementwise_add_into(&la, &a, &la, &a, &la, &mut out).is_err());
}

#[test]
fn matmul_accumulate_adds_into_existing_output() {
    // out += A*B (must accumulate onto a non-zero output, not overwrite).
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]] -> A*B = [[19,22],[43,50]].
    let a = vec![1.0f64, 2.0, 3.0, 4.0];
    let b = vec![5.0f64, 6.0, 7.0, 8.0];
    let mut out = vec![1.0f64; 4]; // pre-seeded
    let l = layout(&[2, 2]);

    matmul_accumulate_into(&l, &a, &l, &b, &l, &mut out).unwrap();
    // [[19+1, 22+1], [43+1, 50+1]]
    assert_eq!(out, vec![20.0, 23.0, 44.0, 51.0]);
}

#[test]
fn batched_matmul_accumulate_adds_per_batch() {
    // out += A*B over a batch of 2. Batch 0: I*B0 = B0; batch 1: 2I*B1 = 2*B1.
    let a = vec![
        1.0f64, 0.0, 0.0, 1.0, // batch 0: identity
        2.0, 0.0, 0.0, 2.0, // batch 1: 2*identity
    ];
    let b = vec![
        5.0f64, 6.0, 7.0, 8.0, // batch 0
        1.0, 1.0, 1.0, 1.0, // batch 1
    ];
    let mut out = vec![1.0f64; 8]; // pre-seeded
    let l = layout(&[2, 2, 2]);

    batched_matmul_accumulate_into(&l, &a, &l, &b, &l, &mut out).unwrap();
    // batch 0: [5,6,7,8] + 1; batch 1: [2,2,2,2] + 1
    assert_eq!(out, vec![6.0, 7.0, 8.0, 9.0, 3.0, 3.0, 3.0, 3.0]);
}
