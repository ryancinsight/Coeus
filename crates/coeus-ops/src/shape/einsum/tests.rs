//! Value-oracle tests for the einsum patterns and their typed rejections.

use super::{einsum, einsum3};
use coeus_core::{BackendError, SequentialBackend};
use coeus_tensor::Tensor;

fn b() -> SequentialBackend {
    SequentialBackend::new()
}

#[test]
fn einsum_matmul() {
    let a = Tensor::from_slice(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("invariant: test backend operation succeeds");
    let bt = Tensor::from_slice(vec![3, 2], &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0])
        .expect("invariant: test backend operation succeeds");
    let out = einsum("ij,jk->ik", &[&a, &bt], &b()).expect("valid einsum test shapes");
    assert_eq!(out.shape(), &[2, 2]);
    // row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
    // row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
    assert_eq!(out.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn einsum_transpose() {
    let a = Tensor::from_slice(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("invariant: test backend operation succeeds");
    let out = einsum("ij->ji", &[&a], &b()).expect("valid einsum test shapes");
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(out.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn einsum_dot_product() {
    let a = Tensor::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let bt = Tensor::from_slice(vec![4], &[5.0f32, 6.0, 7.0, 8.0])
        .expect("invariant: test backend operation succeeds");
    let out = einsum("i,i->", &[&a, &bt], &b()).expect("valid einsum test shapes");
    assert_eq!(out.shape(), &[1]);
    assert_eq!(
        out.as_slice(),
        &[1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0]
    );
}

#[test]
fn einsum_trace() {
    let a = Tensor::from_slice(
        vec![3, 3],
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .expect("invariant: test backend operation succeeds");
    let out = einsum("ii->", &[&a], &b()).expect("valid einsum test shapes");
    assert_eq!(out.as_slice(), &[1.0 + 5.0 + 9.0]);
}

#[test]
fn einsum_outer_product() {
    let a = Tensor::from_slice(vec![2], &[1.0f32, 2.0])
        .expect("invariant: test backend operation succeeds");
    let bt = Tensor::from_slice(vec![3], &[3.0f32, 4.0, 5.0])
        .expect("invariant: test backend operation succeeds");
    let out = einsum("i,j->ij", &[&a, &bt], &b()).expect("valid einsum test shapes");
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.as_slice(), &[3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
}

#[test]
fn einsum_matvec() {
    let a = Tensor::from_slice(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("invariant: test backend operation succeeds");
    let v = Tensor::from_slice(vec![3], &[1.0f32, 0.0, 1.0])
        .expect("invariant: test backend operation succeeds");
    let out = einsum("ij,j->i", &[&a, &v], &b()).expect("valid einsum test shapes");
    assert_eq!(out.shape(), &[2]);
    assert_eq!(out.as_slice(), &[4.0, 10.0]);
}

#[test]
fn einsum_batched_matmul() {
    // batch=1, m=2, k=2, n=2
    let a = Tensor::from_slice(vec![1, 2, 2], &[1.0f32, 2.0, 3.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let bt = Tensor::from_slice(vec![1, 2, 2], &[5.0f32, 6.0, 7.0, 8.0])
        .expect("invariant: test backend operation succeeds");
    let out = einsum("bij,bjk->bik", &[&a, &bt], &b()).expect("valid einsum test shapes");
    assert_eq!(out.shape(), &[1, 2, 2]);
    // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    assert_eq!(out.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn einsum_batched_matmul_non_square_batch() {
    // The batched arm dispatches to the tiled `bmm` kernel, so the oracle
    // covers more than one batch and rectangular tiles rather than the
    // square 1-batch case alone.
    let a = Tensor::from_slice(
        vec![2, 2, 3],
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ],
    )
    .expect("invariant: test backend operation succeeds");
    let bt = Tensor::from_slice(
        vec![2, 3, 2],
        &[
            1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .expect("invariant: test backend operation succeeds");
    let out = einsum("bij,bjk->bik", &[&a, &bt], &b()).expect("valid einsum test shapes");
    assert_eq!(out.shape(), &[2, 2, 2]);
    assert_eq!(out.as_slice(), &[4.0, 5.0, 10.0, 11.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn einsum_inner_dot_transposes_the_second_operand() {
    // "ij,kj->ik" contracts the shared second axis; the arm materializes
    // `b.T` and dispatches to the tiled matmul, so the value oracle is what
    // proves the transpose and the dispatch agree.
    let a = Tensor::from_slice(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("invariant: test backend operation succeeds");
    let bt = Tensor::from_slice(vec![2, 3], &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0])
        .expect("invariant: test backend operation succeeds");
    let out = einsum("ij,kj->ik", &[&a, &bt], &b()).expect("valid einsum test shapes");
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice(), &[50.0, 68.0, 122.0, 167.0]);
}

#[test]
fn einsum_three_operand_matmul_chain() {
    let a = Tensor::from_slice(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let bt = Tensor::from_slice(vec![2, 2], &[5.0f32, 6.0, 7.0, 8.0])
        .expect("invariant: test backend operation succeeds");
    let c = Tensor::from_slice(vec![2, 2], &[9.0f32, 10.0, 11.0, 12.0])
        .expect("invariant: test backend operation succeeds");
    let out =
        einsum3("ij,jk,kl->il", &a, &bt, &c, &b()).expect("valid three-operand einsum test shapes");
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.as_slice(), &[413.0, 454.0, 937.0, 1030.0]);
}

#[test]
fn einsum_rejects_operand_count_mismatch() {
    let a = Tensor::from_slice(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let Err(err) = einsum("ij,jk->ik", &[&a], &b()) else {
        panic!("count mismatch must fail");
    };
    assert!(
        matches!(
            err,
            BackendError::EinsumOperandCount {
                subscript_operands: 2,
                provided: 1,
                ..
            }
        ),
        "expected EinsumOperandCount, got {err:?}"
    );
}

#[test]
fn einsum_rejects_unknown_pattern() {
    let a = Tensor::from_slice(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let bt = Tensor::from_slice(vec![2, 2], &[5.0f32, 6.0, 7.0, 8.0])
        .expect("invariant: test backend operation succeeds");
    let Err(err) = einsum("ab,cd->ef", &[&a, &bt], &b()) else {
        panic!("unknown pattern must fail");
    };
    assert!(
        matches!(
            err,
            BackendError::EinsumUnsupportedPattern {
                operand_count: 2,
                ..
            }
        ),
        "expected EinsumUnsupportedPattern, got {err:?}"
    );
}

#[test]
fn einsum_rejects_rank_mismatch() {
    let a = Tensor::from_slice(vec![2, 2, 2], &[1.0f32; 8])
        .expect("invariant: test backend operation succeeds");
    let Err(err) = einsum("ij->ji", &[&a], &b()) else {
        panic!("rank mismatch must fail");
    };
    assert!(
        matches!(
            err,
            BackendError::EinsumRank {
                operand: 0,
                expected: 2,
                actual: 3,
                ..
            }
        ),
        "expected EinsumRank, got {err:?}"
    );
}

#[test]
fn einsum_rejects_dim_mismatch() {
    let a = Tensor::from_slice(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("invariant: test backend operation succeeds");
    let bt = Tensor::from_slice(
        vec![4, 2],
        &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
    )
    .expect("invariant: test backend operation succeeds");
    let Err(err) = einsum("ij,j->i", &[&a, &bt], &b()) else {
        panic!("rank mismatch must fail");
    };
    // "ij,j->i" with a 2-D second operand is a rank rejection on operand 1.
    assert!(
        matches!(
            err,
            BackendError::EinsumRank {
                operand: 1,
                expected: 1,
                actual: 2,
                ..
            }
        ),
        "expected EinsumRank, got {err:?}"
    );
    let v = Tensor::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let Err(err) = einsum("ij,j->i", &[&a, &v], &b()) else {
        panic!("dim mismatch must fail");
    };
    // Contraction dim k = 3 against a length-4 vector.
    assert!(
        matches!(
            err,
            BackendError::EinsumDim {
                operand: 1,
                expected: 3,
                actual: 4,
                ..
            }
        ),
        "expected EinsumDim, got {err:?}"
    );
}

#[test]
fn einsum_rejects_empty_subscript_without_panic() {
    // "->" parses to empty labels; the last-two-dims swap check must
    // decline (n < 2 would underflow `n - 2`), not panic.
    let a = Tensor::from_slice(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let Err(err) = einsum("->", &[&a], &b()) else {
        panic!("empty subscript must fail");
    };
    assert!(
        matches!(
            err,
            BackendError::EinsumUnsupportedPattern {
                operand_count: 1,
                ..
            }
        ),
        "expected EinsumUnsupportedPattern, got {err:?}"
    );
}

#[test]
fn einsum_dot_rejects_length_mismatch() {
    // Same label, one extent: zipping to the shorter length would be a
    // quiet short sum, so mismatch is a typed error.
    let a = Tensor::from_slice(vec![4], &[1.0f32, 2.0, 3.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let bt = Tensor::from_slice(vec![3], &[1.0f32, 0.0, 1.0])
        .expect("invariant: test backend operation succeeds");
    let Err(err) = einsum("i,i->", &[&a, &bt], &b()) else {
        panic!("length mismatch must fail");
    };
    assert!(
        matches!(
            err,
            BackendError::EinsumDim {
                operand: 1,
                expected: 4,
                actual: 3,
                ..
            }
        ),
        "expected EinsumDim, got {err:?}"
    );
}

#[test]
fn einsum3_rejects_unknown_pattern() {
    let a = Tensor::from_slice(vec![2, 2], &[1.0f32, 2.0, 3.0, 4.0])
        .expect("invariant: test backend operation succeeds");
    let bt = Tensor::from_slice(vec![2, 2], &[5.0f32, 6.0, 7.0, 8.0])
        .expect("invariant: test backend operation succeeds");
    let c = Tensor::from_slice(vec![2, 2], &[9.0f32, 10.0, 11.0, 12.0])
        .expect("invariant: test backend operation succeeds");
    let Err(err) = einsum3("ab,bc,cd->ad", &a, &bt, &c, &b()) else {
        panic!("unknown pattern must fail");
    };
    assert!(
        matches!(
            err,
            BackendError::EinsumUnsupportedPattern {
                operand_count: 3,
                ..
            }
        ),
        "expected EinsumUnsupportedPattern, got {err:?}"
    );
}

#[test]
fn einsum_swap_rejects_label_rank_mismatch() {
    // "bij->bji" names three labels: a 2-D tensor must be rejected, not
    // silently transposed by the generic last-two-dims swap arm.
    let a = Tensor::from_slice(vec![2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("invariant: test backend operation succeeds");
    let Err(err) = einsum("bij->bji", &[&a], &b()) else {
        panic!("label/rank mismatch must fail");
    };
    assert!(
        matches!(
            err,
            BackendError::EinsumRank {
                operand: 0,
                expected: 3,
                actual: 2,
                ..
            }
        ),
        "expected EinsumRank, got {err:?}"
    );
}
