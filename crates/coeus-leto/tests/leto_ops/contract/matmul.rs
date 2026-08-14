#![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
use super::support::layout;
use super::{batched_matmul_into, matmul_into, Layout, Shape, Strides};

#[test]
fn matmul_matches_reference() {
    // [[1,2,3],[4,5,6]] x [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut out = vec![0.0f64; 4];

    matmul_into(
        &layout(&[2, 3]),
        &a,
        &layout(&[3, 2]),
        &b,
        &layout(&[2, 2]),
        &mut out,
    )
    .unwrap();
    assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn matmul_handles_transposed_input_view() {
    // a stored as [3,2] but used transposed as [2,3] via explicit strides.
    let a_storage = vec![1.0f64, 4.0, 2.0, 5.0, 3.0, 6.0]; // logical [3,2]
                                                           // transposed layout: shape [2,3], strides swapped.
    let a_t = Layout::from_shape_strides(
        Shape::from(vec![2usize, 3]),
        Strides::from_slice(&[1usize, 2]),
        0,
    );
    let b = vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut out = vec![0.0f64; 4];

    matmul_into(
        &a_t,
        &a_storage,
        &layout(&[3, 2]),
        &b,
        &layout(&[2, 2]),
        &mut out,
    )
    .unwrap();
    // transposed a is [[1,2,3],[4,5,6]] -> same product as the contiguous case.
    assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn batched_matmul_dispatch_covers_rhs_batch_broadcast() {
    let lhs = vec![
        1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, //
        7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let rhs = vec![2.0f64, 3.0, 5.0, 7.0, 11.0, 13.0];
    let mut out = vec![0.0f64; 8];
    let lhs_layout = layout(&[2, 2, 3]);
    let rhs_layout = layout(&[1, 3, 2]);
    let out_layout = layout(&[2, 2, 2]);

    batched_matmul_into(&lhs_layout, &lhs, &rhs_layout, &rhs, &out_layout, &mut out).unwrap();

    assert_eq!(
        out,
        vec![45.0, 56.0, 99.0, 125.0, 153.0, 194.0, 207.0, 263.0]
    );
}
