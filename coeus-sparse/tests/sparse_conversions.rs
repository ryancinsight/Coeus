// Round-trip correctness for the sparse format conversions
// (dense <-> COO <-> CSR). Densify-after-convert must reproduce the original
// dense matrix exactly (identity oracle), and the two routes to CSR
// (dense_to_csr vs coo_to_csr . dense_to_coo) must agree structurally.

use coeus_core::SequentialBackend;
use coeus_ops::{coo_to_csr, coo_to_dense, csr_to_dense, dense_to_coo, dense_to_csr};
use coeus_tensor::Tensor;

type Seq = SequentialBackend;

/// A 3x4 matrix with a realistic scatter of structural zeros (5 non-zeros),
/// including a row with a single entry and a fully-dense final row tail.
fn dense_3x4() -> Tensor<f32, Seq> {
    #[rustfmt::skip]
    let data = vec![
        1.0f32, 0.0, 0.0, 2.0,
        0.0,    3.0, 0.0, 0.0,
        0.0,    0.0, 4.0, 5.0,
    ];
    Tensor::<f32, Seq>::from_slice([3, 4], &data)
}

#[test]
fn dense_coo_roundtrip_is_identity() {
    let s = SequentialBackend::new();
    let dense = dense_3x4();
    let coo = dense_to_coo(&dense, &s);
    assert_eq!(coo.nnz(), 5, "expected 5 structural non-zeros");
    let back = coo_to_dense(&coo, &s);
    assert_eq!(back.shape(), dense.shape());
    assert_eq!(back.as_slice(), dense.as_slice());
}

#[test]
fn dense_csr_roundtrip_is_identity() {
    let s = SequentialBackend::new();
    let dense = dense_3x4();
    let csr = dense_to_csr(&dense, &s);
    assert_eq!(csr.nnz(), 5);
    // row_offsets must be monotonic and span [0, nnz] over rows + 1.
    let ro = csr.row_offsets().as_slice();
    assert_eq!(ro.len(), 4); // rows + 1
    assert_eq!(ro[0], 0);
    assert_eq!(ro[3], 5);
    let back = csr_to_dense(&csr, &s);
    assert_eq!(back.as_slice(), dense.as_slice());
}

#[test]
fn dense_coo_csr_dense_full_chain_is_identity() {
    let s = SequentialBackend::new();
    let dense = dense_3x4();
    let coo = dense_to_coo(&dense, &s);
    let csr = coo_to_csr(&coo, &s);
    let back = csr_to_dense(&csr, &s);
    assert_eq!(back.as_slice(), dense.as_slice());
}

#[test]
fn dense_to_csr_matches_coo_to_csr_route() {
    // Both routes to CSR must produce structurally identical tensors.
    let s = SequentialBackend::new();
    let dense = dense_3x4();
    let direct = dense_to_csr(&dense, &s);
    let via_coo = coo_to_csr(&dense_to_coo(&dense, &s), &s);

    assert_eq!(direct.values().as_slice(), via_coo.values().as_slice());
    assert_eq!(
        direct.col_indices().as_slice(),
        via_coo.col_indices().as_slice()
    );
    assert_eq!(
        direct.row_offsets().as_slice(),
        via_coo.row_offsets().as_slice()
    );
}
