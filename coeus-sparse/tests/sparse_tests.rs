// Value-semantic tests for the COO / CSR sparse tensor types.
//
// `coeus-sparse` holds the pure sparse data structures (conversions and
// arithmetic live in `coeus-ops`/`coeus-leto` per DIP). These tests pin the
// construction invariants the public constructors enforce and verify accessor
// round-trips against a known 2×3 sparse matrix:
//
//     [[1, 0, 2],
//      [0, 3, 0]]
//
// COO indices are `[rank, nnz]`: row coords `[0,0,1]`, col coords `[0,2,1]`.
// CSR is `values=[1,2,3]`, `col_indices=[0,2,1]`, `row_offsets=[0,2,3]`.

use coeus_core::{SequentialBackend, Shape};
use coeus_sparse::{CooTensor, CsrTensor};
use coeus_tensor::Tensor;

type Seq = SequentialBackend;

fn shape2(r: usize, c: usize) -> Shape {
    Shape::from(vec![r, c])
}

// ── COO ───────────────────────────────────────────────────────────────────

fn coo_2x3() -> CooTensor<f32, Seq> {
    // indices [rank=2, nnz=3] row-major: rows then cols.
    let indices = Tensor::<i64, Seq>::from_slice(vec![2, 3], &[0, 0, 1, 0, 2, 1]);
    let values = Tensor::<f32, Seq>::from_slice(vec![3], &[1.0, 2.0, 3.0]);
    CooTensor::new(shape2(2, 3), indices, values)
}

#[test]
fn coo_construction_and_accessors() {
    let coo = coo_2x3();
    assert_eq!(&coo.shape()[..], &[2usize, 3]);
    assert_eq!(coo.nnz(), 3);
    assert_eq!(coo.indices().shape(), &[2, 3]);
    assert_eq!(coo.indices().as_slice(), &[0, 0, 1, 0, 2, 1]);
    assert_eq!(coo.values().as_slice(), &[1.0, 2.0, 3.0]);
}

#[test]
fn coo_clone_preserves_values() {
    let coo = coo_2x3();
    let cloned = coo.clone();
    assert_eq!(&cloned.shape()[..], &coo.shape()[..]);
    assert_eq!(cloned.values().as_slice(), coo.values().as_slice());
    assert_eq!(cloned.indices().as_slice(), coo.indices().as_slice());
}

#[test]
#[should_panic(expected = "Indices tensor must be 2D")]
fn coo_rejects_non_2d_indices() {
    let indices = Tensor::<i64, Seq>::from_slice(vec![6], &[0, 0, 1, 0, 2, 1]);
    let values = Tensor::<f32, Seq>::from_slice(vec![3], &[1.0, 2.0, 3.0]);
    CooTensor::new(shape2(2, 3), indices, values);
}

#[test]
#[should_panic(expected = "Indices row count must match tensor rank")]
fn coo_rejects_rank_mismatch() {
    // rank-2 shape but indices declare 3 coordinate rows.
    let indices = Tensor::<i64, Seq>::from_slice(vec![3, 3], &[0; 9]);
    let values = Tensor::<f32, Seq>::from_slice(vec![3], &[1.0, 2.0, 3.0]);
    CooTensor::new(shape2(2, 3), indices, values);
}

#[test]
#[should_panic(expected = "Indices col count must match number of values")]
fn coo_rejects_nnz_mismatch() {
    // indices declare 3 nnz but only 2 values supplied.
    let indices = Tensor::<i64, Seq>::from_slice(vec![2, 3], &[0, 0, 1, 0, 2, 1]);
    let values = Tensor::<f32, Seq>::from_slice(vec![2], &[1.0, 2.0]);
    CooTensor::new(shape2(2, 3), indices, values);
}

// ── CSR ───────────────────────────────────────────────────────────────────

fn csr_2x3() -> CsrTensor<f32, Seq> {
    let values = Tensor::<f32, Seq>::from_slice(vec![3], &[1.0, 2.0, 3.0]);
    let col_indices = Tensor::<i64, Seq>::from_slice(vec![3], &[0, 2, 1]);
    let row_offsets = Tensor::<i64, Seq>::from_slice(vec![3], &[0, 2, 3]);
    CsrTensor::new(shape2(2, 3), values, col_indices, row_offsets)
}

#[test]
fn csr_construction_and_accessors() {
    let csr = csr_2x3();
    assert_eq!(&csr.shape()[..], &[2usize, 3]);
    assert_eq!(csr.nnz(), 3);
    assert_eq!(csr.values().as_slice(), &[1.0, 2.0, 3.0]);
    assert_eq!(csr.col_indices().as_slice(), &[0, 2, 1]);
    assert_eq!(csr.row_offsets().as_slice(), &[0, 2, 3]);
}

#[test]
fn csr_clone_preserves_structure() {
    let csr = csr_2x3();
    let cloned = csr.clone();
    assert_eq!(cloned.values().as_slice(), csr.values().as_slice());
    assert_eq!(
        cloned.col_indices().as_slice(),
        csr.col_indices().as_slice()
    );
    assert_eq!(
        cloned.row_offsets().as_slice(),
        csr.row_offsets().as_slice()
    );
}

#[test]
#[should_panic(expected = "CSR format is restricted to 2D matrices")]
fn csr_rejects_non_2d_shape() {
    let values = Tensor::<f32, Seq>::from_slice(vec![3], &[1.0, 2.0, 3.0]);
    let col_indices = Tensor::<i64, Seq>::from_slice(vec![3], &[0, 2, 1]);
    let row_offsets = Tensor::<i64, Seq>::from_slice(vec![3], &[0, 2, 3]);
    CsrTensor::new(Shape::from(vec![2, 2, 3]), values, col_indices, row_offsets);
}

#[test]
#[should_panic(expected = "col_indices length must match values count")]
fn csr_rejects_col_indices_mismatch() {
    let values = Tensor::<f32, Seq>::from_slice(vec![3], &[1.0, 2.0, 3.0]);
    let col_indices = Tensor::<i64, Seq>::from_slice(vec![2], &[0, 2]);
    let row_offsets = Tensor::<i64, Seq>::from_slice(vec![3], &[0, 2, 3]);
    CsrTensor::new(shape2(2, 3), values, col_indices, row_offsets);
}

#[test]
#[should_panic(expected = "row_offsets length must equal rows + 1")]
fn csr_rejects_row_offsets_mismatch() {
    let values = Tensor::<f32, Seq>::from_slice(vec![3], &[1.0, 2.0, 3.0]);
    let col_indices = Tensor::<i64, Seq>::from_slice(vec![3], &[0, 2, 1]);
    // 2 rows requires row_offsets length 3; supply 2.
    let row_offsets = Tensor::<i64, Seq>::from_slice(vec![2], &[0, 3]);
    CsrTensor::new(shape2(2, 3), values, col_indices, row_offsets);
}
