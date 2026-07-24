use coeus_leto::{spmm_into, spmv_into, CsrDispatch};
use leto::{ArrayView1, ArrayView2, Layout};
use leto_ops::{spmm_into as leto_spmm_into, spmv_into as leto_spmv_into, CsrMatrix};

fn sample_csr() -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    (
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![0, 2, 1, 0, 2],
        vec![0, 2, 3, 5],
    )
}

#[test]
fn spmv_dispatch_matches_direct_output() {
    let (values, col_indices, row_ptr) = sample_csr();
    let csr =
        CsrMatrix::from_parts(values.clone(), col_indices.clone(), row_ptr.clone(), 3, 3).unwrap();
    let x = vec![1.0, 2.0, 3.0];
    let x_view = ArrayView1::try_new(Layout::c_contiguous([3]).unwrap(), &x).unwrap();
    let mut expected = vec![0.0; 3];
    let mut actual = vec![0.0; 3];
    let dispatch = CsrDispatch::new(&values, &col_indices, &row_ptr, 3, 3);

    leto_spmv_into(&csr, &x_view, &mut expected).unwrap();
    spmv_into(dispatch, &x, &mut actual).unwrap();

    assert_eq!(actual, expected);
    assert_eq!(actual, vec![7.0, 6.0, 19.0]);
}

#[test]
fn spmm_dispatch_matches_direct_output() {
    let (values, col_indices, row_ptr) = sample_csr();
    let csr =
        CsrMatrix::from_parts(values.clone(), col_indices.clone(), row_ptr.clone(), 3, 3).unwrap();
    let b = vec![
        1.0, 2.0, //
        3.0, 4.0, //
        5.0, 6.0,
    ];
    let b_view = ArrayView2::try_new(Layout::c_contiguous([3, 2]).unwrap(), &b).unwrap();
    let mut expected = vec![0.0; 6];
    let mut actual = vec![0.0; 6];
    let dispatch = CsrDispatch::new(&values, &col_indices, &row_ptr, 3, 3);

    leto_spmm_into(&csr, &b_view, &mut expected).unwrap();
    spmm_into(dispatch, &b, 2, &mut actual).unwrap();

    assert_eq!(actual, expected);
    assert_eq!(actual, vec![11.0, 14.0, 9.0, 12.0, 29.0, 38.0]);
}
