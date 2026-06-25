use leto::{ArrayView1, ArrayView2, Layout as LetoLayout, Result};
use leto_ops::Scalar as LetoScalar;
use leto_ops::{spmm_into as leto_sparse_spmm_into, spmv_into as leto_sparse_spmv_into, CsrMatrix};

/// Borrowed CSR matrix descriptor used at the coeus-to-leto dispatch boundary.
#[derive(Clone, Copy, Debug)]
pub struct CsrDispatch<'a, T> {
    /// Nonzero values in row-major CSR order.
    pub values: &'a [T],
    /// Column index for each nonzero value.
    pub col_indices: &'a [usize],
    /// Row pointer offsets with length `nrows + 1`.
    pub row_ptr: &'a [usize],
    /// Number of matrix rows.
    pub nrows: usize,
    /// Number of matrix columns.
    pub ncols: usize,
}

impl<'a, T> CsrDispatch<'a, T> {
    /// Creates a borrowed CSR descriptor without copying the underlying slices.
    pub const fn new(
        values: &'a [T],
        col_indices: &'a [usize],
        row_ptr: &'a [usize],
        nrows: usize,
        ncols: usize,
    ) -> Self {
        Self {
            values,
            col_indices,
            row_ptr,
            nrows,
            ncols,
        }
    }
}

/// Sparse matrix-vector product: `y = A x`, routed through leto-ops sparse
/// kernels after rebuilding leto's validated CSR representation from coeus's
/// borrowed CSR slices.
pub fn spmv_into<T: LetoScalar>(a: CsrDispatch<'_, T>, x: &[T], y_out: &mut [T]) -> Result<()> {
    let ncols = a.ncols;
    let a = CsrMatrix::from_parts(
        a.values.to_vec(),
        a.col_indices.to_vec(),
        a.row_ptr.to_vec(),
        a.nrows,
        ncols,
    )?;
    let x_view = ArrayView1::try_new(LetoLayout::c_contiguous([ncols])?, x)?;
    leto_sparse_spmv_into(&a, &x_view, y_out)
}

/// Sparse-dense matrix multiply: `C = A B`, routed through leto-ops sparse
/// kernels after rebuilding leto's validated CSR representation from coeus's
/// borrowed CSR slices.
pub fn spmm_into<T: LetoScalar>(
    a: CsrDispatch<'_, T>,
    b: &[T],
    b_cols: usize,
    c_out: &mut [T],
) -> Result<()> {
    let ncols = a.ncols;
    let a = CsrMatrix::from_parts(
        a.values.to_vec(),
        a.col_indices.to_vec(),
        a.row_ptr.to_vec(),
        a.nrows,
        ncols,
    )?;
    let b_view = ArrayView2::try_new(LetoLayout::c_contiguous([ncols, b_cols])?, b)?;
    leto_sparse_spmm_into(&a, &b_view, c_out)
}
