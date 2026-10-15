//! The existing usize-index bridge preserves Leto's canonical-CSR validation
//! and scalar/SIMD arithmetic, while borrowing all three arrays. Owned and
//! borrowed entry points execute the same provider kernel, so this change is
//! bitwise equivalent even when SIMD uses fused multiply-add.
//!
//! This descriptor differs from Coeus's signed-index tensor contract: Coeus also
//! accepts unsorted/duplicate columns and performs separate multiply/add.
//! Replacing its backend-parallel forward kernels requires resolving those
//! provider contracts first; see `docs/verification/sparse-recovery.md`.

use leto::{ArrayView1, ArrayView2, Layout as LetoLayout, Result};
use leto_ops::Scalar as LetoScalar;
use leto_ops::{
    spmm_view_into as leto_sparse_spmm_into, spmv_view_into as leto_sparse_spmv_into, CsrView,
};

/// Borrowed CSR matrix descriptor used at the coeus-to-leto dispatch boundary.
///
/// # Examples
///
/// Build a descriptor for a 3x3 CSR matrix with 5 nonzeros and read back the
/// borrowed slices without copying:
///
/// ```
/// use coeus_leto::CsrDispatch;
///
/// let values = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let col_indices = [0, 2, 1, 0, 2];
/// let row_ptr = [0, 2, 3, 5];
/// let csr = CsrDispatch::new(&values, &col_indices, &row_ptr, 3, 3);
/// assert_eq!(csr.values, &values[..]);
/// assert_eq!(csr.nrows, 3);
/// assert_eq!(csr.ncols, 3);
/// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_leto::CsrDispatch;
    ///
    /// let values = [1.0_f64, 2.0, 3.0];
    /// let col_indices = [0, 1, 2];
    /// let row_ptr = [0, 1, 2, 3];
    /// let csr = CsrDispatch::new(&values, &col_indices, &row_ptr, 3, 3);
    /// assert_eq!(csr.row_ptr, &row_ptr[..]);
    /// ```
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
/// kernels through a validated borrowed view. All three CSR arrays remain
/// borrowed; construction performs no allocation or index reinterpretation.
///
/// # Examples
///
/// Multiply a 3x3 CSR matrix by a length-3 vector:
///
/// ```
/// use coeus_leto::{spmv_into, CsrDispatch};
///
/// let values = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let col_indices = [0, 2, 1, 0, 2];
/// let row_ptr = [0, 2, 3, 5];
/// let csr = CsrDispatch::new(&values, &col_indices, &row_ptr, 3, 3);
/// let x = [1.0_f64, 2.0, 3.0];
/// let mut y = [0.0_f64; 3];
/// spmv_into(csr, &x, &mut y).unwrap();
/// assert_eq!(y, [7.0, 6.0, 19.0]);
/// ```
pub fn spmv_into<T: LetoScalar>(a: CsrDispatch<'_, T>, x: &[T], y_out: &mut [T]) -> Result<()> {
    let ncols = a.ncols;
    let a = CsrView::from_slices(a.values, a.col_indices, a.row_ptr, a.nrows, ncols)?;
    let x_view = ArrayView1::try_new(LetoLayout::c_contiguous([ncols])?, x)?;
    leto_sparse_spmv_into(&a, &x_view, y_out)
}

/// Sparse-dense matrix multiply: `C = A B`, routed through leto-ops sparse
/// kernels through a validated borrowed view. Stored-entry accumulation order
/// and Leto's scalar/SIMD policy are identical to the owned-matrix entry point.
///
/// # Examples
///
/// Multiply a 3x3 CSR matrix by a 3x2 dense matrix into a 3x2 result:
///
/// ```
/// use coeus_leto::{spmm_into, CsrDispatch};
///
/// let values = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let col_indices = [0, 2, 1, 0, 2];
/// let row_ptr = [0, 2, 3, 5];
/// let csr = CsrDispatch::new(&values, &col_indices, &row_ptr, 3, 3);
/// let b = [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2, row-major
/// let mut c = [0.0_f64; 6];
/// spmm_into(csr, &b, 2, &mut c).unwrap();
/// assert_eq!(c, [11.0, 14.0, 9.0, 12.0, 29.0, 38.0]);
/// ```
pub fn spmm_into<T: LetoScalar>(
    a: CsrDispatch<'_, T>,
    b: &[T],
    b_cols: usize,
    c_out: &mut [T],
) -> Result<()> {
    let ncols = a.ncols;
    let a = CsrView::from_slices(a.values, a.col_indices, a.row_ptr, a.nrows, ncols)?;
    let b_view = ArrayView2::try_new(LetoLayout::c_contiguous([ncols, b_cols])?, b)?;
    leto_sparse_spmm_into(&a, &b_view, c_out)
}
