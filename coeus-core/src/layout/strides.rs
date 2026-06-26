// ── Strides computation ──
// Row-major (C-contiguous) stride derivation.

use smallvec::SmallVec;

/// Strides type: inline for up to 4 dims.
pub type Strides = SmallVec<[usize; 4]>;

/// Compute row-major (C order) strides from shape.
///
/// For shape [d0, d1, d2], returns [d1*d2, d2, 1].
///
/// # Examples
///
/// ```
/// use coeus_core::layout::row_major_strides;
///
/// let strides = row_major_strides(&[2, 3, 4]);
/// assert_eq!(strides.as_slice(), &[12, 4, 1]);
/// ```
#[inline]
pub fn row_major_strides(shape: &[usize]) -> Strides {
    let ndim = shape.len();
    if ndim == 0 {
        return Strides::new();
    }
    let mut strides = Strides::from_elem(0, ndim);
    strides[ndim - 1] = 1;
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Check if strides represent contiguous row-major layout for given shape.
///
/// # Examples
///
/// ```
/// use coeus_core::layout::{is_contiguous, row_major_strides};
///
/// let shape = [2, 3, 4];
/// let strides = row_major_strides(&shape);
/// assert!(is_contiguous(&shape, &strides));
///
/// let non_contiguous = [12, 5, 1]; // gap in dim 1
/// assert!(!is_contiguous(&shape, &non_contiguous));
/// ```
#[inline]
pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    let mut expected = 1;
    for (&dim, &s) in shape.iter().rev().zip(strides.iter().rev()) {
        if dim > 1 && s != expected {
            return false;
        }
        expected *= dim;
    }
    true
}
