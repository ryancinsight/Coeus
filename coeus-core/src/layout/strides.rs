// ── Strides computation ──
// Row-major (C-contiguous) stride derivation.

use smallvec::SmallVec;

/// Strides type: inline for up to 4 dims.
pub type Strides = SmallVec<[usize; 4]>;

/// Compute row-major (C order) strides from shape.
///
/// For shape [d0, d1, d2], returns [d1*d2, d2, 1].
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
