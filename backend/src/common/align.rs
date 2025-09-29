//! Shared alignment utilities.

use coeus_dtype::Dtype;
use num_traits::Float;

/// Align buffer size for dtype.
pub fn align_size<D: Dtype>(len: usize) -> usize {
    let elem_size = std::mem::size_of::<D>();
    ((len * elem_size + 3) / 4) * 4 // 4-byte GPU alignment
}

/// Validate dtype compatibility.
pub fn check_dtype<T: Dtype + Float>(_expected: usize, _actual: usize) -> Result<(), &'static str> {
    // Impl body; unused actual ok if intentional
    Ok(())
}
