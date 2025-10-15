//! Iterator utilities for broadcasted tensor operations.
//!
//! Provides stride-based iteration enabling zero-copy broadcasting.

/// Computes linear index from multi-dimensional index using strides.
///
/// # Arguments
///
/// * `linear_idx` - Linear index in the output tensor
/// * `shape` - Shape of the output tensor
/// * `strides` - Strides for the input tensor (with 0 for broadcast dims)
///
/// # Returns
///
/// Linear index into the input tensor's data array.
///
/// # Examples
///
/// ```
/// use coeus_storage::iter::compute_strided_index;
///
/// // 2D tensor [3, 4] with row-major strides [4, 1]
/// // Linear index 5 -> multi-index [1, 1] -> strided index 5
/// let idx = compute_strided_index(5, &[3, 4], &[4, 1]);
/// assert_eq!(idx, 5);
///
/// // Broadcasting: [1, 4] broadcast to [3, 4] with strides [0, 1]
/// // Linear index 5 (multi-index [1,1]) -> strided index 1 (row 0, col 1)
/// let idx = compute_strided_index(5, &[3, 4], &[0, 1]);
/// assert_eq!(idx, 1);  // Same as linear index 1 in original [1,4] tensor
/// ```
#[inline]
#[must_use]
pub fn compute_strided_index(linear_idx: usize, shape: &[usize], strides: &[usize]) -> usize {
    let ndim = shape.len();
    let mut result = 0;
    let mut remaining = linear_idx;

    for i in 0..ndim {
        let dim_stride = strides[i];

        // Compute multi-dimensional index for this dimension
        let multi_idx = remaining / shape[(i + 1)..].iter().product::<usize>().max(1);
        remaining %= shape[(i + 1)..].iter().product::<usize>().max(1);

        // Apply stride (0 stride means broadcast, always use index 0)
        result += multi_idx * dim_stride;
    }

    result
}

/// Computes strided index more efficiently for common case.
///
/// Optimized version that avoids repeated shape product calculations.
#[inline]
#[must_use]
pub fn compute_strided_index_fast(linear_idx: usize, shape: &[usize], strides: &[usize]) -> usize {
    let ndim = shape.len();

    if ndim == 0 {
        return 0; // Scalar
    }

    if ndim == 1 {
        return (linear_idx % shape[0]) * strides[0];
    }

    if ndim == 2 {
        let row = linear_idx / shape[1];
        let col = linear_idx % shape[1];
        return row * strides[0] + col * strides[1];
    }

    // General case for ndim > 2
    let mut result = 0;
    let mut idx = linear_idx;

    for i in (0..ndim).rev() {
        let dim_idx = idx % shape[i];
        idx /= shape[i];
        result += dim_idx * strides[i];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strided_index_no_broadcast() {
        // [3, 4] with row-major strides [4, 1]
        assert_eq!(compute_strided_index_fast(0, &[3, 4], &[4, 1]), 0); // [0,0]
        assert_eq!(compute_strided_index_fast(1, &[3, 4], &[4, 1]), 1); // [0,1]
        assert_eq!(compute_strided_index_fast(4, &[3, 4], &[4, 1]), 4); // [1,0]
        assert_eq!(compute_strided_index_fast(5, &[3, 4], &[4, 1]), 5); // [1,1]
    }

    #[test]
    fn test_strided_index_broadcast_row() {
        // [1, 4] broadcast to [3, 4] with strides [0, 1]
        // All elements in first column should map to index 0
        assert_eq!(compute_strided_index_fast(0, &[3, 4], &[0, 1]), 0); // [0,0] -> 0
        assert_eq!(compute_strided_index_fast(4, &[3, 4], &[0, 1]), 0); // [1,0] -> 0
        assert_eq!(compute_strided_index_fast(8, &[3, 4], &[0, 1]), 0); // [2,0] -> 0

        // Second column should map to index 1
        assert_eq!(compute_strided_index_fast(1, &[3, 4], &[0, 1]), 1); // [0,1] -> 1
        assert_eq!(compute_strided_index_fast(5, &[3, 4], &[0, 1]), 1); // [1,1] -> 1
    }

    #[test]
    fn test_strided_index_broadcast_col() {
        // [3, 1] broadcast to [3, 4] with strides [1, 0]
        // First row, all columns map to index 0
        assert_eq!(compute_strided_index_fast(0, &[3, 4], &[1, 0]), 0); // [0,0] -> 0
        assert_eq!(compute_strided_index_fast(1, &[3, 4], &[1, 0]), 0); // [0,1] -> 0
        assert_eq!(compute_strided_index_fast(2, &[3, 4], &[1, 0]), 0); // [0,2] -> 0

        // Second row maps to index 1
        assert_eq!(compute_strided_index_fast(4, &[3, 4], &[1, 0]), 1); // [1,0] -> 1
        assert_eq!(compute_strided_index_fast(5, &[3, 4], &[1, 0]), 1); // [1,1] -> 1
    }

    #[test]
    fn test_strided_index_scalar() {
        // Scalar [] broadcast to [3, 4] with strides [0, 0]
        assert_eq!(compute_strided_index_fast(0, &[3, 4], &[0, 0]), 0);
        assert_eq!(compute_strided_index_fast(5, &[3, 4], &[0, 0]), 0);
        assert_eq!(compute_strided_index_fast(11, &[3, 4], &[0, 0]), 0);
    }

    #[test]
    fn test_strided_index_1d() {
        // 1D tensor [5] with stride [1]
        assert_eq!(compute_strided_index_fast(0, &[5], &[1]), 0);
        assert_eq!(compute_strided_index_fast(3, &[5], &[1]), 3);
        assert_eq!(compute_strided_index_fast(4, &[5], &[1]), 4);
    }
}
