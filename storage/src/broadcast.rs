//! Broadcasting logic for tensor operations.
//!
//! Implements NumPy-compatible broadcasting rules for shape compatibility
//! and stride computation.
//!
//! # Broadcasting Rules
//!
//! Two shapes are compatible for broadcasting when:
//! 1. Dimensions are aligned from the rightmost position
//! 2. Each dimension pair is either:
//!    - Equal in size, OR
//!    - One of them is 1
//! 3. Missing dimensions are treated as size 1
//!
//! # Examples
//!
//! ```
//! use coeus_storage::broadcast::broadcast_shapes;
//!
//! // Compatible shapes
//! assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]).unwrap(), vec![3, 4]);
//! assert_eq!(broadcast_shapes(&[3, 4], &[4]).unwrap(), vec![3, 4]);
//!
//! // Incompatible shapes
//! assert!(broadcast_shapes(&[3, 4], &[5]).is_err());
//! ```

use crate::error::StorageError;
use crate::Result;
use alloc::vec::Vec;

/// Computes the broadcast shape from two input shapes.
///
/// Returns the output shape if shapes are broadcast-compatible,
/// otherwise returns an error.
///
/// # Algorithm
///
/// 1. Align shapes from rightmost dimension
/// 2. For each dimension pair:
///    - If equal: use that size
///    - If one is 1: use the other size
///    - Otherwise: incompatible
///
/// # Errors
///
/// Returns `StorageError::BroadcastError` if shapes are incompatible.
///
/// # Examples
///
/// ```
/// use coeus_storage::broadcast::broadcast_shapes;
///
/// // Scalar broadcast to vector
/// let result = broadcast_shapes(&[], &[5]).unwrap();
/// assert_eq!(result, vec![5]);
///
/// // 2D broadcast
/// let result = broadcast_shapes(&[3, 1], &[1, 4]).unwrap();
/// assert_eq!(result, vec![3, 4]);
///
/// // Right-aligned broadcast
/// let result = broadcast_shapes(&[5, 3, 4], &[4]).unwrap();
/// assert_eq!(result, vec![5, 3, 4]);
/// ```
pub fn broadcast_shapes(shape_a: &[usize], shape_b: &[usize]) -> Result<Vec<usize>> {
    let ndim_a = shape_a.len();
    let ndim_b = shape_b.len();
    let ndim_out = ndim_a.max(ndim_b);

    let mut result = Vec::with_capacity(ndim_out);

    for i in 0..ndim_out {
        // Index from the right (align rightmost dimensions)
        let idx_a = ndim_a.checked_sub(ndim_out - i);
        let idx_b = ndim_b.checked_sub(ndim_out - i);

        let dim_a = idx_a.map_or(1, |idx| shape_a[idx]);
        let dim_b = idx_b.map_or(1, |idx| shape_b[idx]);

        if dim_a == dim_b {
            result.push(dim_a);
        } else if dim_a == 1 {
            result.push(dim_b);
        } else if dim_b == 1 {
            result.push(dim_a);
        } else {
            return Err(StorageError::BroadcastError {
                shape_a: shape_a.to_vec(),
                shape_b: shape_b.to_vec(),
                dimension: i,
            });
        }
    }

    Ok(result)
}

/// Computes broadcast strides for a given shape to match a target broadcast shape.
///
/// Returns strides where dimensions that are broadcast (size 1 expanded to size > 1)
/// have stride 0, enabling zero-copy broadcasting via repeated indexing.
///
/// # Arguments
///
/// * `original_shape` - The original tensor shape
/// * `broadcast_shape` - The target broadcast shape (output of `broadcast_shapes`)
/// * `original_strides` - The original tensor strides
///
/// # Examples
///
/// ```
/// use coeus_storage::broadcast::broadcast_strides;
///
/// // Shape [1, 4] with row-major strides [4, 1]
/// // Broadcast to [3, 4] should have strides [0, 1]
/// // (stride 0 on first dim means repeat same row)
/// let strides = broadcast_strides(&[1, 4], &[3, 4], &[4, 1]);
/// assert_eq!(strides, vec![0, 1]);
/// ```
#[must_use]
pub fn broadcast_strides(
    original_shape: &[usize],
    broadcast_shape: &[usize],
    original_strides: &[usize],
) -> Vec<usize> {
    let ndim_orig = original_shape.len();
    let ndim_out = broadcast_shape.len();

    broadcast_shape
        .iter()
        .enumerate()
        .map(|(i, &dim_out)| {
            let idx_orig = ndim_orig.checked_sub(ndim_out - i);

            if let Some(idx) = idx_orig {
                let dim_orig = original_shape[idx];

                if dim_orig == dim_out {
                    // No broadcasting on this dimension
                    original_strides[idx]
                } else if dim_orig == 1 {
                    // Broadcasting: stride becomes 0 (repeat same element)
                    0
                } else {
                    // This should never happen if broadcast_shape is valid
                    // but we handle it gracefully
                    original_strides[idx]
                }
            } else {
                // Missing dimension (treated as 1): stride is 0
                0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_broadcast_shapes_equal() {
        let result = broadcast_shapes(&[3, 4], &[3, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_scalar() {
        // Scalar (shape []) broadcasts to any shape
        let result = broadcast_shapes(&[], &[5]).unwrap();
        assert_eq!(result, vec![5]);

        let result = broadcast_shapes(&[3, 4], &[]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_1d_to_2d() {
        // [4] broadcasts with [3, 4] -> [3, 4]
        let result = broadcast_shapes(&[4], &[3, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);

        let result = broadcast_shapes(&[3, 4], &[4]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_column_row() {
        // [3, 1] with [1, 4] -> [3, 4]
        let result = broadcast_shapes(&[3, 1], &[1, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_3d() {
        // [5, 1, 4] with [1, 3, 4] -> [5, 3, 4]
        let result = broadcast_shapes(&[5, 1, 4], &[1, 3, 4]).unwrap();
        assert_eq!(result, vec![5, 3, 4]);
    }

    #[test]
    fn test_broadcast_shapes_incompatible() {
        // [3, 4] with [5] -> error (4 != 5)
        assert!(broadcast_shapes(&[3, 4], &[5]).is_err());

        // [3, 4] with [3, 5] -> error
        assert!(broadcast_shapes(&[3, 4], &[3, 5]).is_err());
    }

    #[test]
    fn test_broadcast_strides_no_broadcast() {
        // Shape [3, 4] stays [3, 4], strides unchanged
        let strides = broadcast_strides(&[3, 4], &[3, 4], &[4, 1]);
        assert_eq!(strides, vec![4, 1]);
    }

    #[test]
    fn test_broadcast_strides_repeat_row() {
        // [1, 4] -> [3, 4]: first dimension stride becomes 0
        let strides = broadcast_strides(&[1, 4], &[3, 4], &[4, 1]);
        assert_eq!(strides, vec![0, 1]);
    }

    #[test]
    fn test_broadcast_strides_repeat_column() {
        // [3, 1] -> [3, 4]: second dimension stride becomes 0
        let strides = broadcast_strides(&[3, 1], &[3, 4], &[1, 1]);
        assert_eq!(strides, vec![1, 0]);
    }

    #[test]
    fn test_broadcast_strides_scalar() {
        // [] -> [3, 4]: both strides are 0
        let strides = broadcast_strides(&[], &[3, 4], &[]);
        assert_eq!(strides, vec![0, 0]);
    }

    #[test]
    fn test_broadcast_strides_1d_to_2d() {
        // [4] -> [3, 4]: first stride is 0, second unchanged
        let strides = broadcast_strides(&[4], &[3, 4], &[1]);
        assert_eq!(strides, vec![0, 1]);
    }
}
