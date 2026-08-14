// ── Broadcast helpers ──
// Shape broadcasting logic for element-wise binary ops.

use coeus_core::Shape;

/// Compute the broadcast shape of two shapes.
///
/// Returns `None` if the shapes are not broadcast-compatible.
///
/// # Broadcasting rules (same as NumPy/PyTorch):
/// 1. Align shapes from the right.
/// 2. For each dimension, either dims are equal or one is 1.
/// 3. The output dim is max(a_dim, b_dim).
#[inline]
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Shape> {
    coeus_leto::broadcast_shape(a, b).ok().map(Shape::from)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "ratchet COEUS-UNWRAP-1")]
    use super::*;

    #[test]
    fn test_broadcast_same() {
        let s = broadcast_shapes(&[3, 4], &[3, 4]).unwrap();
        assert_eq!(s.as_ref(), &[3, 4]);
    }

    #[test]
    fn test_broadcast_scalar() {
        let s = broadcast_shapes(&[3, 4], &[1, 1]).unwrap();
        assert_eq!(s.as_ref(), &[3, 4]);
    }

    #[test]
    fn test_broadcast_vector() {
        let s = broadcast_shapes(&[3, 1], &[1, 4]).unwrap();
        assert_eq!(s.as_ref(), &[3, 4]);
    }

    #[test]
    fn test_broadcast_incompatible() {
        assert!(broadcast_shapes(&[3, 4], &[5, 6]).is_none());
    }
}
