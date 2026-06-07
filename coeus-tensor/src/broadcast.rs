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
    let max_len = a.len().max(b.len());
    let mut out = Shape::new();
    out.0.resize(max_len, 0);

    for i in 0..max_len {
        let dim_a = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let dim_b = if i < b.len() { b[b.len() - 1 - i] } else { 1 };

        if dim_a == dim_b {
            out.0[max_len - 1 - i] = dim_a;
        } else if dim_a == 1 {
            out.0[max_len - 1 - i] = dim_b;
        } else if dim_b == 1 {
            out.0[max_len - 1 - i] = dim_a;
        } else {
            return None;
        }
    }

    Some(out)
}

#[cfg(test)]
mod tests {
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
