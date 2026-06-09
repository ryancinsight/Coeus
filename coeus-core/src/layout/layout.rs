// ── Layout descriptor ──
// Maps logical tensor indices to physical memory offsets.

use crate::layout::shape::Shape;
use crate::layout::strides::{is_contiguous, row_major_strides, Strides};

/// Multi-dimensional layout descriptor.
///
/// Owns shape, pre-computed strides, and an optional base offset
/// for views/slices into shared storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub(crate) shape: Shape,
    pub(crate) strides: Strides,
    pub(crate) offset: usize,
}

impl Layout {
    /// Create contiguous row-major layout from shape.
    #[inline]
    pub fn new(shape: Shape) -> Self {
        let strides = row_major_strides(&shape);
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    /// Create from shape and explicit strides (for views).
    #[inline]
    pub fn from_shape_strides(shape: Shape, strides: Strides, offset: usize) -> Self {
        Self {
            shape,
            strides,
            offset,
        }
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of logical elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Borrow shape slice.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Clone shape.
    #[inline]
    pub fn shape_cloned(&self) -> Shape {
        self.shape.clone()
    }

    /// Borrow strides slice.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Clone strides.
    #[inline]
    pub fn strides_cloned(&self) -> Strides {
        self.strides.clone()
    }

    /// Base physical offset.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// True if row-major contiguous.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        is_contiguous(&self.shape, &self.strides)
    }

    /// Compute 1D physical offset from multi-dimensional index.
    ///
    /// # Panics
    /// In debug: index length mismatch or out-of-bounds.
    #[inline]
    pub fn physical_index(&self, index: &[usize]) -> usize {
        debug_assert_eq!(index.len(), self.shape.len(), "index ndim mismatch");
        let mut off = self.offset;
        for (i, &idx) in index.iter().enumerate() {
            debug_assert!(
                idx < self.shape[i],
                "index[{i}]={idx} out of bounds (dim={})",
                self.shape[i]
            );
            off += idx * self.strides[i];
        }
        off
    }

    /// Create a zero-copy sliced layout.
    ///
    /// `ranges` is a slice of `(start, end)` pairs, one per dimension.
    #[inline]
    pub fn slice(&self, ranges: &[(usize, usize)]) -> Self {
        assert_eq!(ranges.len(), self.shape.len(), "ranges.len() != ndim");
        let mut shape = Shape::with_capacity(self.ndim());
        let mut strides = Strides::with_capacity(self.ndim());
        let mut offset = self.offset;

        for (i, &(start, end)) in ranges.iter().enumerate() {
            assert!(
                start <= end && end <= self.shape[i],
                "slice [{start}..{end}) out of dim {i}={}",
                self.shape[i]
            );
            shape.push(end - start);
            strides.push(self.strides[i]);
            offset += start * self.strides[i];
        }

        Self {
            shape,
            strides,
            offset,
        }
    }

    /// Create a zero-copy unsqueezed layout by inserting a dimension of size 1 at `axis`.
    #[inline]
    pub fn unsqueeze(&self, axis: usize) -> Self {
        let ndim = self.ndim();
        assert!(
            axis <= ndim,
            "unsqueeze: axis {axis} out of bounds for ndim {ndim}"
        );

        let mut shape = Shape::with_capacity(ndim + 1);
        let mut strides = Strides::with_capacity(ndim + 1);

        for i in 0..axis {
            shape.push(self.shape[i]);
            strides.push(self.strides[i]);
        }

        shape.push(1);
        let stride_val = if axis < ndim {
            self.strides[axis]
        } else if ndim > 0 {
            self.strides[ndim - 1]
        } else {
            1
        };
        strides.push(stride_val);

        for i in axis..ndim {
            shape.push(self.shape[i]);
            strides.push(self.strides[i]);
        }

        Self {
            shape,
            strides,
            offset: self.offset,
        }
    }

    /// Create a zero-copy squeezed layout by removing the dimension of size 1 at `axis`.
    #[inline]
    pub fn squeeze(&self, axis: usize) -> Self {
        let ndim = self.ndim();
        assert!(
            axis < ndim,
            "squeeze: axis {axis} out of bounds for ndim {ndim}"
        );
        assert_eq!(
            self.shape[axis], 1,
            "squeeze: axis {axis} has size {}, expected 1",
            self.shape[axis]
        );

        let mut shape = Shape::with_capacity(ndim - 1);
        let mut strides = Strides::with_capacity(ndim - 1);

        for i in 0..ndim {
            if i != axis {
                shape.push(self.shape[i]);
                strides.push(self.strides[i]);
            }
        }

        Self {
            shape,
            strides,
            offset: self.offset,
        }
    }

    /// Create a zero-copy squeezed layout by removing all dimensions of size 1.
    #[inline]
    pub fn squeeze_all(&self) -> Self {
        let ndim = self.ndim();
        let mut shape = Shape::with_capacity(ndim);
        let mut strides = Strides::with_capacity(ndim);

        for i in 0..ndim {
            if self.shape[i] != 1 {
                shape.push(self.shape[i]);
                strides.push(self.strides[i]);
            }
        }

        Self {
            shape,
            strides,
            offset: self.offset,
        }
    }
}

/// Compile-time layout descriptor parameterized by const arity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstLayout<const DIMS: usize> {
    pub shape: crate::layout::ConstShape<DIMS>,
    pub strides: [usize; DIMS],
    pub offset: usize,
}

impl<const DIMS: usize> ConstLayout<DIMS> {
    /// Compute a contiguous row-major layout at compile-time.
    #[inline]
    pub const fn new(shape: crate::layout::ConstShape<DIMS>) -> Self {
        let mut strides = [0; DIMS];
        if DIMS > 0 {
            strides[DIMS - 1] = 1;
            let mut i = DIMS - 1;
            while i > 0 {
                i -= 1;
                strides[i] = strides[i + 1] * shape.dims[i + 1];
            }
        }
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    /// Query dimensional contiguity at compile-time.
    #[inline]
    pub const fn is_contiguous(&self) -> bool {
        let mut expected = 1;
        let mut i = DIMS;
        while i > 0 {
            i -= 1;
            let dim = self.shape.dims[i];
            let s = self.strides[i];
            if dim > 1 && s != expected {
                return false;
            }
            expected *= dim;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::ConstShape;

    #[test]
    fn test_const_layout() {
        const SHAPE: ConstShape<3> = ConstShape::new([2, 3, 4]);
        const LAYOUT: ConstLayout<3> = ConstLayout::new(SHAPE);

        // Verify shape
        assert_eq!(LAYOUT.shape.dims, [2, 3, 4]);

        // Verify calculated strides
        assert_eq!(LAYOUT.strides, [12, 4, 1]);

        // Verify contiguity
        assert!(LAYOUT.is_contiguous());

        // Verify non-contiguous case
        const NON_CONTIG: ConstLayout<3> = ConstLayout {
            shape: ConstShape::new([2, 3, 4]),
            strides: [12, 5, 1], // not contiguous
            offset: 0,
        };
        assert!(!NON_CONTIG.is_contiguous());
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let l = Layout::new([2, 3].into());
        assert_eq!(l.shape(), &[2, 3]);
        assert_eq!(l.strides(), &[3, 1]);

        // Unsqueeze at 0 -> [1, 2, 3]
        let l2 = l.unsqueeze(0);
        assert_eq!(l2.shape(), &[1, 2, 3]);
        assert_eq!(l2.strides(), &[3, 3, 1]);

        // Unsqueeze at 1 -> [2, 1, 3]
        let l3 = l.unsqueeze(1);
        assert_eq!(l3.shape(), &[2, 1, 3]);
        assert_eq!(l3.strides(), &[3, 1, 1]);

        // Unsqueeze at 2 -> [2, 3, 1]
        let l4 = l.unsqueeze(2);
        assert_eq!(l4.shape(), &[2, 3, 1]);
        assert_eq!(l4.strides(), &[3, 1, 1]);

        // Squeeze axis 1 of l3 -> [2, 3]
        let l5 = l3.squeeze(1);
        assert_eq!(l5.shape(), &[2, 3]);
        assert_eq!(l5.strides(), &[3, 1]);

        // Squeeze all on a layout with multiple 1s: [1, 2, 1, 3] -> [2, 3]
        let l_multi =
            Layout::from_shape_strides([1, 2, 1, 3].into(), smallvec::smallvec![6, 3, 3, 1], 0);
        let l_squeezed = l_multi.squeeze_all();
        assert_eq!(l_squeezed.shape(), &[2, 3]);
        assert_eq!(l_squeezed.strides(), &[3, 1]);
    }
}
