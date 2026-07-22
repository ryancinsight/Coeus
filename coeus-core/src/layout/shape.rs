// ── Shape descriptor ──
// SmallVec-backed dimension array. Elides heap alloc for ndim ≤ 4.

use smallvec::SmallVec;
use std::ops::{Deref, DerefMut};

/// Shape type: inline for up to 4 dims, spills to heap beyond.
///
/// # Examples
///
/// Construct from an array, iterate dimensions, and compute the product:
///
/// ```
/// use coeus_core::Shape;
///
/// let shape: Shape = [2, 3, 4].into();
/// assert_eq!(shape.len(), 3); // Shape derefs to &[usize]
/// assert_eq!(shape.iter().product::<usize>(), 24);
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape(pub SmallVec<[usize; 4]>);

/// Const-generic shape marker (ZST).
///
/// Used to statically encode shapes in type signatures,
/// enabling monomorphized dispatch and zero-cost dimension checks.
///
/// # Examples
///
/// ```
/// use coeus_core::ConstShape;
///
/// const SHAPE: ConstShape<3> = ConstShape::new([2, 3, 4]);
/// assert_eq!(SHAPE.numel(), 24);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConstShape<const DIMS: usize> {
    /// Dimension values.
    pub dims: [usize; DIMS],
}

impl<const DIMS: usize> ConstShape<DIMS> {
    /// Create from array.
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_core::ConstShape;
    ///
    /// const S: ConstShape<2> = ConstShape::new([3, 4]);
    /// assert_eq!(S.dims, [3, 4]);
    /// ```
    #[inline]
    pub const fn new(dims: [usize; DIMS]) -> Self {
        Self { dims }
    }

    /// Number of elements (product of all dimensions).
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_core::ConstShape;
    ///
    /// const S: ConstShape<3> = ConstShape::new([2, 3, 4]);
    /// assert_eq!(S.numel(), 24);
    /// ```
    #[inline]
    pub const fn numel(&self) -> usize {
        let mut n = 1;
        let mut i = 0;
        while i < DIMS {
            n *= self.dims[i];
            i += 1;
        }
        n
    }

    /// Convert to dynamic [`Shape`].
    ///
    /// # Examples
    ///
    /// ```
    /// use coeus_core::ConstShape;
    ///
    /// let s = ConstShape::new([2, 3]);
    /// let dyn_shape = s.to_shape();
    /// assert_eq!(&*dyn_shape, &[2, 3]);
    /// ```
    #[inline]
    pub fn to_shape(&self) -> Shape {
        let mut s = Shape::new();
        for &d in &self.dims {
            s.push(d);
        }
        s
    }
}

impl Shape {
    /// Compute a new empty shape.
    #[inline]
    pub fn new() -> Self {
        Self(SmallVec::new())
    }

    /// Compute a new shape with capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(SmallVec::with_capacity(capacity))
    }

    /// Push a dimension.
    #[inline]
    pub fn push(&mut self, val: usize) {
        self.0.push(val);
    }
}

impl Deref for Shape {
    type Target = [usize];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Shape {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Vec<usize>> for Shape {
    #[inline]
    fn from(v: Vec<usize>) -> Self {
        Self(SmallVec::from_vec(v))
    }
}

impl From<&[usize]> for Shape {
    #[inline]
    fn from(s: &[usize]) -> Self {
        Self(SmallVec::from_slice(s))
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    #[inline]
    fn from(arr: [usize; N]) -> Self {
        Self(SmallVec::from_slice(&arr))
    }
}

impl std::iter::FromIterator<usize> for Shape {
    #[inline]
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Self(SmallVec::from_iter(iter))
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Shape({:?})", self.0)
    }
}
