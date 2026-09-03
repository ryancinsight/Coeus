//! The spatial-window geometry both window families are planned over.
//!
//! Pooling and unfold/fold take the same four per-axis parameters, and every
//! backend that implements those seams needs them. Keeping the type here — in
//! the crate that owns both seams and that every backend already depends on —
//! is what stops a second copy appearing per backend.

/// The kernel geometry of one spatial-window operation, per spatial axis.
///
/// `S` is the number of spatial axes: 2 for image pooling, 3 for volumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowConfiguration<const S: usize> {
    /// Kernel extent along each spatial axis.
    pub kernel: [usize; S],
    /// Step between consecutive window placements along each spatial axis.
    pub stride: [usize; S],
    /// Implicit padding added at both ends of each spatial axis.
    pub padding: [usize; S],
    /// Spacing between kernel taps along each spatial axis.
    pub dilation: [usize; S],
}
