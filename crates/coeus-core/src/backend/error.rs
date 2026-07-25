use thiserror::Error;

/// Failure categories shared by CPU-backed operation implementations.
///
/// Backend-specific crates define richer errors for device dispatch. This
/// foundation error keeps the CPU operation seam typed without making
/// `coeus-core` depend on a provider crate.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BackendError {
    /// The operation does not have a monomorphized kernel for this rank.
    #[error("{operation} does not support layout rank {rank}; maximum rank is {max_rank}")]
    UnsupportedRank {
        /// Operation family that rejected the rank.
        operation: &'static str,
        /// Requested layout rank.
        rank: usize,
        /// Largest supported rank.
        max_rank: usize,
    },
    /// Two layout descriptors have different ranks.
    #[error("{operation} layout rank mismatch: lhs {lhs}, rhs {rhs}")]
    LayoutRankMismatch {
        /// Operation family that rejected the layouts.
        operation: &'static str,
        /// Left-hand layout rank.
        lhs: usize,
        /// Right-hand layout rank.
        rhs: usize,
    },
    /// The operation received incompatible shapes.
    #[error("{operation} shape mismatch: lhs {lhs:?}, rhs {rhs:?}")]
    ShapeMismatch {
        /// Operation family that rejected the shapes.
        operation: &'static str,
        /// Left-hand shape.
        lhs: Vec<usize>,
        /// Right-hand shape.
        rhs: Vec<usize>,
    },
    /// The operation received an axis outside the layout rank.
    #[error("{operation} axis {axis} is out of bounds for rank {rank}")]
    AxisOutOfRange {
        /// Operation family that rejected the axis.
        operation: &'static str,
        /// Requested axis.
        axis: usize,
        /// Number of dimensions in the input layout.
        rank: usize,
    },
    /// The operation received shapes that cannot be broadcast.
    #[error("{operation} incompatible broadcast: {from:?} to {to:?}")]
    IncompatibleBroadcast {
        /// Operation family that rejected the broadcast.
        operation: &'static str,
        /// Source shape.
        from: Vec<usize>,
        /// Requested target shape.
        to: Vec<usize>,
    },
    /// Layout arithmetic exceeded the representable range.
    #[error("{operation} layout arithmetic overflow: {reason}")]
    Overflow {
        /// Operation family that detected the overflow.
        operation: &'static str,
        /// Provider-reported overflow location.
        reason: &'static str,
    },
    /// Storage metadata or buffer lengths violate the operation contract.
    #[error("{operation} storage error: {reason}")]
    Storage {
        /// Operation family that rejected storage metadata.
        operation: &'static str,
        /// Provider-reported storage detail.
        reason: String,
    },
}
