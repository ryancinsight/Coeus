use thiserror::Error;

use crate::ReductionOp;

/// Failure categories shared by CPU-backed operation implementations.
///
/// Backend-specific crates define richer errors for device dispatch. This
/// foundation error keeps the CPU operation seam typed without making
/// `coeus-core` depend on a provider crate.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BackendError {
    /// Sequence length arrays do not each describe every batch sample.
    #[error(
        "sequence length counts must equal batch {batch}, got inputs {inputs}, targets {targets}"
    )]
    SequenceLengthCounts {
        /// Number of samples.
        batch: usize,
        /// Number of input-length entries.
        inputs: usize,
        /// Number of target-length entries.
        targets: usize,
    },
    /// A sample requests more input frames than storage contains.
    #[error("sample {sample} input length {actual} exceeds {maximum}")]
    SequenceInputLength {
        /// Affected sample.
        sample: usize,
        /// Requested frame count.
        actual: usize,
        /// Available frame count.
        maximum: usize,
    },
    /// A temporal alignment label is out of range or equals the blank label.
    #[error("invalid sequence label {label} with blank {blank} and {classes} classes")]
    SequenceLabel {
        /// Rejected class index.
        label: usize,
        /// Blank class index.
        blank: usize,
        /// Alphabet size.
        classes: usize,
    },
    /// Temporal alignment state dimensions overflow addressable storage.
    #[error("sequence state size overflows for {frames} frames and {targets} targets")]
    SequenceStateOverflow {
        /// Input frame count.
        frames: usize,
        /// Target count or accumulated state extent.
        targets: usize,
    },
    /// A provider cannot allocate its operation state.
    #[error("{operation} allocation failed")]
    Allocation {
        /// Operation requiring storage.
        operation: &'static str,
        /// Original storage-reservation failure.
        #[source]
        source: std::collections::TryReserveError,
    },
    /// A mutable layout maps different logical elements to the same storage.
    #[error("{operation} destination layout aliases logical elements")]
    AliasedLayout {
        /// Operation rejecting overlapping writes.
        operation: &'static str,
    },
    /// An active log-probability is positive, NaN, or positive infinity.
    #[error("{operation} invalid log-probability at {index:?}")]
    InvalidLogProbability {
        /// Operation rejecting the numeric input.
        operation: &'static str,
        /// Logical `[frame, sample, class]` coordinate.
        index: [usize; 3],
    },
    /// A normalization count is not finite in the selected scalar.
    #[error("{operation} normalization extent {extent} is not representable")]
    UnrepresentableExtent {
        /// Operation requiring scalar normalization.
        operation: &'static str,
        /// Rejected extent.
        extent: usize,
    },
    /// Arithmetic for one sample is not finite in the selected scalar.
    #[error("{operation} arithmetic is not finite for sample {sample}")]
    NonFiniteSample {
        /// Operation whose arithmetic failed.
        operation: &'static str,
        /// Affected sample.
        sample: usize,
    },
    /// A sample has no positive-probability alignment to differentiate.
    #[error("{operation} gradient is undefined for sample {sample}")]
    UndefinedGradient {
        /// Operation requiring a finite derivative.
        operation: &'static str,
        /// Affected sample.
        sample: usize,
    },
    /// A provider reports a failure not covered by this version's categories.
    #[error("{operation} provider failure: {reason}")]
    ProviderFailure {
        /// Operation reporting the failure.
        operation: &'static str,
        /// Unrecognized provider diagnostic.
        reason: String,
    },
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
    /// An operation requires a non-empty named dimension.
    #[error("{operation} requires a non-empty {dimension} dimension")]
    EmptyDimension {
        /// Operation family that rejected the dimension.
        operation: &'static str,
        /// Semantic dimension that must be non-empty.
        dimension: &'static str,
    },
    /// An index does not identify an element within its semantic bound.
    #[error("{operation} index {index} at position {position} is outside 0..{bound}")]
    IndexOutOfRange {
        /// Operation family that rejected the index.
        operation: &'static str,
        /// Position containing the invalid index.
        position: usize,
        /// Invalid index value.
        index: usize,
        /// Exclusive upper bound.
        bound: usize,
    },
    /// Numeric input violates an operation's finite-value contract.
    #[error("{operation} invalid numeric input: {reason}")]
    InvalidNumericInput {
        /// Operation family that rejected the value.
        operation: &'static str,
        /// Provider-preserved numeric failure detail.
        reason: String,
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
    /// A reduction without an identity received an empty axis.
    #[error("{operation} {reduction:?} is undefined for an empty axis")]
    EmptyReduction {
        /// Operation family that rejected the empty axis.
        operation: &'static str,
        /// Reduction whose result is undefined without an input value.
        reduction: ReductionOp,
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

#[cfg(test)]
mod tests {
    use super::BackendError;
    use std::error::Error;

    #[test]
    fn allocation_retains_reservation_cause() {
        let source = Vec::<u8>::new()
            .try_reserve(usize::MAX)
            .expect_err("invariant: a byte vector cannot exceed isize::MAX bytes");
        let error = BackendError::Allocation {
            operation: "ctc_forward",
            source: source.clone(),
        };
        assert_eq!(error.to_string(), "ctc_forward allocation failed");
        let cause = error
            .source()
            .expect("invariant: allocation errors preserve their cause");
        assert_eq!(cause.to_string(), source.to_string());
        assert_eq!(cause.downcast_ref(), Some(&source));
    }
}
