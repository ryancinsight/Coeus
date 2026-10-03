//! Reduction operation tags.

/// Reduction operation tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// Sum of all elements.
    Sum,
    /// Product of all elements.
    Prod,
    /// Arithmetic mean of all elements.
    Mean,
    /// Maximum element.
    Max,
    /// Minimum element.
    Min,
}

/// A reduction every element type computes exactly in its own arithmetic:
/// [`ReductionOp`] without [`ReductionOp::Mean`], whose division needs a
/// floating-point element.
///
/// A dispatch that matches on `ClosedReduction` has no mean arm, so a mean
/// cannot reach a kernel instantiated for an integer element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosedReduction {
    /// Sum of all elements.
    Sum,
    /// Product of all elements.
    Prod,
    /// Maximum element.
    Max,
    /// Minimum element.
    Min,
}

impl From<ClosedReduction> for ReductionOp {
    fn from(reduction: ClosedReduction) -> Self {
        match reduction {
            ClosedReduction::Sum => Self::Sum,
            ClosedReduction::Prod => Self::Prod,
            ClosedReduction::Max => Self::Max,
            ClosedReduction::Min => Self::Min,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ClosedReduction, ReductionOp};

    #[test]
    fn closed_reductions_name_their_reduction_op() {
        let pairs = [
            (ClosedReduction::Sum, ReductionOp::Sum),
            (ClosedReduction::Prod, ReductionOp::Prod),
            (ClosedReduction::Max, ReductionOp::Max),
            (ClosedReduction::Min, ReductionOp::Min),
        ];
        for (closed, op) in pairs {
            assert_eq!(ReductionOp::from(closed), op);
        }
    }
}
