//! Reduction operation tags.

use crate::backend::BackendError;

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

impl ClosedReduction {
    /// The closed reduction `op` names.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::FloatOnlyReduction`] naming `operation` for
    /// [`ReductionOp::Mean`].
    pub fn from_op(operation: &'static str, op: ReductionOp) -> Result<Self, BackendError> {
        match op {
            ReductionOp::Sum => Ok(Self::Sum),
            ReductionOp::Prod => Ok(Self::Prod),
            ReductionOp::Max => Ok(Self::Max),
            ReductionOp::Min => Ok(Self::Min),
            ReductionOp::Mean => Err(BackendError::FloatOnlyReduction {
                operation,
                reduction: op,
            }),
        }
    }
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
    use super::{BackendError, ClosedReduction, ReductionOp};

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
            assert_eq!(ClosedReduction::from_op("reduce", op), Ok(closed));
        }
    }

    #[test]
    fn mean_is_rejected_with_the_operation_name() {
        assert_eq!(
            ClosedReduction::from_op("fused reduction", ReductionOp::Mean),
            Err(BackendError::FloatOnlyReduction {
                operation: "fused reduction",
                reduction: ReductionOp::Mean,
            })
        );
    }
}
