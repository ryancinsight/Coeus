//! The Hephaestus fused-reduction selector for each Coeus reduction tag.

use coeus_core::ReductionOp;
use hephaestus_core::FusedReduction;

/// The Hephaestus fused selector that evaluates `reduction`.
///
/// The providers' generic fused reductions reject [`ReductionOp::Mean`]
/// before selecting; only their float-bound fused-mean entry points select
/// [`FusedReduction::Mean`].
#[must_use]
pub fn fused_selector(reduction: ReductionOp) -> FusedReduction {
    match reduction {
        ReductionOp::Sum => FusedReduction::Sum,
        ReductionOp::Prod => FusedReduction::Product,
        ReductionOp::Mean => FusedReduction::Mean,
        ReductionOp::Max => FusedReduction::Maximum,
        ReductionOp::Min => FusedReduction::Minimum,
    }
}

#[cfg(test)]
mod tests {
    use super::fused_selector;
    use coeus_core::ReductionOp;
    use hephaestus_core::FusedReduction;

    #[test]
    fn each_reduction_selects_its_fused_kernel() {
        for (reduction, selector) in [
            (ReductionOp::Sum, FusedReduction::Sum),
            (ReductionOp::Prod, FusedReduction::Product),
            (ReductionOp::Mean, FusedReduction::Mean),
            (ReductionOp::Max, FusedReduction::Maximum),
            (ReductionOp::Min, FusedReduction::Minimum),
        ] {
            assert_eq!(fused_selector(reduction), selector);
        }
    }
}
