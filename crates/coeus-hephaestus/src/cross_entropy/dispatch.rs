use super::{CrossEntropyBackend, CrossEntropyProvider};
use crate::layout::ranked;
use coeus_core::Layout;
use hephaestus_core::{
    CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyOps, StridedView,
};

fn exactly_ranked<const N: usize>(
    operation: &'static str,
    layout: &Layout,
) -> Result<leto::Layout<N>, coeus_core::BackendError> {
    let rank = layout.ndim();
    if rank != N {
        return Err(coeus_core::BackendError::LayoutRankMismatch {
            operation,
            lhs: rank,
            rhs: N,
        });
    }
    ranked::<N>(operation, layout)
}

#[expect(
    clippy::too_many_arguments,
    reason = "the function assembles the complete provider forward request"
)]
pub(super) fn forward<B: CrossEntropyBackend>(
    _backend: &B,
    logits: &B::DeviceBuffer<f32>,
    logits_layout: &Layout,
    targets: &B::DeviceBuffer<u32>,
    loss: &mut B::DeviceBuffer<f32>,
    loss_layout: &Layout,
    probabilities: &mut B::DeviceBuffer<f32>,
    probabilities_layout: &Layout,
) -> Result<(), B::Error> {
    let logits_layout = exactly_ranked::<2>("cross_entropy_forward_logits", logits_layout)?;
    let batch = logits_layout.shape[0];
    let target_count = coeus_core::Storage::len(targets);
    if target_count != batch {
        return Err(coeus_core::BackendError::ShapeMismatch {
            operation: "cross_entropy_forward_targets",
            lhs: vec![target_count],
            rhs: vec![batch],
        }
        .into());
    }
    let target_layout = Layout::new([batch].into());
    let target_layout = ranked::<1>("cross_entropy_forward_targets", &target_layout)?;
    let loss_layout = exactly_ranked::<1>("cross_entropy_forward_loss", loss_layout)?;
    let probabilities_layout =
        exactly_ranked::<2>("cross_entropy_forward_probabilities", probabilities_layout)?;
    let loss_candidate = B::cross_entropy_candidate(loss, false, "cross_entropy_forward_loss")?;
    let probabilities_candidate =
        B::cross_entropy_candidate(probabilities, false, "cross_entropy_forward_probabilities")?;
    let operations = <B::Provider as CrossEntropyProvider>::Operations::default();
    operations
        .cross_entropy_forward_into(
            <B::Provider as crate::HephaestusProvider>::device(),
            CrossEntropyForwardOperands {
                logits: StridedView::new(B::cross_entropy_buffer(logits), &logits_layout),
                targets: StridedView::new(B::cross_entropy_target_buffer(targets), &target_layout),
                loss: StridedView::new(B::cross_entropy_buffer(&loss_candidate), &loss_layout),
                probabilities: StridedView::new(
                    B::cross_entropy_buffer(&probabilities_candidate),
                    &probabilities_layout,
                ),
            },
        )
        .map_err(|source| B::cross_entropy_dispatch_error("cross_entropy_forward", source))?;
    B::install_cross_entropy_candidate(loss, loss_candidate);
    B::install_cross_entropy_candidate(probabilities, probabilities_candidate);
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "the function assembles the complete provider backward request"
)]
pub(super) fn backward<B: CrossEntropyBackend>(
    _backend: &B,
    output_gradient: &B::DeviceBuffer<f32>,
    output_gradient_layout: &Layout,
    probabilities: &B::DeviceBuffer<f32>,
    probabilities_layout: &Layout,
    targets: &B::DeviceBuffer<u32>,
    logit_gradient: &mut B::DeviceBuffer<f32>,
    logit_gradient_layout: &Layout,
) -> Result<(), B::Error> {
    let output_gradient_layout = exactly_ranked::<1>(
        "cross_entropy_backward_output_gradient",
        output_gradient_layout,
    )?;
    let probabilities_layout =
        exactly_ranked::<2>("cross_entropy_backward_probabilities", probabilities_layout)?;
    let batch = probabilities_layout.shape[0];
    let target_count = coeus_core::Storage::len(targets);
    if target_count != batch {
        return Err(coeus_core::BackendError::ShapeMismatch {
            operation: "cross_entropy_backward_targets",
            lhs: vec![target_count],
            rhs: vec![batch],
        }
        .into());
    }
    let target_layout = Layout::new([batch].into());
    let target_layout = ranked::<1>("cross_entropy_backward_targets", &target_layout)?;
    let logit_gradient_layout = exactly_ranked::<2>(
        "cross_entropy_backward_logit_gradient",
        logit_gradient_layout,
    )?;
    let logit_gradient_candidate = B::cross_entropy_candidate(
        logit_gradient,
        true,
        "cross_entropy_backward_logit_gradient",
    )?;
    let operations = <B::Provider as CrossEntropyProvider>::Operations::default();
    operations
        .cross_entropy_backward_accumulate(
            <B::Provider as crate::HephaestusProvider>::device(),
            CrossEntropyBackwardOperands {
                output_gradient: StridedView::new(
                    B::cross_entropy_buffer(output_gradient),
                    &output_gradient_layout,
                ),
                probabilities: StridedView::new(
                    B::cross_entropy_buffer(probabilities),
                    &probabilities_layout,
                ),
                targets: StridedView::new(B::cross_entropy_target_buffer(targets), &target_layout),
                logit_gradient: StridedView::new(
                    B::cross_entropy_buffer(&logit_gradient_candidate),
                    &logit_gradient_layout,
                ),
            },
        )
        .map_err(|source| B::cross_entropy_dispatch_error("cross_entropy_backward", source))?;
    B::install_cross_entropy_candidate(logit_gradient, logit_gradient_candidate);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::exactly_ranked;
    use coeus_core::{BackendError, Layout};

    #[test]
    fn cross_entropy_rank_gate_rejects_left_padding() {
        let error = exactly_ranked::<2>("cross_entropy_forward_logits", &Layout::new([12].into()))
            .expect_err("cross-entropy matrices require exactly rank two");

        assert!(matches!(
            error,
            BackendError::LayoutRankMismatch { lhs: 1, rhs: 2, .. }
        ));
    }
}
