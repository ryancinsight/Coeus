use super::{CrossEntropyBackend, CrossEntropyProvider};
use crate::layout::ranked;
use coeus_core::Layout;
use hephaestus_core::{
    CrossEntropyBackwardOperands, CrossEntropyForwardOperands, CrossEntropyOps, StridedView,
};

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
    let target_layout = Layout::new([logits_layout.shape()[0]].into());
    let logits_layout = ranked::<2>("cross_entropy_forward_logits", logits_layout)?;
    let target_layout = ranked::<1>("cross_entropy_forward_targets", &target_layout)?;
    let loss_layout = ranked::<1>("cross_entropy_forward_loss", loss_layout)?;
    let probabilities_layout =
        ranked::<2>("cross_entropy_forward_probabilities", probabilities_layout)?;
    let operations = <B::Provider as CrossEntropyProvider>::Operations::default();
    operations
        .cross_entropy_forward_into(
            <B::Provider as crate::HephaestusProvider>::device(),
            CrossEntropyForwardOperands {
                logits: StridedView::new(B::cross_entropy_buffer(logits), &logits_layout),
                targets: StridedView::new(B::cross_entropy_target_buffer(targets), &target_layout),
                loss: StridedView::new(B::cross_entropy_buffer(loss), &loss_layout),
                probabilities: StridedView::new(
                    B::cross_entropy_buffer(probabilities),
                    &probabilities_layout,
                ),
            },
        )
        .map_err(|source| B::cross_entropy_dispatch_error("cross_entropy_forward", source))
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
    let target_layout = Layout::new([probabilities_layout.shape()[0]].into());
    let output_gradient_layout = ranked::<1>(
        "cross_entropy_backward_output_gradient",
        output_gradient_layout,
    )?;
    let probabilities_layout =
        ranked::<2>("cross_entropy_backward_probabilities", probabilities_layout)?;
    let target_layout = ranked::<1>("cross_entropy_backward_targets", &target_layout)?;
    let logit_gradient_layout = ranked::<2>(
        "cross_entropy_backward_logit_gradient",
        logit_gradient_layout,
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
                    B::cross_entropy_buffer(logit_gradient),
                    &logit_gradient_layout,
                ),
            },
        )
        .map_err(|source| B::cross_entropy_dispatch_error("cross_entropy_backward", source))
}
