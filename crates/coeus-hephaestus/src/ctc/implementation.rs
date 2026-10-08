use super::dispatch;
use super::provider::CtcProvider;
use crate::HephaestusBackend;
use coeus_core::Layout;
use coeus_ops::{CtcBatch, CtcOps};
use hephaestus_core::{ComputeDevice, CtcOps as ProviderOps, CtcStateBuffers};

/// The provider states CTC in `f32`, so the accelerator backend binds the
/// Coeus seam at that scalar rather than generically — the device contract
/// fixes the type, and a generic impl here would be falsely generic. The
/// retained state is the provider's device state itself: alpha and beta
/// stay on the device between forward and backward, never round-tripping
/// through the host.
impl<P> CtcOps<f32> for HephaestusBackend<P>
where
    P: CtcProvider,
    <P::Operations as ProviderOps<P::Device>>::Ctc: Send + Sync,
    <P::Device as ComputeDevice>::Buffer<f32>: Send + Sync,
    <P::Device as ComputeDevice>::Buffer<u32>: Send + Sync,
{
    type CtcState = CtcStateBuffers<P::Device>;

    fn ctc_forward(
        &self,
        log_probs: &Self::DeviceBuffer<f32>,
        log_probs_layout: &Layout,
        batch: CtcBatch,
        loss: &mut Self::DeviceBuffer<f32>,
        loss_layout: &Layout,
    ) -> Result<Self::CtcState, Self::Error> {
        dispatch::ctc_forward::<Self>((log_probs, log_probs_layout), batch, (loss, loss_layout))
    }

    fn ctc_backward_accumulate(
        &self,
        state: &Self::CtcState,
        upstream: &Self::DeviceBuffer<f32>,
        upstream_layout: &Layout,
        grad: &mut Self::DeviceBuffer<f32>,
        grad_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::ctc_backward::<Self>(state, (upstream, upstream_layout), (grad, grad_layout))
    }
}
