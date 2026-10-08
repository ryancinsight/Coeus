use super::dispatch;
use super::provider::FixedFdProvider;
use crate::HephaestusBackend;
use coeus_core::Layout;
use coeus_ops::{Axis, FiniteDifference3DOps, FiniteDifference3DScheme};
use hephaestus_core::FixedFd3DOps as ProviderOps;

/// The provider states the sweeps in `f32`, so the accelerator backend binds
/// the Coeus seam at that scalar rather than generically — the device
/// contract fixes the type, and a generic impl here would be falsely generic.
impl<P> FiniteDifference3DOps<f32> for HephaestusBackend<P>
where
    P: FixedFdProvider,
    <P::Operations as ProviderOps<P::Device>>::FixedFd3D: Send + Sync,
{
    fn finite_difference(
        &self,
        scheme: FiniteDifference3DScheme,
        axis: Axis,
        spacing: [f32; 3],
        input: &Self::DeviceBuffer<f32>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<f32>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::sweep::<Self>(
            scheme,
            axis,
            spacing,
            (input, input_layout),
            (output, output_layout),
        )
    }

    fn finite_difference_adjoint(
        &self,
        scheme: FiniteDifference3DScheme,
        axis: Axis,
        spacing: [f32; 3],
        upstream: &Self::DeviceBuffer<f32>,
        upstream_layout: &Layout,
        grad: &mut Self::DeviceBuffer<f32>,
        grad_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::adjoint::<Self>(
            scheme,
            axis,
            spacing,
            (upstream, upstream_layout),
            (grad, grad_layout),
        )
    }
}
