use super::dispatch::{self, PreparedStaggeredPair};
use super::provider::StaggeredProvider;
use crate::HephaestusBackend;
use coeus_core::Layout;
use coeus_ops::{FiniteDifferenceAxis as Axis, StaggeredPairOps};

/// The provider states the pair in `f32`, so the accelerator backend binds the
/// Coeus seam at that scalar rather than generically — the device contract
/// fixes the type, and a generic impl here would be falsely generic.
impl<P> StaggeredPairOps<f32> for HephaestusBackend<P>
where
    P: StaggeredProvider,
{
    type StaggeredPair = PreparedStaggeredPair<Self>;

    fn prepare_staggered_pair(
        &self,
        order: usize,
        spacing: [f32; 3],
    ) -> Result<Self::StaggeredPair, Self::Error> {
        PreparedStaggeredPair::new(order, spacing)
    }

    fn staggered_gradient(
        &self,
        pair: &Self::StaggeredPair,
        axis: Axis,
        input: &Self::DeviceBuffer<f32>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<f32>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::gradient::<Self>(pair, axis, (input, input_layout), (output, output_layout))
    }

    fn staggered_divergence(
        &self,
        pair: &Self::StaggeredPair,
        axis: Axis,
        input: &Self::DeviceBuffer<f32>,
        input_layout: &Layout,
        output: &mut Self::DeviceBuffer<f32>,
        output_layout: &Layout,
    ) -> Result<(), Self::Error> {
        dispatch::divergence::<Self>(pair, axis, (input, input_layout), (output, output_layout))
    }
}
