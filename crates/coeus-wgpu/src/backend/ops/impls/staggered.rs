use crate::backend::WgpuBackend;
use coeus_accelerator::{PreparedStaggeredPair, StaggeredBackend, StaggeredProvider};
use coeus_core::Layout;
use coeus_ops::{Axis, StaggeredPairOps};
use hephaestus_wgpu::WgpuStaggered3DOps;

impl StaggeredProvider for WgpuBackend {
    type Operations = WgpuStaggered3DOps;
}

impl StaggeredBackend for WgpuBackend {
    type Operations = WgpuStaggered3DOps;
}

/// The provider states the pair in `f32` — WGSL does not guarantee `f64`
/// storage — so the binding is concrete at that scalar rather than generic.
impl StaggeredPairOps<f32> for WgpuBackend {
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
        coeus_accelerator::staggered_gradient::<WgpuBackend>(
            pair,
            axis,
            (input, input_layout),
            (output, output_layout),
        )
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
        coeus_accelerator::staggered_divergence::<WgpuBackend>(
            pair,
            axis,
            (input, input_layout),
            (output, output_layout),
        )
    }
}
