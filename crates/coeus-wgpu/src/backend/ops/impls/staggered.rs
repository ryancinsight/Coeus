use crate::backend::{get_wgpu_context, WgpuBackend, WgpuBackendError};
use coeus_core::{BackendError, Layout};
use coeus_hephaestus::{PreparedStaggeredPair, StaggeredBackend, StaggeredProvider};
use coeus_ops::{Axis, StaggeredPairOps};
use hephaestus_core::{ComputeDevice, HephaestusError};
use hephaestus_wgpu::{WgpuDevice, WgpuStaggered3DOps};

impl StaggeredProvider for WgpuBackend {
    type Operations = WgpuStaggered3DOps;
}

impl StaggeredBackend for WgpuBackend {
    type Device = WgpuDevice;
    type Operations = WgpuStaggered3DOps;

    fn staggered_device() -> &'static Self::Device {
        &get_wgpu_context().hephaestus_device
    }

    fn staggered_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<f32> {
        storage.buffer.as_ref()
    }

    fn staggered_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        WgpuBackendError::Validation(BackendError::Storage { operation, reason })
    }

    fn staggered_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        WgpuBackendError::dispatch(operation, source)
    }
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
        coeus_hephaestus::staggered_gradient::<WgpuBackend>(
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
        coeus_hephaestus::staggered_divergence::<WgpuBackend>(
            pair,
            axis,
            (input, input_layout),
            (output, output_layout),
        )
    }
}
