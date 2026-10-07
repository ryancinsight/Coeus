use crate::backend::{get_cuda_device, CudaBackend};
use crate::CudaBackendError;
use coeus_core::{BackendError, Layout};
use coeus_hephaestus::{PreparedStaggeredPair, StaggeredBackend, StaggeredProvider};
use coeus_ops::{Axis, StaggeredPairOps};
use hephaestus_core::{ComputeDevice, HephaestusError};
use hephaestus_cuda::{CudaDevice, CudaStaggered3DOps};

impl StaggeredProvider for CudaBackend {
    type Operations = CudaStaggered3DOps;
}

impl StaggeredBackend for CudaBackend {
    type Device = CudaDevice;
    type Operations = CudaStaggered3DOps;

    fn staggered_device() -> &'static Self::Device {
        get_cuda_device()
    }

    fn staggered_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<f32> {
        storage.buffer()
    }

    fn staggered_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        CudaBackendError::Validation {
            source: BackendError::Storage { operation, reason },
        }
    }

    fn staggered_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        CudaBackendError::dispatch(operation, source)
    }
}

/// The provider states the pair in `f32` — the bridge is concrete at that
/// scalar rather than generic, mirroring the WGSL binding.
impl StaggeredPairOps<f32> for CudaBackend {
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
        coeus_hephaestus::staggered_gradient::<CudaBackend>(
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
        coeus_hephaestus::staggered_divergence::<CudaBackend>(
            pair,
            axis,
            (input, input_layout),
            (output, output_layout),
        )
    }
}
