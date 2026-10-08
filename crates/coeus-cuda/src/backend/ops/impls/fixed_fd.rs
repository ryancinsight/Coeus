use std::sync::OnceLock;

use crate::backend::{get_cuda_device, CudaBackend};
use crate::CudaBackendError;
use coeus_core::{BackendError, Layout};
use coeus_hephaestus::{FixedFdBackend, FixedFdProvider};
use coeus_ops::{Axis, FiniteDifference3DOps, FiniteDifference3DScheme};
use hephaestus_core::{ComputeDevice, FixedFd3DOps, HephaestusError};
use hephaestus_cuda::{CudaDevice, CudaFixedFd3DOps, FixedFd3DKernel};

/// One compiled sweep kernel for the process-wide CUDA device.
static KERNEL: OnceLock<FixedFd3DKernel> = OnceLock::new();

impl FixedFdProvider for CudaBackend {
    type Operations = CudaFixedFd3DOps;

    fn fixed_fd_kernel(
    ) -> hephaestus_core::Result<&'static <Self::Operations as FixedFd3DOps<CudaDevice>>::FixedFd3D>
    {
        if let Some(kernel) = KERNEL.get() {
            return Ok(kernel);
        }
        let candidate = CudaFixedFd3DOps.prepare_fixed_fd_3d(get_cuda_device())?;
        let _ = KERNEL.set(candidate);
        KERNEL
            .get()
            .ok_or_else(|| hephaestus_core::HephaestusError::DeviceUnavailable {
                message: "fixed-fd kernel initialization did not publish the compiled kernel"
                    .to_owned(),
            })
    }
}

impl FixedFdBackend for CudaBackend {
    type Device = CudaDevice;
    type Operations = CudaFixedFd3DOps;
    type Kernel = FixedFd3DKernel;

    fn fixed_fd_device() -> &'static Self::Device {
        get_cuda_device()
    }

    fn fixed_fd_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<f32> {
        storage.buffer()
    }

    fn fixed_fd_kernel() -> Result<&'static Self::Kernel, Self::Error> {
        <Self as FixedFdProvider>::fixed_fd_kernel()
            .map_err(|source| Self::fixed_fd_dispatch_error("fixed_fd_kernel", source))
    }

    fn fixed_fd_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        CudaBackendError::Validation {
            source: BackendError::Storage { operation, reason },
        }
    }

    fn fixed_fd_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        CudaBackendError::dispatch(operation, source)
    }
}

/// The provider states the sweeps in `f32` — the bridge is concrete at that
/// scalar rather than generic, mirroring the staggered binding.
impl FiniteDifference3DOps<f32> for CudaBackend {
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
        coeus_hephaestus::fixed_fd_sweep::<CudaBackend>(
            scheme,
            axis,
            spacing,
            (input, input_layout),
            (output, output_layout),
        )
    }
}
