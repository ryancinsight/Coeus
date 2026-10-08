use std::sync::OnceLock;

use crate::backend::{get_wgpu_context, WgpuBackend, WgpuBackendError};
use coeus_core::{BackendError, Layout};
use coeus_hephaestus::{FixedFdBackend, FixedFdProvider};
use coeus_ops::{Axis, FiniteDifference3DOps, FiniteDifference3DScheme};
use hephaestus_core::{ComputeDevice, FixedFd3DOps, HephaestusError};
use hephaestus_wgpu::{FixedFd3DKernel, WgpuDevice, WgpuFixedFd3DOps};

/// One compiled sweep kernel for the process-wide WGPU device.
static KERNEL: OnceLock<FixedFd3DKernel> = OnceLock::new();

impl FixedFdProvider for WgpuBackend {
    type Operations = WgpuFixedFd3DOps;

    fn fixed_fd_kernel(
    ) -> hephaestus_core::Result<&'static <Self::Operations as FixedFd3DOps<WgpuDevice>>::FixedFd3D>
    {
        if let Some(kernel) = KERNEL.get() {
            return Ok(kernel);
        }
        let candidate =
            WgpuFixedFd3DOps.prepare_fixed_fd_3d(&get_wgpu_context().hephaestus_device)?;
        let _ = KERNEL.set(candidate);
        KERNEL
            .get()
            .ok_or_else(|| hephaestus_core::HephaestusError::DeviceUnavailable {
                message: "fixed-fd kernel initialization did not publish the compiled kernel"
                    .to_owned(),
            })
    }
}

impl FixedFdBackend for WgpuBackend {
    type Device = WgpuDevice;
    type Operations = WgpuFixedFd3DOps;
    type Kernel = FixedFd3DKernel;

    fn fixed_fd_device() -> &'static Self::Device {
        &get_wgpu_context().hephaestus_device
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
        WgpuBackendError::Validation(BackendError::Storage { operation, reason })
    }

    fn fixed_fd_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        WgpuBackendError::dispatch(operation, source)
    }
}

/// The provider states the sweeps in `f32` — WGSL does not guarantee `f64`
/// storage — so the binding is concrete at that scalar rather than generic.
impl FiniteDifference3DOps<f32> for WgpuBackend {
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
        coeus_hephaestus::fixed_fd_sweep::<WgpuBackend>(
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
        coeus_hephaestus::fixed_fd_adjoint::<WgpuBackend>(
            scheme,
            axis,
            spacing,
            (upstream, upstream_layout),
            (grad, grad_layout),
        )
    }
}
