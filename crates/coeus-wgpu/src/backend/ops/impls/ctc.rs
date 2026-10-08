use std::sync::OnceLock;

use crate::backend::{get_wgpu_context, WgpuBackend, WgpuBackendError};
use coeus_core::{BackendError, Layout};
use coeus_hephaestus::{ctc_backward, ctc_forward, get_or_try_init, CtcBackend, CtcProvider};
use coeus_ops::{CtcBatch, CtcOps};
use hephaestus_core::{ComputeDevice, CtcOps as ProviderCtcOps, CtcStateBuffers, HephaestusError};
use hephaestus_wgpu::{CtcKernel, WgpuCtcOps, WgpuDevice};

/// One compiled CTC kernel set for the process-wide WGPU device.
static KERNEL: OnceLock<CtcKernel> = OnceLock::new();

impl CtcProvider for WgpuBackend {
    type Operations = WgpuCtcOps;

    fn ctc_kernel() -> hephaestus_core::Result<
        &'static <Self::Operations as hephaestus_core::CtcOps<WgpuDevice>>::Ctc,
    > {
        get_or_try_init(
            &KERNEL,
            "ctc kernel initialization did not publish the compiled kernel",
            || WgpuCtcOps.prepare_ctc(&get_wgpu_context().hephaestus_device),
        )
    }
}

impl CtcBackend for WgpuBackend {
    type Device = WgpuDevice;
    type Operations = WgpuCtcOps;
    type Kernel = CtcKernel;

    fn ctc_device() -> &'static Self::Device {
        &get_wgpu_context().hephaestus_device
    }

    fn ctc_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<Self::Device as ComputeDevice>::Buffer<f32> {
        storage.buffer()
    }

    fn ctc_kernel() -> Result<&'static Self::Kernel, Self::Error> {
        <Self as CtcProvider>::ctc_kernel()
            .map_err(|source| Self::ctc_dispatch_error("ctc_kernel", source))
    }

    fn ctc_configuration_error(operation: &'static str, reason: String) -> Self::Error {
        WgpuBackendError::Validation(BackendError::Storage { operation, reason })
    }

    fn ctc_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        WgpuBackendError::dispatch(operation, source)
    }

    fn ctc_impossible_error(operation: &'static str, sample: usize) -> Self::Error {
        WgpuBackendError::Validation(BackendError::UndefinedGradient { operation, sample })
    }

    fn ctc_arithmetic_error(operation: &'static str, sample: usize) -> Self::Error {
        WgpuBackendError::Validation(BackendError::NonFiniteSample { operation, sample })
    }
}

/// The provider states CTC in `f32` — WGSL does not guarantee `f64`
/// storage — so the binding is concrete at that scalar rather than generic.
impl CtcOps<f32> for WgpuBackend {
    type CtcState = CtcStateBuffers<WgpuDevice>;

    fn ctc_forward(
        &self,
        log_probs: &Self::DeviceBuffer<f32>,
        log_probs_layout: &Layout,
        batch: CtcBatch,
        loss: &mut Self::DeviceBuffer<f32>,
        loss_layout: &Layout,
    ) -> Result<Self::CtcState, Self::Error> {
        ctc_forward::<Self>((log_probs, log_probs_layout), batch, (loss, loss_layout))
    }

    fn ctc_backward_accumulate(
        &self,
        state: &Self::CtcState,
        upstream: &Self::DeviceBuffer<f32>,
        upstream_layout: &Layout,
        grad: &mut Self::DeviceBuffer<f32>,
        grad_layout: &Layout,
    ) -> Result<(), Self::Error> {
        ctc_backward::<Self>(state, (upstream, upstream_layout), (grad, grad_layout))
    }
}
