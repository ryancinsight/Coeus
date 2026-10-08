use std::sync::OnceLock;

use crate::backend::{get_cuda_device, CudaBackend};
use crate::CudaBackendError;
use coeus_core::{BackendError, Layout};
use coeus_hephaestus::{ctc_backward, ctc_forward, CtcBackend, CtcProvider};
use coeus_ops::{CtcBatch, CtcOps};
use hephaestus_core::{ComputeDevice, CtcOps as ProviderCtcOps, CtcStateBuffers, HephaestusError};
use hephaestus_cuda::{CtcKernel, CudaCtcOps, CudaDevice};

/// One compiled CTC kernel set for the process-wide CUDA device.
static KERNEL: OnceLock<CtcKernel> = OnceLock::new();

impl CtcProvider for CudaBackend {
    type Operations = CudaCtcOps;

    fn ctc_kernel() -> hephaestus_core::Result<
        &'static <Self::Operations as hephaestus_core::CtcOps<CudaDevice>>::Ctc,
    > {
        if let Some(kernel) = KERNEL.get() {
            return Ok(kernel);
        }
        let candidate = CudaCtcOps.prepare_ctc(get_cuda_device())?;
        let _ = KERNEL.set(candidate);
        KERNEL
            .get()
            .ok_or_else(|| hephaestus_core::HephaestusError::DeviceUnavailable {
                message: "ctc kernel initialization did not publish the compiled kernel".to_owned(),
            })
    }
}

impl CtcBackend for CudaBackend {
    type Device = CudaDevice;
    type Operations = CudaCtcOps;
    type Kernel = CtcKernel;

    fn ctc_device() -> &'static Self::Device {
        get_cuda_device()
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
        CudaBackendError::Validation {
            source: BackendError::Storage { operation, reason },
        }
    }

    fn ctc_dispatch_error(operation: &'static str, source: HephaestusError) -> Self::Error {
        CudaBackendError::dispatch(operation, source)
    }

    fn ctc_impossible_error(operation: &'static str, sample: usize) -> Self::Error {
        CudaBackendError::Validation {
            source: BackendError::UndefinedGradient { operation, sample },
        }
    }

    fn ctc_arithmetic_error(operation: &'static str, sample: usize) -> Self::Error {
        CudaBackendError::Validation {
            source: BackendError::NonFiniteSample { operation, sample },
        }
    }
}

/// The provider states CTC in `f32` — the bridge is concrete at that scalar
/// rather than generic, mirroring the fixed-scheme binding.
impl CtcOps<f32> for CudaBackend {
    type CtcState = CtcStateBuffers<CudaDevice>;

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
