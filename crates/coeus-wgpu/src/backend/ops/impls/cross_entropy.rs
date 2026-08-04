use crate::backend::{WgpuBackend, WgpuBackendError};
use coeus_core::Layout;
use coeus_hephaestus::{prepare_cross_entropy_targets, CrossEntropyBackend, CrossEntropyProvider};
use hephaestus_core::{ComputeDevice, HephaestusError};
use hephaestus_wgpu::{WgpuCrossEntropyOps, WgpuDevice};

impl CrossEntropyProvider for WgpuBackend {
    type Operations = WgpuCrossEntropyOps;
}

impl CrossEntropyBackend for WgpuBackend {
    type Provider = Self;

    fn cross_entropy_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<WgpuDevice as ComputeDevice>::Buffer<f32> {
        storage.buffer.as_ref()
    }

    fn cross_entropy_target_buffer(
        storage: &Self::DeviceBuffer<u32>,
    ) -> &<WgpuDevice as ComputeDevice>::Buffer<u32> {
        storage.buffer.as_ref()
    }

    fn cross_entropy_dispatch_error(
        operation: &'static str,
        source: HephaestusError,
    ) -> Self::Error {
        WgpuBackendError::dispatch(operation, source)
    }
}

impl coeus_ops::CrossEntropyOps<f32> for WgpuBackend {
    type Targets = Self::DeviceBuffer<u32>;

    fn prepare_cross_entropy_targets(
        &self,
        targets: &[usize],
    ) -> Result<Self::Targets, Self::Error> {
        prepare_cross_entropy_targets(self, targets)
    }

    fn cross_entropy_forward(
        &self,
        logits: &Self::DeviceBuffer<f32>,
        logits_layout: &Layout,
        targets: &Self::Targets,
        loss: &mut Self::DeviceBuffer<f32>,
        loss_layout: &Layout,
        probabilities: &mut Self::DeviceBuffer<f32>,
        probabilities_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.dispatch_cross_entropy_forward(
            logits,
            logits_layout,
            targets,
            loss,
            loss_layout,
            probabilities,
            probabilities_layout,
        )
    }

    fn cross_entropy_backward_accumulate(
        &self,
        output_gradient: &Self::DeviceBuffer<f32>,
        output_gradient_layout: &Layout,
        probabilities: &Self::DeviceBuffer<f32>,
        probabilities_layout: &Layout,
        targets: &Self::Targets,
        logit_gradient: &mut Self::DeviceBuffer<f32>,
        logit_gradient_layout: &Layout,
    ) -> Result<(), Self::Error> {
        self.dispatch_cross_entropy_backward(
            output_gradient,
            output_gradient_layout,
            probabilities,
            probabilities_layout,
            targets,
            logit_gradient,
            logit_gradient_layout,
        )
    }
}
