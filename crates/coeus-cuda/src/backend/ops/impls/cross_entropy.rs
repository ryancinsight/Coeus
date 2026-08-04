use crate::backend::CudaBackend;
use crate::CudaBackendError;
use coeus_core::Layout;
use coeus_hephaestus::{prepare_cross_entropy_targets, CrossEntropyBackend, CrossEntropyProvider};
use hephaestus_core::{ComputeDevice, DeviceBuffer, HephaestusError};
use hephaestus_cuda::{CudaCrossEntropyOps, CudaDevice};
use themis::PlacementHint;

impl CrossEntropyProvider for CudaBackend {
    type Operations = CudaCrossEntropyOps;
}

impl CrossEntropyBackend for CudaBackend {
    type Provider = Self;

    fn cross_entropy_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<CudaDevice as ComputeDevice>::Buffer<f32> {
        storage.buffer.as_ref()
    }

    fn cross_entropy_candidate(
        storage: &Self::DeviceBuffer<f32>,
        preserve_contents: bool,
        operation: &'static str,
    ) -> Result<Self::DeviceBuffer<f32>, Self::Error> {
        let device = crate::backend::get_cuda_device();
        let candidate = device
            .alloc_uninitialized_with_hint(
                storage.buffer.len(),
                PlacementHint::Tier(storage.buffer.tier()),
            )
            .map_err(|source| CudaBackendError::dispatch(operation, source))?;
        if preserve_contents {
            device
                .copy_buffer(storage.buffer.as_ref(), &candidate)
                .map_err(|source| CudaBackendError::dispatch(operation, source))?;
        }
        Ok(crate::storage::CudaStorage::from_buffer(candidate))
    }

    fn install_cross_entropy_candidate(
        storage: &mut Self::DeviceBuffer<f32>,
        candidate: Self::DeviceBuffer<f32>,
    ) {
        *storage = candidate;
    }

    fn cross_entropy_target_buffer(
        storage: &Self::DeviceBuffer<u32>,
    ) -> &<CudaDevice as ComputeDevice>::Buffer<u32> {
        storage.buffer.as_ref()
    }

    fn cross_entropy_dispatch_error(
        operation: &'static str,
        source: HephaestusError,
    ) -> Self::Error {
        CudaBackendError::dispatch(operation, source)
    }
}

impl coeus_ops::CrossEntropyOps<f32> for CudaBackend {
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
