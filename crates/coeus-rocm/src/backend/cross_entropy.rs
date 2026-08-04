use super::{RocmBackend, RocmProvider};
use coeus_core::Layout;
use coeus_hephaestus::{
    prepare_cross_entropy_targets, CrossEntropyBackend, HephaestusBackendError, HephaestusStorage,
};
use coeus_ops::CrossEntropyOps;
use hephaestus_core::{ComputeDevice, HephaestusError};

type Storage<T> = HephaestusStorage<RocmProvider, T>;

impl CrossEntropyBackend for RocmBackend {
    type Provider = RocmProvider;

    fn cross_entropy_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<hephaestus_rocm::RocmDevice as ComputeDevice>::Buffer<f32> {
        storage.buffer()
    }

    fn cross_entropy_target_buffer(
        storage: &Self::DeviceBuffer<u32>,
    ) -> &<hephaestus_rocm::RocmDevice as ComputeDevice>::Buffer<u32> {
        storage.buffer()
    }

    fn cross_entropy_dispatch_error(
        operation: &'static str,
        source: HephaestusError,
    ) -> Self::Error {
        HephaestusBackendError::Device { operation, source }
    }
}

impl CrossEntropyOps<f32> for RocmBackend {
    type Targets = Storage<u32>;

    fn prepare_cross_entropy_targets(
        &self,
        targets: &[usize],
    ) -> Result<Self::Targets, Self::Error> {
        prepare_cross_entropy_targets(self, targets)
    }

    fn cross_entropy_forward(
        &self,
        logits: &Storage<f32>,
        logits_layout: &Layout,
        targets: &Self::Targets,
        loss: &mut Storage<f32>,
        loss_layout: &Layout,
        probabilities: &mut Storage<f32>,
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
        output_gradient: &Storage<f32>,
        output_gradient_layout: &Layout,
        probabilities: &Storage<f32>,
        probabilities_layout: &Layout,
        targets: &Self::Targets,
        logit_gradient: &mut Storage<f32>,
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
