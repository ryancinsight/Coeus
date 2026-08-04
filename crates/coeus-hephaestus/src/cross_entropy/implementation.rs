use super::{CrossEntropyBackend, CrossEntropyProvider};
use crate::HephaestusBackend;
use coeus_core::{BackendError, ComputeBackend, Layout};

/// Convert host control indices once into provider-native retained storage.
///
/// # Errors
///
/// Returns a typed overflow error when an index is not representable by the
/// provider's `u32` target contract.
pub fn prepare_targets<B: ComputeBackend>(
    backend: &B,
    targets: &[usize],
) -> Result<B::DeviceBuffer<u32>, B::Error> {
    let encoded = targets
        .iter()
        .map(|&target| {
            u32::try_from(target).map_err(|_| BackendError::Overflow {
                operation: "cross_entropy_targets",
                reason: "target index exceeds the provider u32 contract",
            })
        })
        .collect::<Result<Box<[_]>, _>>()?;
    let mut storage = backend.allocate(encoded.len());
    backend.copy_to_device(&encoded, &mut storage);
    Ok(storage)
}

impl<P> coeus_ops::CrossEntropyOps<f32> for HephaestusBackend<P>
where
    P: CrossEntropyProvider,
{
    type Targets = Self::DeviceBuffer<u32>;

    fn prepare_cross_entropy_targets(
        &self,
        targets: &[usize],
    ) -> Result<Self::Targets, Self::Error> {
        prepare_targets(self, targets)
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
