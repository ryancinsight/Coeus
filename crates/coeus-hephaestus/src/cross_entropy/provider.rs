use crate::{reduction::HephaestusBackend, HephaestusProvider};
use coeus_core::Layout;
use hephaestus_core::{CrossEntropyOps, HephaestusError};

/// Hephaestus provider owning mean cross-entropy kernels.
pub trait CrossEntropyProvider: HephaestusProvider {
    /// Monomorphized operation marker selected by this provider.
    type Operations: CrossEntropyOps<Self::Device, f32> + Default;
}

/// Projects Coeus buffers into one provider-owned cross-entropy path.
pub trait CrossEntropyBackend: coeus_core::ComputeBackend {
    /// Hephaestus provider selected by this Coeus backend.
    type Provider: CrossEntropyProvider;

    #[doc(hidden)]
    fn cross_entropy_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<<Self::Provider as HephaestusProvider>::Device as hephaestus_core::ComputeDevice>::Buffer<f32>;

    #[doc(hidden)]
    fn cross_entropy_target_buffer(
        storage: &Self::DeviceBuffer<u32>,
    ) -> &<<Self::Provider as HephaestusProvider>::Device as hephaestus_core::ComputeDevice>::Buffer<u32>;

    #[doc(hidden)]
    fn cross_entropy_dispatch_error(
        operation: &'static str,
        source: HephaestusError,
    ) -> Self::Error;

    #[doc(hidden)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the method mirrors the provider forward boundary"
    )]
    fn dispatch_cross_entropy_forward(
        &self,
        logits: &Self::DeviceBuffer<f32>,
        logits_layout: &Layout,
        targets: &Self::DeviceBuffer<u32>,
        loss: &mut Self::DeviceBuffer<f32>,
        loss_layout: &Layout,
        probabilities: &mut Self::DeviceBuffer<f32>,
        probabilities_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        super::dispatch::forward(
            self,
            logits,
            logits_layout,
            targets,
            loss,
            loss_layout,
            probabilities,
            probabilities_layout,
        )
    }

    #[doc(hidden)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the method mirrors the provider backward boundary"
    )]
    fn dispatch_cross_entropy_backward(
        &self,
        output_gradient: &Self::DeviceBuffer<f32>,
        output_gradient_layout: &Layout,
        probabilities: &Self::DeviceBuffer<f32>,
        probabilities_layout: &Layout,
        targets: &Self::DeviceBuffer<u32>,
        logit_gradient: &mut Self::DeviceBuffer<f32>,
        logit_gradient_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
    {
        super::dispatch::backward(
            self,
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

impl<P> CrossEntropyBackend for HephaestusBackend<P>
where
    P: CrossEntropyProvider,
{
    type Provider = P;

    fn cross_entropy_buffer(
        storage: &Self::DeviceBuffer<f32>,
    ) -> &<P::Device as hephaestus_core::ComputeDevice>::Buffer<f32> {
        storage.buffer()
    }

    fn cross_entropy_target_buffer(
        storage: &Self::DeviceBuffer<u32>,
    ) -> &<P::Device as hephaestus_core::ComputeDevice>::Buffer<u32> {
        storage.buffer()
    }

    fn cross_entropy_dispatch_error(
        operation: &'static str,
        source: HephaestusError,
    ) -> Self::Error {
        crate::HephaestusBackendError::device(operation, source)
    }
}
