use crate::{reduction::HephaestusBackend, HephaestusProvider, HephaestusStorage};
use coeus_core::{Layout, Scalar, Storage};
use hephaestus_core::{ComputeDevice, CrossEntropyOps, DeviceBuffer, HephaestusError};
use themis::PlacementHint;

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
    fn cross_entropy_buffer<T: Scalar>(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<<Self::Provider as HephaestusProvider>::Device as hephaestus_core::ComputeDevice>::Buffer<T>;

    #[doc(hidden)]
    fn cross_entropy_candidate<T: Scalar>(
        storage: &Self::DeviceBuffer<T>,
        preserve_contents: bool,
        operation: &'static str,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error>;

    #[doc(hidden)]
    fn install_cross_entropy_candidate<T: Scalar>(
        storage: &mut Self::DeviceBuffer<T>,
        candidate: Self::DeviceBuffer<T>,
    );

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

/// Double-precision cross-entropy dispatch entry points.
///
/// Backends whose provider covers f64 (CUDA, WGPU, ROCm) opt in with an empty
/// implementation; f32-only providers (Metal has no FP64 hardware) simply do
/// not implement it. Each twin carries its own f64 operations bound so the
/// default bodies check without implied bounds through the provider marker.
pub trait CrossEntropyBackendF64: CrossEntropyBackend {
    #[doc(hidden)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the method mirrors the provider forward boundary"
    )]
    fn dispatch_cross_entropy_forward_f64(
        &self,
        logits: &Self::DeviceBuffer<f64>,
        logits_layout: &Layout,
        targets: &Self::DeviceBuffer<u32>,
        loss: &mut Self::DeviceBuffer<f64>,
        loss_layout: &Layout,
        probabilities: &mut Self::DeviceBuffer<f64>,
        probabilities_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
        <Self::Provider as CrossEntropyProvider>::Operations:
            CrossEntropyOps<<Self::Provider as HephaestusProvider>::Device, f64>,
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
    fn dispatch_cross_entropy_backward_f64(
        &self,
        output_gradient: &Self::DeviceBuffer<f64>,
        output_gradient_layout: &Layout,
        probabilities: &Self::DeviceBuffer<f64>,
        probabilities_layout: &Layout,
        targets: &Self::DeviceBuffer<u32>,
        logit_gradient: &mut Self::DeviceBuffer<f64>,
        logit_gradient_layout: &Layout,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
        <Self::Provider as CrossEntropyProvider>::Operations:
            CrossEntropyOps<<Self::Provider as HephaestusProvider>::Device, f64>,
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

    fn cross_entropy_buffer<T: Scalar>(
        storage: &Self::DeviceBuffer<T>,
    ) -> &<P::Device as hephaestus_core::ComputeDevice>::Buffer<T> {
        storage.buffer()
    }

    fn cross_entropy_candidate<T: Scalar>(
        storage: &Self::DeviceBuffer<T>,
        preserve_contents: bool,
        operation: &'static str,
    ) -> Result<Self::DeviceBuffer<T>, Self::Error> {
        prepare_candidate::<P, T>(storage, preserve_contents, operation)
    }

    fn install_cross_entropy_candidate<T: Scalar>(
        storage: &mut Self::DeviceBuffer<T>,
        candidate: Self::DeviceBuffer<T>,
    ) {
        *storage = candidate;
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

impl<P> CrossEntropyBackendF64 for HephaestusBackend<P> where P: CrossEntropyProvider {}

/// Allocate a fallible provider-native candidate for failure-atomic writes.
///
/// # Errors
///
/// Returns the provider allocation or device-copy failure without changing the
/// source storage.
pub fn prepare_candidate<P, T>(
    storage: &HephaestusStorage<P, T>,
    preserve_contents: bool,
    operation: &'static str,
) -> Result<HephaestusStorage<P, T>, crate::HephaestusBackendError>
where
    P: CrossEntropyProvider,
    T: Scalar,
{
    let device = P::device();
    let candidate = device
        .alloc_uninitialized_with_hint(storage.len(), PlacementHint::Tier(storage.buffer().tier()))
        .map_err(|source| crate::HephaestusBackendError::device(operation, source))?;
    if preserve_contents {
        device
            .copy_buffer(storage.buffer(), &candidate)
            .map_err(|source| crate::HephaestusBackendError::device(operation, source))?;
    }
    Ok(HephaestusStorage::from_buffer(candidate))
}
