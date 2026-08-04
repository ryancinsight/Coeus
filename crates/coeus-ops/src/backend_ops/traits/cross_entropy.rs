//! Provider-selected mean cross-entropy capability.

use coeus_core::{ComputeBackend, Layout, Scalar};

/// Mean cross-entropy forward and additive-backward dispatch.
///
/// Backends choose their provider through the implementation: CPU backends
/// call Leto over borrowed storage, while accelerator backends call
/// Hephaestus over device-resident buffers.
pub trait CrossEntropyOps<T: Scalar>: ComputeBackend {
    /// Provider-native target representation retained for backward.
    type Targets: Send + Sync + 'static;

    /// Validate and retain one target index per batch row.
    ///
    /// # Errors
    ///
    /// Returns a typed backend error when a target cannot be represented by
    /// the selected provider.
    fn prepare_cross_entropy_targets(
        &self,
        targets: &[usize],
    ) -> Result<Self::Targets, Self::Error>;

    /// Write mean loss and saved probabilities without changing providers.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, or dispatch failure. Provider
    /// validation completes before either destination changes.
    #[expect(
        clippy::too_many_arguments,
        reason = "the method carries the complete provider forward contract"
    )]
    fn cross_entropy_forward(
        &self,
        logits: &Self::DeviceBuffer<T>,
        logits_layout: &Layout,
        targets: &Self::Targets,
        loss: &mut Self::DeviceBuffer<T>,
        loss_layout: &Layout,
        probabilities: &mut Self::DeviceBuffer<T>,
        probabilities_layout: &Layout,
    ) -> Result<(), Self::Error>;

    /// Add the mean-reduced logit gradient into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns a typed validation, preparation, or dispatch failure. Provider
    /// validation completes before the destination changes.
    #[expect(
        clippy::too_many_arguments,
        reason = "the method carries the complete provider backward contract"
    )]
    fn cross_entropy_backward_accumulate(
        &self,
        output_gradient: &Self::DeviceBuffer<T>,
        output_gradient_layout: &Layout,
        probabilities: &Self::DeviceBuffer<T>,
        probabilities_layout: &Layout,
        targets: &Self::Targets,
        logit_gradient: &mut Self::DeviceBuffer<T>,
        logit_gradient_layout: &Layout,
    ) -> Result<(), Self::Error>;
}
