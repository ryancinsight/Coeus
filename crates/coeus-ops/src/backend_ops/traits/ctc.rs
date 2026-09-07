//! Connectionist temporal classification over backend-owned log probabilities.

use coeus_core::{ComputeBackend, Layout, Scalar};

/// Concatenated target labels and valid sequence lengths for a CTC batch.
///
/// Each length slice has one entry per sample. Targets contain exactly the
/// sum of `target_lengths` labels; every label is in range and differs from
/// `blank`. Input lengths cannot exceed the log-probability frame extent.
#[derive(Debug, Clone, Copy)]
pub struct CtcBatch<'a> {
    /// Target sequences concatenated in batch order.
    pub targets: &'a [usize],
    /// Number of active input frames for each sample.
    pub input_lengths: &'a [usize],
    /// Number of target labels for each sample.
    pub target_lengths: &'a [usize],
    /// Alphabet index removed when collapsing alignments.
    pub blank: usize,
}

/// Mean CTC loss and additive derivatives with respect to log probabilities.
///
/// Inputs have shape `[frames, batch, classes]`. Reduction is the batch mean
/// of each sample's negative log-likelihood divided by its target length,
/// with an empty target using divisor one. Backward differentiates independent
/// log probabilities; composition with log-softmax produces logit gradients.
///
/// CPU backends borrow storage for the Leto provider. This separate capability
/// adds no CTC requirement to backends that do not implement sequence alignment.
///
/// # Examples
///
/// One frame and an empty target admit only the blank alignment. The derivative
/// with respect to its log probability is minus one.
///
/// ```
/// use coeus_core::{CpuAddressableStorage, CpuStorage, Layout, SequentialBackend};
/// use coeus_ops::{CtcBatch, CtcOps};
///
/// let backend = SequentialBackend::new();
/// let layout = Layout::new([1, 1, 2].into());
/// let scalar = Layout::new([1].into());
/// let input = CpuStorage::from_slice(&[-std::f32::consts::LN_2; 2]);
/// let mut loss = CpuStorage::from_slice(&[0.0_f32]);
/// let state = backend.ctc_forward(&input, &layout,
///     CtcBatch { targets: &[], input_lengths: &[1], target_lengths: &[0], blank: 0 },
///     &mut loss, &scalar)?;
/// assert_eq!(loss.as_slice(), &[std::f32::consts::LN_2]);
/// let upstream = CpuStorage::from_slice(&[1.0_f32]);
/// let mut gradient = CpuStorage::from_slice(&[0.0_f32; 2]);
/// backend.ctc_backward_accumulate(&state, &upstream, &scalar, &mut gradient, &layout)?;
/// assert_eq!(gradient.as_slice(), &[-1.0, 0.0]);
/// # Ok::<(), coeus_core::BackendError>(())
/// ```
#[diagnostic::on_unimplemented(
    message = "the backend must implement CTC loss and additive log-probability gradients"
)]
pub trait CtcOps<T: Scalar>: ComputeBackend {
    /// Owned provider recurrences retained from forward for backward.
    type CtcState: Send + Sync + 'static;

    /// Write the mean loss to a `[1]` destination and return its saved state.
    ///
    /// Empty targets use the all-blank alignment. An impossible alignment
    /// writes positive infinity; its derivative is rejected by backward.
    /// Inactive frames are ignored. Active log probabilities must be
    /// nonpositive and finite or negative infinity.
    ///
    /// # Errors
    ///
    /// Rejects malformed layouts, lengths, labels, numeric values, or provider
    /// allocation failures before changing the loss destination.
    fn ctc_forward(
        &self,
        log_probs: &Self::DeviceBuffer<T>,
        log_probs_layout: &Layout,
        batch: CtcBatch<'_>,
        loss: &mut Self::DeviceBuffer<T>,
        loss_layout: &Layout,
    ) -> Result<Self::CtcState, Self::Error>;

    /// Add the `[1]` upstream gradient times the saved input derivative.
    ///
    /// The destination shape must equal the forward input shape. Inactive
    /// frames receive no contribution; pre-existing gradient values remain.
    ///
    /// # Errors
    ///
    /// Rejects invalid scalar/destination layouts, nonfinite arithmetic, and
    /// impossible alignments before changing any destination element.
    fn ctc_backward_accumulate(
        &self,
        state: &Self::CtcState,
        output_gradient: &Self::DeviceBuffer<T>,
        output_gradient_layout: &Layout,
        gradient: &mut Self::DeviceBuffer<T>,
        gradient_layout: &Layout,
    ) -> Result<(), Self::Error>;
}
