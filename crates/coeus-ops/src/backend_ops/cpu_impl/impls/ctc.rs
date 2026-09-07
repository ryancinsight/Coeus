use super::super::{error::map_leto_error, CpuBackend};
use crate::{CtcBatch, CtcOps};
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};
use coeus_leto::{to_leto_view, to_leto_view_mut};
use leto_ops::ctc::{CtcError, CtcState};

fn scalar_shape(operation: &'static str, layout: &Layout) -> Result<(), BackendError> {
    if layout.shape() != [1] {
        return Err(BackendError::ShapeMismatch {
            operation,
            lhs: layout.shape().to_vec(),
            rhs: vec![1],
        });
    }
    Ok(())
}

fn sequence_rank(operation: &'static str, layout: &Layout) -> Result<(), BackendError> {
    if layout.shape().len() != 3 {
        return Err(BackendError::LayoutRankMismatch {
            operation,
            lhs: layout.shape().len(),
            rhs: 3,
        });
    }
    Ok(())
}

fn ctc_error(operation: &'static str, error: CtcError) -> BackendError {
    match error {
        CtcError::EmptyDimension { shape } => BackendError::EmptyDimension {
            operation,
            dimension: if shape[1] == 0 { "batch" } else { "class" },
        },
        CtcError::LengthCount {
            batch,
            inputs,
            targets,
        } => BackendError::SequenceLengthCounts {
            batch,
            inputs,
            targets,
        },
        CtcError::InputLength {
            sample,
            actual,
            maximum,
        } => BackendError::SequenceInputLength {
            sample,
            actual,
            maximum,
        },
        CtcError::TargetCount { expected, actual } => BackendError::ShapeMismatch {
            operation: "ctc_targets",
            lhs: vec![actual],
            rhs: vec![expected],
        },
        CtcError::Label {
            label,
            blank,
            classes,
        } => BackendError::SequenceLabel {
            label,
            blank,
            classes,
        },
        CtcError::SizeOverflow { frames, targets } => {
            BackendError::SequenceStateOverflow { frames, targets }
        }
        CtcError::Allocation(source) => BackendError::Allocation { operation, source },
        CtcError::Layout(source) => map_leto_error(operation, source),
        CtcError::GradientShape { expected, actual } => BackendError::ShapeMismatch {
            operation,
            lhs: actual.to_vec(),
            rhs: expected.to_vec(),
        },
        CtcError::AliasedGradient => BackendError::AliasedLayout { operation },
        CtcError::LogProbability { index } => {
            BackendError::InvalidLogProbability { operation, index }
        }
        CtcError::ScalarExtent { extent } => {
            BackendError::UnrepresentableExtent { operation, extent }
        }
        CtcError::Arithmetic { sample } => BackendError::NonFiniteSample { operation, sample },
        CtcError::ImpossibleAlignment { sample } => {
            BackendError::UndefinedGradient { operation, sample }
        }
        // The provider error is non-exhaustive; an unknown future category
        // stays distinct from every mapped validation or arithmetic failure.
        source => BackendError::ProviderFailure {
            operation,
            reason: source.to_string(),
        },
    }
}

// CpuBackend fixes the storage/error family. A second CPU backend inherits the
// same borrowed-view dispatch; provider specialization belongs in Leto.
impl<T, B> CtcOps<T> for B
where
    T: Scalar + leto_ops::RealScalar,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    type CtcState = CtcState<T>;

    fn ctc_forward(
        &self,
        log_probs: &Self::DeviceBuffer<T>,
        log_probs_layout: &Layout,
        batch: CtcBatch<'_>,
        loss: &mut Self::DeviceBuffer<T>,
        loss_layout: &Layout,
    ) -> Result<Self::CtcState, Self::Error> {
        sequence_rank("ctc_log_probs", log_probs_layout)?;
        scalar_shape("ctc_loss", loss_layout)?;
        let mut loss = to_leto_view_mut::<T, 1>(loss_layout, loss.as_mut_slice())
            .map_err(|error| map_leto_error("ctc_loss", error))?;
        let destination = loss
            .get_mut([0])
            .expect("invariant: the validated scalar view contains coordinate zero");
        let log_probs = to_leto_view::<T, 3>(log_probs_layout, log_probs.as_slice())
            .map_err(|error| map_leto_error("ctc_log_probs", error))?;
        let CtcBatch {
            targets,
            input_lengths,
            target_lengths,
            blank,
        } = batch;
        let state = CtcState::forward(&log_probs, targets, input_lengths, target_lengths, blank)
            .map_err(|error| ctc_error("ctc_forward", error))?;
        *destination = state.loss();
        Ok(state)
    }

    fn ctc_backward_accumulate(
        &self,
        state: &Self::CtcState,
        output_gradient: &Self::DeviceBuffer<T>,
        output_gradient_layout: &Layout,
        gradient: &mut Self::DeviceBuffer<T>,
        gradient_layout: &Layout,
    ) -> Result<(), Self::Error> {
        sequence_rank("ctc_gradient", gradient_layout)?;
        scalar_shape("ctc_output_gradient", output_gradient_layout)?;
        let output_gradient =
            to_leto_view::<T, 1>(output_gradient_layout, output_gradient.as_slice())
                .map_err(|error| map_leto_error("ctc_output_gradient", error))?;
        let upstream = *output_gradient
            .get([0])
            .expect("invariant: the validated scalar view contains coordinate zero");
        let mut gradient = to_leto_view_mut::<T, 3>(gradient_layout, gradient.as_mut_slice())
            .map_err(|error| map_leto_error("ctc_gradient", error))?;
        state
            .backward_accumulate(upstream, &mut gradient)
            .map_err(|error| ctc_error("ctc_backward", error))
    }
}
