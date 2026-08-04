use super::super::CpuBackend;
use crate::CrossEntropyOps;
use coeus_core::{BackendError, CpuAddressableStorage, CpuAddressableStorageMut, Layout, Scalar};
use coeus_leto::{to_leto_view, to_leto_view_mut};

fn provider_error(operation: &'static str, source: impl std::fmt::Display) -> BackendError {
    BackendError::Storage {
        operation,
        reason: source.to_string(),
    }
}

fn cross_entropy_error(source: leto_ops::CrossEntropyError) -> BackendError {
    use leto_ops::CrossEntropyError;

    match source {
        CrossEntropyError::Shape {
            expected, actual, ..
        } => BackendError::ShapeMismatch {
            operation: "cross_entropy",
            lhs: actual,
            rhs: expected,
        },
        CrossEntropyError::EmptyBatch => BackendError::EmptyDimension {
            operation: "cross_entropy",
            dimension: "batch",
        },
        CrossEntropyError::EmptyClasses => BackendError::EmptyDimension {
            operation: "cross_entropy",
            dimension: "class",
        },
        CrossEntropyError::TargetCount { expected, actual } => BackendError::ShapeMismatch {
            operation: "cross_entropy_targets",
            lhs: vec![actual],
            rhs: vec![expected],
        },
        CrossEntropyError::TargetOutOfRange {
            batch,
            target,
            classes,
        } => BackendError::IndexOutOfRange {
            operation: "cross_entropy_target",
            position: batch,
            index: target,
            bound: classes,
        },
        CrossEntropyError::Layout { source, .. } => provider_error("cross_entropy", source),
        source @ (CrossEntropyError::ScalarExtent { .. }
        | CrossEntropyError::ProbabilityResolution { .. }
        | CrossEntropyError::NonFinite { .. }
        | CrossEntropyError::ArithmeticNonFinite { .. }
        | CrossEntropyError::InvalidProbabilities { .. }) => BackendError::InvalidNumericInput {
            operation: "cross_entropy",
            reason: source.to_string(),
        },
        source => BackendError::InvalidNumericInput {
            operation: "cross_entropy",
            reason: source.to_string(),
        },
    }
}

impl<T, B> CrossEntropyOps<T> for B
where
    T: Scalar + leto_ops::RealScalar + eunomia::RealField,
    B: CpuBackend,
    B::DeviceBuffer<T>: CpuAddressableStorageMut<T>,
{
    type Targets = Box<[usize]>;

    fn prepare_cross_entropy_targets(
        &self,
        targets: &[usize],
    ) -> Result<Self::Targets, Self::Error> {
        Ok(targets.into())
    }

    fn cross_entropy_forward(
        &self,
        logits: &Self::DeviceBuffer<T>,
        logits_layout: &Layout,
        targets: &Self::Targets,
        loss: &mut Self::DeviceBuffer<T>,
        loss_layout: &Layout,
        probabilities: &mut Self::DeviceBuffer<T>,
        probabilities_layout: &Layout,
    ) -> Result<(), Self::Error> {
        let logits = to_leto_view::<T, 2>(logits_layout, logits.as_slice())
            .map_err(|source| provider_error("cross_entropy_logits", source))?;
        let mut loss = to_leto_view_mut::<T, 1>(loss_layout, loss.as_mut_slice())
            .map_err(|source| provider_error("cross_entropy_loss", source))?;
        let mut probabilities =
            to_leto_view_mut::<T, 2>(probabilities_layout, probabilities.as_mut_slice())
                .map_err(|source| provider_error("cross_entropy_probabilities", source))?;
        leto_ops::cross_entropy_forward_into(&logits, targets, &mut loss, &mut probabilities)
            .map_err(cross_entropy_error)
    }

    fn cross_entropy_backward_accumulate(
        &self,
        output_gradient: &Self::DeviceBuffer<T>,
        output_gradient_layout: &Layout,
        probabilities: &Self::DeviceBuffer<T>,
        probabilities_layout: &Layout,
        targets: &Self::Targets,
        logit_gradient: &mut Self::DeviceBuffer<T>,
        logit_gradient_layout: &Layout,
    ) -> Result<(), Self::Error> {
        let output_gradient =
            to_leto_view::<T, 1>(output_gradient_layout, output_gradient.as_slice())
                .map_err(|source| provider_error("cross_entropy_output_gradient", source))?;
        let probabilities = to_leto_view::<T, 2>(probabilities_layout, probabilities.as_slice())
            .map_err(|source| provider_error("cross_entropy_probabilities", source))?;
        let mut logit_gradient =
            to_leto_view_mut::<T, 2>(logit_gradient_layout, logit_gradient.as_mut_slice())
                .map_err(|source| provider_error("cross_entropy_logit_gradient", source))?;
        leto_ops::cross_entropy_backward_accumulate(
            &output_gradient,
            &probabilities,
            targets,
            &mut logit_gradient,
        )
        .map_err(cross_entropy_error)
    }
}
