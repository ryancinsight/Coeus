//! Class-label and distribution losses.
//!
//! Probabilistic losses over class structure: cross-entropy against class
//! indices, the binary cross-entropy family, negative log-likelihood over
//! log-probabilities, the multi-label soft-margin alias, and the KL
//! divergence between a target and a log-probability field.

use coeus_autograd::Var;
use coeus_core::{Float, FloatElement};
use coeus_tensor::Tensor;

/// Cross-entropy loss (for classification with logits).
///
/// Logits shape: `[N, C]` where N is batch size, C is number of classes.
/// Targets: slice of N target indices in `[0, C)`.
/// Returns a scalar Var (shape `[1]`).
///
/// # Errors
///
/// Returns a typed validation error for an invalid rank, target count, empty
/// class axis, or out-of-range target. Provider preparation and dispatch
/// failures propagate without registering a partial autograd node.
pub fn cross_entropy_loss<T, B>(
    logits: &Var<T, B>,
    targets: &[usize],
) -> Result<Var<T, B>, B::Error>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + coeus_ops::CrossEntropyOps<T> + Default,
{
    let shape = logits.tensor.shape();
    if shape.len() != 2 {
        return Err(coeus_core::BackendError::UnsupportedRank {
            operation: "cross_entropy_forward",
            rank: shape.len(),
            max_rank: 2,
        }
        .into());
    }
    let n = shape[0];
    let c = shape[1];
    if n == 0 {
        return Err(coeus_core::BackendError::EmptyDimension {
            operation: "cross_entropy_forward",
            dimension: "batch",
        }
        .into());
    }
    if c == 0 {
        return Err(coeus_core::BackendError::EmptyDimension {
            operation: "cross_entropy_forward",
            dimension: "class",
        }
        .into());
    }
    if targets.len() != n {
        return Err(coeus_core::BackendError::ShapeMismatch {
            operation: "cross_entropy_forward",
            lhs: vec![targets.len()],
            rhs: vec![n],
        }
        .into());
    }
    if let Some((position, &index)) = targets.iter().enumerate().find(|(_, index)| **index >= c) {
        return Err(coeus_core::BackendError::IndexOutOfRange {
            operation: "cross_entropy_target",
            position,
            index,
            bound: c,
        }
        .into());
    }
    let backend = B::default();
    let saved_targets = backend.prepare_cross_entropy_targets(targets)?;
    let mut output = Tensor::alloc_on([1], &backend);
    let mut probabilities = Tensor::alloc_on([n, c], &backend);
    let (output_storage, output_layout) = output.storage_mut_and_layout();
    let (probability_storage, probability_layout) = probabilities.storage_mut_and_layout();
    backend.cross_entropy_forward(
        logits.tensor.storage(),
        logits.tensor.layout(),
        &saved_targets,
        output_storage,
        output_layout,
        probability_storage,
        probability_layout,
    )?;

    Ok(coeus_autograd::cross_entropy_loss(
        logits,
        saved_targets,
        output,
        probabilities,
    ))
}

/// Binary Cross-Entropy Loss.
/// pred: `[N]` probabilities, target: `[N]` float targets (0.0 or 1.0).
/// eps: clamp for numerical stability (e.g., 1e-7 as T).
#[inline]
pub fn binary_cross_entropy<
    T: coeus_core::FloatElement + Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
    eps: T,
) -> Var<T, B> {
    coeus_autograd::binary_cross_entropy(pred, target, eps)
}

/// Binary Cross-Entropy with logits (numerically stable sigmoid + BCE).
/// logits and target share shape; reduces with `mean`. Mirrors PyTorch
/// `BCEWithLogitsLoss(reduction="mean")`.
#[inline]
pub fn bce_with_logits<
    T: coeus_core::FloatElement + Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    logits: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::bce_with_logits(logits, target)
}

/// Negative Log-Likelihood Loss.
/// log_probs: `[N, C]` log-probabilities, targets: `[N]` class indices.
#[inline]
pub fn nll_loss<
    T: Float + FloatElement + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    log_probs: &Var<T, B>,
    targets: &[usize],
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    coeus_autograd::nll_loss(log_probs, targets)
}

/// Multi-label soft-margin loss (PyTorch `MultiLabelSoftMarginLoss` with
/// `reduction="mean"`).
///
/// Computes per-label sigmoid binary cross-entropy averaged over all elements.
/// Mathematically identical to `BCEWithLogitsLoss` when targets are binary.
/// Delegates to `bce_with_logits` directly.
#[inline]
pub fn multi_label_soft_margin_loss<
    T: coeus_core::FloatElement + Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    x: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::bce_with_logits(x, target)
}

/// KL divergence loss.
///
/// `input` is log-probabilities and `target` is probabilities. Computes
/// `mean(target * (log(target) - input))`.
#[inline]
pub fn kl_divergence<
    T: coeus_core::FloatElement + Float + coeus_leto::RealScalar,
    B: coeus_ops::BackendOps<T> + Default,
>(
    input: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::kl_divergence(input, target)
}
