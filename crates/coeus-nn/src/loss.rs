// ── Loss functions ──

use coeus_autograd::Var;
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Float};
use coeus_tensor::Tensor;

/// Mean Squared Error loss.
///
/// Computes mean squared error between pred and target.
/// Returns a scalar Var (shape `[1]`).
#[inline]
pub fn mse_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    let diff = coeus_autograd::sub(pred, target);
    let sq = coeus_autograd::mul(&diff, &diff);
    coeus_autograd::mean(&sq)
}

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
pub fn binary_cross_entropy<T: Float, B: coeus_ops::BackendOps<T> + Default>(
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
pub fn bce_with_logits<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    logits: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::bce_with_logits(logits, target)
}

/// Negative Log-Likelihood Loss.
/// log_probs: `[N, C]` log-probabilities, targets: `[N]` class indices.
#[inline]
pub fn nll_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    log_probs: &Var<T, B>,
    targets: &[usize],
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    coeus_autograd::nll_loss(log_probs, targets)
}

/// Multi-class margin loss (PyTorch `MultiMarginLoss`, `reduction="mean"`).
/// x: `[N, C]` scores, targets: `[N]` class indices, p >= 1, margin.
/// Computes `mean_i (1/C) sum_{j != y_i} max(0, margin - x[i,y_i] + x[i,j])^p`.
#[inline]
pub fn multi_margin<
    T: Float,
    B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default,
>(
    x: &Var<T, B>,
    targets: &[usize],
    p: T,
    margin: T,
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    coeus_autograd::multi_margin(x, targets, p, margin)
}

/// Huber (Smooth L1) Loss.
/// pred: `[N]`, target: `[N]`, delta: huber threshold.
///
/// # Errors
///
/// Returns the backend error type when the input shapes differ, the reduction
/// is empty, or `delta` is non-finite or non-positive.
#[inline]
pub fn huber_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
    delta: T,
) -> Result<Var<T, B>, B::Error> {
    coeus_autograd::huber_loss(pred, target, delta)
}

/// L1 (mean absolute error) loss.
/// pred: `[N]`, target: `[N]`. Computes `mean(|pred - target|)`.
#[inline]
pub fn l1_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::l1_loss(pred, target)
}

/// Poisson negative-log-likelihood loss (log-input form).
/// input holds `log(λ)`, target the observed counts; both share shape.
/// Computes `mean(exp(input) - target * input)` (PyTorch
/// `PoissonNLLLoss(log_input=True, full=False)`).
#[inline]
pub fn poisson_nll<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::poisson_nll(input, target)
}

/// Soft-margin (logistic) loss. input: `[..]`, target: `[..]` in `{-1, +1}`.
/// Computes `mean(log(1 + exp(-target * input)))` (PyTorch `SoftMarginLoss`).
#[inline]
pub fn soft_margin<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::soft_margin(input, target)
}

/// Row-wise p-norm pairwise distance (PyTorch `PairwiseDistance`).
/// x1, x2: `[N, D]`; returns `[N]` with `out_i = (sum_k |x1-x2|^p + eps)^(1/p)`.
#[inline]
pub fn pairwise_distance<
    T: Float,
    B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default,
>(
    x1: &Var<T, B>,
    x2: &Var<T, B>,
    p: T,
    eps: T,
) -> Var<T, B> {
    coeus_autograd::pairwise_distance(x1, x2, p, eps)
}

/// Triplet-margin loss (PyTorch `TripletMarginLoss`, `reduction="mean"`):
/// `mean_i max(0, d(a_i, p_i) - d(a_i, n_i) + margin)` where `d` is the
/// row-wise p-norm [`pairwise_distance`]. anchor/positive/negative share shape
/// `[N, D]`. Composed from the tracked pairwise-distance, subtract, shift, ReLU,
/// and mean ops, so backward (including the anchor's two gradient paths) is the
/// autograd graph's — no bespoke node.
#[inline]
pub fn triplet_margin_loss<
    T: Float,
    B: coeus_ops::BackendOps<T> + coeus_ops::ScalarPowerOps<T> + Default,
>(
    anchor: &Var<T, B>,
    positive: &Var<T, B>,
    negative: &Var<T, B>,
    margin: T,
    p: T,
    eps: T,
) -> Var<T, B> {
    let d_ap = coeus_autograd::pairwise_distance(anchor, positive, p, eps);
    let d_an = coeus_autograd::pairwise_distance(anchor, negative, p, eps);
    let shifted = coeus_autograd::scalar_add(&coeus_autograd::sub(&d_ap, &d_an), margin);
    coeus_autograd::mean(&coeus_autograd::relu(&shifted))
}

/// KL divergence loss.
///
/// `input` is log-probabilities and `target` is probabilities. Computes
/// `mean(target * (log(target) - input))`.
#[inline]
pub fn kl_divergence<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::kl_divergence(input, target)
}

/// Margin ranking loss.
///
/// `target` contains `+1` or `-1` labels. Computes
/// `mean(max(0, -target * (input1 - input2) + margin))`.
#[inline]
pub fn margin_ranking_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input1: &Var<T, B>,
    input2: &Var<T, B>,
    target: &[T],
    margin: T,
) -> Var<T, B> {
    coeus_autograd::margin_ranking_loss(input1, input2, target, margin)
}

/// Cosine Embedding Loss.
/// x1: `[N, D]`, x2: `[N, D]`, y: `[N]`, margin: threshold.
#[inline]
pub fn cosine_embedding_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x1: &Var<T, B>,
    x2: &Var<T, B>,
    y: &[T],
    margin: T,
) -> Var<T, B> {
    coeus_autograd::cosine_embedding_loss(x1, x2, y, margin)
}

/// Smooth L1 (Huber-β) loss (PyTorch
/// `SmoothL1Loss(reduction="mean", beta=float)`). Computes
/// `mean_i loss_smooth(pred[i] - target[i], beta)` with
/// `loss_smooth(z, β) = 0.5 z²/β` if `|z| < β`, else `|z| - 0.5 β`.
/// `pred` and `target` must share shape.
#[inline]
pub fn smooth_l1_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
    beta: T,
) -> Var<T, B> {
    coeus_autograd::smooth_l1_loss(pred, target, beta)
}

/// Row-wise cosine similarity along `dim=1`
/// (PyTorch `F.cosine_similarity(x1, x2, dim=1, eps=...)`).
/// `x1` and `x2` must share shape `[N, D]`; returns `[N]` where
/// `out_i = <x1_i, x2_i> / max(||x1_i|| * ||x2_i||, eps)`.
///
/// # Panics
///
/// Panics when the inputs do not share a two-dimensional non-empty shape,
/// `dim` is not one, or `eps` is not finite and strictly positive.
#[must_use]
#[inline]
pub fn cosine_similarity<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x1: &Var<T, B>,
    x2: &Var<T, B>,
    dim: usize,
    eps: T,
) -> Var<T, B> {
    coeus_autograd::cosine_similarity(x1, x2, dim, eps)
}

/// Hinge embedding loss (PyTorch `HingeEmbeddingLoss` with `reduction="mean"`).
///
/// `target` contains `+1` or `-1`. For `y_i == 1`, computes `mean(max(0, margin - x_i))`;
/// for `y_i == -1`, computes `mean(max(0, -x_i))`.
///
/// Composed from `where_cond`, `relu`, `neg`, and `scalar_sub` — no dedicated autograd node.
#[inline]
pub fn hinge_embedding_loss<T, B>(x: &Var<T, B>, target: &[T], margin: T) -> Var<T, B>
where
    T: Float,
    B: ComputeBackend + Default + coeus_ops::BackendOps<T>,
    B::DeviceBuffer<T>: CpuAddressableStorage<T> + CpuAddressableStorageMut<T>,
{
    assert_eq!(
        target.len(),
        x.tensor.numel(),
        "target length must match input length"
    );
    let backend = B::default();
    let zero = T::zero();

    let mask_data: Vec<T> = target
        .iter()
        .map(|&y| if y > zero { T::one() } else { zero })
        .collect();
    let mask_tensor = Tensor::from_slice_on(x.tensor.shape(), &mask_data, &backend);
    let mask_var = Var::new(mask_tensor, false);

    // PyTorch HingeEmbeddingLoss: target = +1 → loss = x (identity, no clamp);
    // target = -1 → loss = max(0, margin - x). `mask` is 1 where target > 0, so
    // `where_cond` selects the identity branch there and the hinge branch (the
    // `-1` case) otherwise.
    let hinge = coeus_autograd::relu(&coeus_autograd::neg(&coeus_autograd::scalar_sub(x, margin)));
    let selected = coeus_autograd::where_cond(&mask_var, x, &hinge);
    coeus_autograd::mean(&selected)
}

/// Multi-label soft-margin loss (PyTorch `MultiLabelSoftMarginLoss` with
/// `reduction="mean"`).
///
/// Computes per-label sigmoid binary cross-entropy averaged over all elements.
/// Mathematically identical to `BCEWithLogitsLoss` when targets are binary.
/// Delegates to `bce_with_logits` directly.
#[inline]
pub fn multi_label_soft_margin_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    target: &Var<T, B>,
) -> Var<T, B> {
    coeus_autograd::bce_with_logits(x, target)
}

/// Triplet margin loss with pluggable distance function
/// (PyTorch `TripletMarginWithDistanceLoss`, `reduction="mean"`).
///
/// Generalizes `triplet_margin_loss` by accepting a custom distance function.
/// `distance(a, p)` and `distance(a, n)` are computed via the provided closure.
/// Returns `mean(max(0, d_ap - d_an + margin))`.
pub fn triplet_margin_with_distance_loss<T, B, F>(
    anchor: &Var<T, B>,
    positive: &Var<T, B>,
    negative: &Var<T, B>,
    distance: F,
    margin: T,
) -> Var<T, B>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    F: Fn(&Var<T, B>, &Var<T, B>) -> Var<T, B>,
{
    let d_ap = distance(anchor, positive);
    let d_an = distance(anchor, negative);
    let shifted = coeus_autograd::scalar_add(&coeus_autograd::sub(&d_ap, &d_an), margin);
    coeus_autograd::mean(&coeus_autograd::relu(&shifted))
}

/// Multi-label margin loss (PyTorch `MultiLabelMarginLoss` with `reduction="mean"`).
///
/// `x`: shape `(N, C)` scores, `target`: shape `(N, C)` where
/// `target[i][j] >= 0` are valid class indices and `-1` means ignore padding.
/// Computes `mean_i sum_{t: target[i][t] >= 0} sum_{j != t} max(0, 1 - (x[i][t] - x[i][j]))`.
#[inline]
pub fn multi_label_margin_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    target: &[isize],
) -> Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    coeus_autograd::multi_label_margin_loss(x, target)
}

/// Gaussian negative-log-likelihood loss (PyTorch `GaussianNLLLoss` with
/// `reduction="mean"` and `full=False`).
///
/// `input`, `target`, and `var` share shape. Computes:
/// `loss = 0.5 * mean((input - target)^2 / var + log(var))`
///
/// Composed from existing autograd ops. When `full=true`, adds `0.5 * log(2π)`.
#[inline]
pub fn gaussian_nll_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    input: &Var<T, B>,
    target: &Var<T, B>,
    var: &Var<T, B>,
    full: bool,
) -> Var<T, B> {
    let diff = coeus_autograd::sub(input, target);
    let diff_sq = coeus_autograd::mul(&diff, &diff);
    let var_term = coeus_autograd::div(&diff_sq, var);
    let log_var = coeus_autograd::log(var);
    let loss =
        coeus_autograd::scalar_mul(&coeus_autograd::add(&var_term, &log_var), T::from_f64(0.5));
    if full {
        let two_pi = T::from_f64(2.0 * std::f64::consts::PI);
        coeus_autograd::scalar_add(
            &coeus_autograd::mean(&loss),
            T::from_f64(0.5) * two_pi.log_op(),
        )
    } else {
        coeus_autograd::mean(&loss)
    }
}

/// CTC (Connectionist Temporal Classification) loss.
///
/// Computes the negative log-likelihood of a sequence labeling task where the
/// alignment between input frames and output labels is unknown (e.g. speech
/// recognition). Matches `torch.nn.functional.ctc_loss` with
/// `reduction='mean'` and `zero_infinity=False`.
///
/// # Arguments
/// - `log_probs` — `[T, N, C]` log-probabilities (output of `log_softmax`).
/// - `targets` — flat target indices `[sum(target_lengths)]` (no padding).
/// - `input_lengths` — `[N]` valid frame count per sample (<= T).
/// - `target_lengths` — `[N]` target sequence length per sample.
/// - `blank` — blank class index (default 0 in PyTorch).
///
/// Returns a scalar `Var` (shape `[1]`) containing the mean CTC loss.
pub fn ctc_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    log_probs: &Var<T, B>,
    targets: &[usize],
    input_lengths: &[usize],
    target_lengths: &[usize],
    blank: usize,
) -> Var<T, B>
where
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    coeus_autograd::ctc_loss(log_probs, targets, input_lengths, target_lengths, blank)
}

/// Sum of all finite elements, treating NaN as zero (`torch.nansum`).
///
/// Returns a scalar `Var` (shape `[1]`).
pub fn nansum<T: coeus_core::Float, B: coeus_ops::BackendOps<T> + Default>(
    x: &coeus_autograd::Var<T, B>,
) -> coeus_autograd::Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    coeus_autograd::nansum(x)
}

/// Mean of all finite elements, treating NaN as missing (`torch.nanmean`).
///
/// Returns a scalar `Var` (shape `[1]`).
pub fn nanmean<T: coeus_core::Float, B: coeus_ops::BackendOps<T> + Default>(
    x: &coeus_autograd::Var<T, B>,
) -> coeus_autograd::Var<T, B>
where
    B::DeviceBuffer<T>:
        coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
{
    coeus_autograd::nanmean(x)
}
