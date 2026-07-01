// ── Loss functions ──

use coeus_autograd::Var;
use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut, Float, Storage};
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
pub fn cross_entropy_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    logits: &Var<T, B>,
    targets: &[usize],
) -> Var<T, B> {
    let shape = logits.tensor.shape();
    assert_eq!(
        shape.len(),
        2,
        "logits must be 2D matrix [batch_size, num_classes]"
    );
    let n = shape[0];
    let c = shape[1];
    assert_eq!(targets.len(), n, "targets length must match batch size");
    let backend = B::default();

    // Ensure logits are contiguous before copying to host
    let temp_logits;
    let logits_cont = if logits.tensor.is_contiguous() && logits.tensor.layout().offset() == 0 {
        &logits.tensor
    } else {
        temp_logits = logits.tensor.to_contiguous_on(&backend);
        &temp_logits
    };

    let host_data = if let Some(slice) = logits_cont.storage().try_as_slice() {
        std::borrow::Cow::Borrowed(&slice[..logits_cont.numel()])
    } else {
        let mut l_data = vec![T::zero(); logits_cont.numel()];
        backend.copy_to_host(logits_cont.storage(), &mut l_data);
        std::borrow::Cow::Owned(l_data)
    };

    // Compute log-sum-exp in T precision — no widening to f64
    let mut loss_val = T::zero();
    let mut probs = vec![T::zero(); n * c];

    for i in 0..n {
        let offset = i * c;
        // Find max for numerical stability (log-sum-exp subtraction trick)
        let mut max_val = host_data[offset];
        for j in 1..c {
            let val = host_data[offset + j];
            if val > max_val {
                max_val = val;
            }
        }

        // Sum exp(x - max) in T precision
        let mut sum_exp = T::zero();
        for j in 0..c {
            let diff = host_data[offset + j] - max_val;
            let val_exp = diff.exp_op();
            probs[offset + j] = val_exp;
            sum_exp = sum_exp + val_exp;
        }

        // Compute sample loss: log(sum(exp(x_j))) - x_y, in T precision
        let log_sum_exp = sum_exp.log_op() + max_val;
        let target_idx = targets[i];
        assert!(target_idx < c, "target index out of bounds");
        let target_logit = host_data[offset + target_idx];
        loss_val = loss_val + (log_sum_exp - target_logit);

        // Compute softmax probabilities for backward pass
        for j in 0..c {
            probs[offset + j] = probs[offset + j] / sum_exp;
        }
    }

    loss_val = loss_val / T::from_f64(n as f64);
    let out_tensor = Tensor::from_slice_on([1], &[loss_val], &backend);

    coeus_autograd::cross_entropy_loss(logits, targets.to_vec(), out_tensor, probs, n, c)
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
) -> Var<T, B> {
    coeus_autograd::nll_loss(log_probs, targets)
}

/// Multi-class margin loss (PyTorch `MultiMarginLoss`, `reduction="mean"`).
/// x: `[N, C]` scores, targets: `[N]` class indices, p >= 1, margin.
/// Computes `mean_i (1/C) sum_{j != y_i} max(0, margin - x[i,y_i] + x[i,j])^p`.
#[inline]
pub fn multi_margin<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    x: &Var<T, B>,
    targets: &[usize],
    p: T,
    margin: T,
) -> Var<T, B> {
    coeus_autograd::multi_margin(x, targets, p, margin)
}

/// Huber (Smooth L1) Loss.
/// pred: `[N]`, target: `[N]`, delta: huber threshold.
#[inline]
pub fn huber_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
    pred: &Var<T, B>,
    target: &Var<T, B>,
    delta: T,
) -> Var<T, B> {
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
pub fn pairwise_distance<T: Float, B: coeus_ops::BackendOps<T> + Default>(
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
pub fn triplet_margin_loss<T: Float, B: coeus_ops::BackendOps<T> + Default>(
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
/// `out_i = <x1_i, x2_i> / (||x1_i|| * ||x2_i|| + eps)`.
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

    let candidate1 =
        coeus_autograd::relu(&coeus_autograd::neg(&coeus_autograd::scalar_sub(x, margin)));
    let candidate2 = coeus_autograd::relu(&coeus_autograd::neg(x));
    let selected = coeus_autograd::where_cond(&mask_var, &candidate1, &candidate2);
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
) -> Var<T, B> {
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
