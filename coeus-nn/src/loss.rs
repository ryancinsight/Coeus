// ── Loss functions ──

use coeus_autograd::Var;
use coeus_core::{Float, Storage};
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
