use crate::tensor::PyTensor;
use pyo3::prelude::*;

/// Mean Squared Error loss.
#[pyfunction]
pub fn mse_loss(pred: &PyTensor, target: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::mse_loss(&pred.inner, &target.inner));
    PyTensor::from_var(inner)
}

/// Cross-entropy loss.
#[pyfunction]
pub fn cross_entropy_loss(logits: &PyTensor, targets: Vec<usize>, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::cross_entropy_loss(&logits.inner, &targets));
    PyTensor::from_var(inner)
}

/// Binary Cross-Entropy Loss.
#[pyfunction]
#[pyo3(signature = (pred, target, eps = 1e-7))]
pub fn binary_cross_entropy(
    pred: &PyTensor,
    target: &PyTensor,
    eps: f64,
    py: Python<'_>,
) -> PyTensor {
    let inner =
        py.allow_threads(|| coeus_nn::loss::binary_cross_entropy(&pred.inner, &target.inner, eps));
    PyTensor::from_var(inner)
}

/// Negative Log-Likelihood Loss.
#[pyfunction]
pub fn nll_loss(log_probs: &PyTensor, targets: Vec<usize>, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::nll_loss(&log_probs.inner, &targets));
    PyTensor::from_var(inner)
}

/// Huber Loss.
#[pyfunction]
#[pyo3(signature = (pred, target, delta = 1.0))]
pub fn huber_loss(pred: &PyTensor, target: &PyTensor, delta: f64, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::huber_loss(&pred.inner, &target.inner, delta));
    PyTensor::from_var(inner)
}

/// KL divergence loss.
#[pyfunction]
pub fn kl_divergence(input: &PyTensor, target: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::kl_divergence(&input.inner, &target.inner));
    PyTensor::from_var(inner)
}

/// Margin ranking loss.
#[pyfunction]
#[pyo3(signature = (input1, input2, target, margin = 0.0))]
pub fn margin_ranking_loss(
    input1: &PyTensor,
    input2: &PyTensor,
    target: Vec<f64>,
    margin: f64,
    py: Python<'_>,
) -> PyTensor {
    let inner = py.allow_threads(|| {
        coeus_nn::loss::margin_ranking_loss(&input1.inner, &input2.inner, &target, margin)
    });
    PyTensor::from_var(inner)
}

/// Cosine Embedding Loss.
#[pyfunction]
#[pyo3(signature = (x1, x2, y, margin = 0.0))]
pub fn cosine_embedding_loss(
    x1: &PyTensor,
    x2: &PyTensor,
    y: Vec<f64>,
    margin: f64,
    py: Python<'_>,
) -> PyTensor {
    let inner = py
        .allow_threads(|| coeus_nn::loss::cosine_embedding_loss(&x1.inner, &x2.inner, &y, margin));
    PyTensor::from_var(inner)
}

/// L1 (mean absolute error) loss: `mean(|pred - target|)`.
#[pyfunction]
pub fn l1_loss(pred: &PyTensor, target: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::l1_loss(&pred.inner, &target.inner));
    PyTensor::from_var(inner)
}

/// Binary cross-entropy with logits (numerically stable). Mirrors PyTorch
/// `BCEWithLogitsLoss(reduction="mean")`.
#[pyfunction]
pub fn bce_with_logits(logits: &PyTensor, target: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::bce_with_logits(&logits.inner, &target.inner));
    PyTensor::from_var(inner)
}

/// Poisson NLL loss (log-input): `mean(exp(input) - target * input)`.
#[pyfunction]
pub fn poisson_nll(input: &PyTensor, target: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::poisson_nll(&input.inner, &target.inner));
    PyTensor::from_var(inner)
}

/// Soft-margin (logistic) loss: `mean(log(1 + exp(-target * input)))`.
#[pyfunction]
pub fn soft_margin(input: &PyTensor, target: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::soft_margin(&input.inner, &target.inner));
    PyTensor::from_var(inner)
}

/// Row-wise p-norm pairwise distance: `[N,D] -> [N]`.
#[pyfunction]
#[pyo3(signature = (x1, x2, p = 2.0, eps = 1e-6))]
pub fn pairwise_distance(
    x1: &PyTensor,
    x2: &PyTensor,
    p: f64,
    eps: f64,
    py: Python<'_>,
) -> PyTensor {
    let inner =
        py.allow_threads(|| coeus_nn::loss::pairwise_distance(&x1.inner, &x2.inner, p, eps));
    PyTensor::from_var(inner)
}

/// Triplet-margin loss: `mean max(0, d(a,p) - d(a,n) + margin)`.
#[pyfunction]
#[pyo3(signature = (anchor, positive, negative, margin = 1.0, p = 2.0, eps = 1e-6))]
pub fn triplet_margin_loss(
    anchor: &PyTensor,
    positive: &PyTensor,
    negative: &PyTensor,
    margin: f64,
    p: f64,
    eps: f64,
    py: Python<'_>,
) -> PyTensor {
    let inner = py.allow_threads(|| {
        coeus_nn::loss::triplet_margin_loss(
            &anchor.inner,
            &positive.inner,
            &negative.inner,
            margin,
            p,
            eps,
        )
    });
    PyTensor::from_var(inner)
}

/// Multi-class margin loss over scores `[N, C]` with class-index targets.
#[pyfunction]
#[pyo3(signature = (x, targets, p = 1.0, margin = 1.0))]
pub fn multi_margin(
    x: &PyTensor,
    targets: Vec<usize>,
    p: f64,
    margin: f64,
    py: Python<'_>,
) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::multi_margin(&x.inner, &targets, p, margin));
    PyTensor::from_var(inner)
}

/// Smooth L1 (Huber-β) loss: `mean(loss_smooth(pred - target, beta))`.
#[pyfunction]
#[pyo3(signature = (pred, target, beta = 1.0))]
pub fn smooth_l1_loss(pred: &PyTensor, target: &PyTensor, beta: f64, py: Python<'_>) -> PyTensor {
    let inner =
        py.allow_threads(|| coeus_nn::loss::smooth_l1_loss(&pred.inner, &target.inner, beta));
    PyTensor::from_var(inner)
}

/// Row-wise cosine similarity along `dim=1`.
#[pyfunction]
#[pyo3(signature = (x1, x2, dim = 1, eps = 1e-8))]
pub fn cosine_similarity(
    x1: &PyTensor,
    x2: &PyTensor,
    dim: usize,
    eps: f64,
    py: Python<'_>,
) -> PyTensor {
    let inner =
        py.allow_threads(|| coeus_nn::loss::cosine_similarity(&x1.inner, &x2.inner, dim, eps));
    PyTensor::from_var(inner)
}

/// Hinge embedding loss: targets in {-1, +1}, margin threshold.
#[pyfunction]
#[pyo3(signature = (x, target, margin = 1.0))]
pub fn hinge_embedding_loss(
    x: &PyTensor,
    target: Vec<f64>,
    margin: f64,
    py: Python<'_>,
) -> PyTensor {
    let inner =
        py.allow_threads(|| coeus_nn::loss::hinge_embedding_loss(&x.inner, &target, margin));
    PyTensor::from_var(inner)
}

/// Multi-label margin loss (hinge-based multi-label ranking).
#[pyfunction]
pub fn multi_label_margin_loss(x: &PyTensor, target: Vec<isize>, py: Python<'_>) -> PyTensor {
    let inner = py.allow_threads(|| coeus_nn::loss::multi_label_margin_loss(&x.inner, &target));
    PyTensor::from_var(inner)
}

/// Multi-label soft-margin loss (sigmoid + BCE per label).
#[pyfunction]
pub fn multi_label_soft_margin_loss(x: &PyTensor, target: &PyTensor, py: Python<'_>) -> PyTensor {
    let inner =
        py.allow_threads(|| coeus_nn::loss::multi_label_soft_margin_loss(&x.inner, &target.inner));
    PyTensor::from_var(inner)
}

/// Gaussian negative-log-likelihood loss.
#[pyfunction]
#[pyo3(signature = (input, target, var, full = false))]
pub fn gaussian_nll_loss(
    input: &PyTensor,
    target: &PyTensor,
    var: &PyTensor,
    full: bool,
    py: Python<'_>,
) -> PyTensor {
    let inner = py.allow_threads(|| {
        coeus_nn::loss::gaussian_nll_loss(&input.inner, &target.inner, &var.inner, full)
    });
    PyTensor::from_var(inner)
}

/// CTC (Connectionist Temporal Classification) loss.
///
/// `log_probs`: `[T, N, C]` log-probability tensor (output of `log_softmax`).
/// `targets`: flat list of target class indices across all samples.
/// `input_lengths`: list of valid frame counts per sample.
/// `target_lengths`: list of target sequence lengths per sample.
/// `blank`: index of the blank class (default 0).
///
/// Returns a scalar `Tensor` holding the mean CTC loss.
#[pyfunction]
#[pyo3(signature = (log_probs, targets, input_lengths, target_lengths, blank = 0))]
pub fn ctc_loss(
    log_probs: &PyTensor,
    targets: Vec<usize>,
    input_lengths: Vec<usize>,
    target_lengths: Vec<usize>,
    blank: usize,
    py: Python<'_>,
) -> PyTensor {
    let inner = py.allow_threads(|| {
        coeus_nn::loss::ctc_loss(
            &log_probs.inner,
            &targets,
            &input_lengths,
            &target_lengths,
            blank,
        )
    });
    PyTensor::from_var(inner)
}
