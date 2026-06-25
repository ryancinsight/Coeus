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
