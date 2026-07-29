use crate::{nn::error::map_module_error, tensor::PyTensor};
use coeus_tensor::Tensor;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(signature = (input, norm_shape, weight = None, bias = None, eps = 1e-5))]
pub fn layer_norm(
    input: &PyTensor,
    norm_shape: usize,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    eps: f64,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(PyValueError::new_err(
            "layer_norm: eps must be finite and non-negative",
        ));
    }
    let shape = input.inner.tensor.shape();
    if shape.len() < 2 {
        return Err(PyValueError::new_err(
            "layer_norm: input must have rank >= 2",
        ));
    }
    let last_dim = shape[shape.len() - 1];
    if norm_shape != last_dim {
        return Err(PyValueError::new_err(format!(
            "layer_norm: norm_shape ({norm_shape}) must match input last dimension ({last_dim})"
        )));
    }
    if let Some(w) = weight {
        if w.inner.tensor.shape() != [norm_shape] {
            return Err(PyValueError::new_err(
                "layer_norm: weight must have shape [norm_shape]",
            ));
        }
    }
    if let Some(b) = bias {
        if b.inner.tensor.shape() != [norm_shape] {
            return Err(PyValueError::new_err(
                "layer_norm: bias must have shape [norm_shape]",
            ));
        }
    }
    let w = weight.map(|w| w.inner.clone());
    let b = bias.map(|b| b.inner.clone());
    let x = input.inner.clone();
    let inner =
        py.allow_threads(move || coeus_nn::layer_norm(&x, norm_shape, w.as_ref(), b.as_ref(), eps));
    inner.map(PyTensor::from_var).map_err(map_module_error)
}

/// Functional (stateless) batch normalization for 3-D inputs `[N, C, L]`.
///
/// Matches `torch.nn.functional.batch_norm` with `input` of shape
/// `[N, C, L]`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    input,
    running_mean,
    running_var,
    weight = None,
    bias = None,
    training = false,
    momentum = 0.1,
    eps = 1e-5
))]
pub fn batch_norm_1d(
    input: &PyTensor,
    running_mean: &PyTensor,
    running_var: &PyTensor,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    training: bool,
    momentum: f64,
    eps: f64,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    let shape = input.inner.tensor.shape();
    let Some(&num_features) = shape.get(1) else {
        return Err(PyValueError::new_err(format!(
            "batch_norm_1d: input must have rank 2 or 3, got rank {}",
            shape.len()
        )));
    };
    if !matches!(shape.len(), 2 | 3) {
        return Err(PyValueError::new_err(format!(
            "batch_norm_1d: input must have rank 2 or 3, got rank {}",
            shape.len()
        )));
    }
    if running_mean.inner.tensor.shape() != [num_features]
        || running_var.inner.tensor.shape() != [num_features]
    {
        return Err(PyValueError::new_err(
            "batch_norm_1d: running statistics must have shape [C]",
        ));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(PyValueError::new_err(
            "batch_norm_1d: eps must be finite and non-negative",
        ));
    }
    if let Some(parameter) = weight {
        if parameter.inner.tensor.shape() != [num_features] {
            return Err(PyValueError::new_err(
                "batch_norm_1d: weight must have shape [C]",
            ));
        }
    }
    if let Some(parameter) = bias {
        if parameter.inner.tensor.shape() != [num_features] {
            return Err(PyValueError::new_err(
                "batch_norm_1d: bias must have shape [C]",
            ));
        }
    }
    let w = weight.map(|w| w.inner.clone());
    let b = bias.map(|b| b.inner.clone());
    let x = input.inner.clone();
    let rm = running_mean.inner.tensor.clone();
    let rv = running_var.inner.tensor.clone();
    let inner = py.allow_threads(move || {
        use coeus_nn::normalization::BatchNorm1d;
        use coeus_nn::Module;
        let backend = coeus_core::MoiraiBackend::new();
        let weight_var = w.unwrap_or_else(|| {
            coeus_autograd::Var::new(Tensor::ones_on([num_features], &backend), false)
        });
        let bias_var = b.unwrap_or_else(|| {
            coeus_autograd::Var::new(Tensor::zeros_on([num_features], &backend), false)
        });
        let mut bn =
            BatchNorm1d::from_parts(num_features, weight_var, bias_var, eps, momentum, rm, rv);
        bn.set_training(training);
        bn.forward(&x)
    });
    inner.map(PyTensor::from_var).map_err(map_module_error)
}

/// Functional (stateless) RMS normalization.
#[pyfunction]
#[pyo3(signature = (input, weight = None, eps = 1e-8))]
pub fn rms_norm(
    input: &PyTensor,
    weight: Option<&PyTensor>,
    eps: f64,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(PyValueError::new_err(
            "rms_norm: eps must be finite and non-negative",
        ));
    }
    let shape = input.inner.tensor.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "rms_norm: input must be rank-2 [N, D], got rank {}",
            shape.len()
        )));
    }
    let d = shape[1];
    if let Some(w) = weight {
        if w.inner.tensor.shape() != [d] {
            return Err(PyValueError::new_err(
                "rms_norm: weight must have shape [D]",
            ));
        }
    }
    let w = weight.map(|w| w.inner.clone());
    let x = input.inner.clone();
    let inner = py.allow_threads(move || coeus_nn::rms_norm(&x, w.as_ref(), eps));
    inner.map(PyTensor::from_var).map_err(map_module_error)
}

/// Functional (stateless) group normalization.
#[pyfunction]
#[pyo3(signature = (input, num_groups, weight = None, bias = None, eps = 1e-5))]
pub fn group_norm(
    input: &PyTensor,
    num_groups: usize,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    eps: f64,
    py: Python<'_>,
) -> PyResult<PyTensor> {
    let shape = input.inner.tensor.shape();
    if shape.len() < 2 {
        return Err(PyValueError::new_err(
            "group_norm: input must have at least 2 dimensions",
        ));
    }
    if num_groups == 0 {
        return Err(PyValueError::new_err(
            "group_norm: num_groups must be greater than 0",
        ));
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(PyValueError::new_err(
            "group_norm: eps must be finite and non-negative",
        ));
    }
    let channels = shape[1];
    if !channels.is_multiple_of(num_groups) {
        return Err(PyValueError::new_err(format!(
            "group_norm: channels ({channels}) must be divisible by num_groups ({num_groups})"
        )));
    }
    if let Some(w) = weight {
        if w.inner.tensor.shape() != [channels] {
            return Err(PyValueError::new_err(
                "group_norm: weight must have shape [C]",
            ));
        }
    }
    if let Some(b) = bias {
        if b.inner.tensor.shape() != [channels] {
            return Err(PyValueError::new_err(
                "group_norm: bias must have shape [C]",
            ));
        }
    }
    let x = input.inner.tensor.clone();
    let w = weight.map(|w| w.inner.tensor.clone());
    let b = bias.map(|b| b.inner.tensor.clone());
    py.allow_threads(move || coeus_nn::group_norm(&x, num_groups, w.as_ref(), b.as_ref(), eps))
        .map(|tensor| PyTensor::from_var(coeus_autograd::Var::new(tensor, false)))
        .map_err(map_module_error)
}
