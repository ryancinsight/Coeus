use super::tensor::PyTensor;
use dtype::float::Float32;
use pyo3::prelude::Bound;
use pyo3::types::PyAny;
use pyo3::{pyfunction, PyErr, PyResult};

/// Linear function
#[pyfunction]
pub fn linear(input: &PyTensor, weight: &PyTensor, bias: Option<&PyTensor>) -> PyResult<PyTensor> {
    // Use the functional API from nn
    let result = nn::functional_api::linear(&input.inner, &weight.inner, bias.map(|b| &b.inner))
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Functional operation failed: {:?}",
                e
            ))
        })?;
    Ok(PyTensor { inner: result })
}

/// ReLU activation function
#[pyfunction]
pub fn relu(input: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional_api::relu(&input.inner).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("ReLU operation failed: {:?}", e))
    })?;
    Ok(PyTensor { inner: result })
}

/// Sigmoid activation function
#[pyfunction]
pub fn sigmoid(input: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional_api::sigmoid(&input.inner).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Sigmoid operation failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

/// Tanh activation function
#[pyfunction]
pub fn tanh(input: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional_api::tanh(&input.inner).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Tanh operation failed: {:?}", e))
    })?;
    Ok(PyTensor { inner: result })
}

/// GELU activation function
#[pyfunction]
pub fn gelu(input: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional_api::gelu(&input.inner).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Functional operation failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

/// SiLU (Swish) activation function
#[pyfunction]
pub fn silu(input: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional_api::silu(&input.inner).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Functional operation failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

/// Leaky ReLU activation function
#[pyfunction]
#[pyo3(signature = (input, negative_slope=0.01))]
pub fn leaky_relu(input: &PyTensor, negative_slope: f64) -> PyResult<PyTensor> {
    let slope = Float32::new(negative_slope as f32);
    let result = nn::functional_api::leaky_relu(&input.inner, slope).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Functional operation failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

/// ELU activation function
#[pyfunction]
#[pyo3(signature = (input, alpha=1.0))]
pub fn elu(input: &PyTensor, alpha: f64) -> PyResult<PyTensor> {
    let alpha_val = Float32::new(alpha as f32);
    let result = nn::functional_api::elu(&input.inner, alpha_val).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Functional operation failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

/// Mean Squared Error loss
#[pyfunction]
pub fn mse_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    let result = nn::functional_api::mse_loss(&input.inner, &target.inner).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Functional operation failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
#[pyo3(signature = (input, target, weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0))]
pub fn cross_entropy(
    input: &PyTensor,
    target: &PyTensor,
    weight: Option<&Bound<'_, PyAny>>,
    ignore_index: i64,
    reduction: &str,
    label_smoothing: f64,
) -> PyResult<PyTensor> {
    if weight.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "cross_entropy(weight=...) is not implemented",
        ));
    }

    if ignore_index != -100 {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "cross_entropy(ignore_index!= -100) is not implemented",
        ));
    }

    if reduction != "mean" {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "cross_entropy(reduction!= 'mean') is not implemented",
        ));
    }

    if label_smoothing != 0.0 {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "cross_entropy(label_smoothing!=0.0) is not implemented",
        ));
    }

    let result = nn::functional_api::cross_entropy(&input.inner, &target.inner).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Functional operation failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
#[pyo3(signature = (input, dim=None, dtype=None))]
pub fn softmax(
    input: &PyTensor,
    dim: Option<isize>,
    dtype: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTensor> {
    if dtype.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "softmax(dtype=...) is not implemented",
        ));
    }

    let dim = dim.unwrap_or(-1);
    let result = nn::functional_api::softmax_dim(&input.inner, dim).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Functional operation failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

/// Max pooling 2D
#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride=None, padding=None))]
pub fn max_pool2d(
    input: &PyTensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
) -> PyResult<PyTensor> {
    let result = nn::functional_api::max_pool2d(
        &input.inner,
        kernel_size,
        stride,
        padding.unwrap_or((0, 0)),
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("MaxPool2D failed: {:?}", e))
    })?;

    Ok(PyTensor { inner: result })
}

/// Average pooling 2D
#[pyfunction]
#[pyo3(signature = (input, kernel_size, stride=None, padding=None))]
pub fn avg_pool2d(
    input: &PyTensor,
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
) -> PyResult<PyTensor> {
    let result = nn::functional_api::avg_pool2d(
        &input.inner,
        kernel_size,
        stride,
        padding.unwrap_or((0, 0)),
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("AvgPool2D failed: {:?}", e))
    })?;

    Ok(PyTensor { inner: result })
}

#[pyfunction]
#[pyo3(signature = (input, p=0.5, training=true, inplace=false))]
pub fn dropout(input: &PyTensor, p: f64, training: bool, inplace: bool) -> PyResult<PyTensor> {
    if inplace {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "dropout(inplace=True) is not implemented",
        ));
    }

    let result = nn::functional_api::dropout(&input.inner, p, training).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Dropout operation failed: {:?}",
            e
        ))
    })?;
    Ok(PyTensor { inner: result })
}

/// Layer normalization function
#[pyfunction]
#[pyo3(signature = (input, normalized_shape, weight=None, bias=None, eps=1e-5))]
pub fn layer_norm(
    input: &PyTensor,
    normalized_shape: Vec<usize>,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    eps: Option<f64>,
) -> PyResult<PyTensor> {
    let result = nn::functional_api::layer_norm(
        &input.inner,
        &normalized_shape,
        weight.map(|w| &w.inner),
        bias.map(|b| &b.inner),
        eps.unwrap_or(1e-5),
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("LayerNorm failed: {:?}", e))
    })?;

    Ok(PyTensor { inner: result })
}

/// Binary cross-entropy with logits loss
#[pyfunction]
pub fn bce_with_logits_loss(input: &PyTensor, target: &PyTensor) -> PyResult<PyTensor> {
    let result =
        nn::functional_api::bce_with_logits_loss(&input.inner, &target.inner).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "BCEWithLogitsLoss failed: {:?}",
                e
            ))
        })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
#[pyo3(signature = (input, target, weight=None, ignore_index=-100, reduction="mean"))]
pub fn nll_loss(
    input: &PyTensor,
    target: &PyTensor,
    weight: Option<&Bound<'_, PyAny>>,
    ignore_index: i64,
    reduction: &str,
) -> PyResult<PyTensor> {
    if weight.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "nll_loss(weight=...) is not implemented",
        ));
    }

    if ignore_index != -100 {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "nll_loss(ignore_index!= -100) is not implemented",
        ));
    }

    if reduction != "mean" {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "nll_loss(reduction!= 'mean') is not implemented",
        ));
    }

    let result = nn::functional_api::nll_loss(&input.inner, &target.inner).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("nll_loss failed: {:?}", e))
    })?;
    Ok(PyTensor { inner: result })
}

#[pyfunction]
#[pyo3(signature = (input, running_mean=None, running_var=None, weight=None, bias=None, training=false, momentum=0.1, eps=1e-5))]
pub fn batch_norm(
    input: &PyTensor,
    running_mean: Option<&PyTensor>,
    running_var: Option<&PyTensor>,
    weight: Option<&PyTensor>,
    bias: Option<&PyTensor>,
    training: bool,
    momentum: f64,
    eps: f64,
) -> PyResult<PyTensor> {
    if running_mean.is_some() || running_var.is_some() {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "batch_norm with running statistics is not implemented",
        ));
    }

    if !training {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "batch_norm(training=False) is not implemented (requires running statistics)",
        ));
    }

    if (momentum - 0.1).abs() > 1e-12 {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "batch_norm(momentum!=0.1) is not implemented",
        ));
    }

    let result = nn::functional_api::batch_norm(
        &input.inner,
        weight.map(|w| &w.inner),
        bias.map(|b| &b.inner),
        eps,
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("batch_norm failed: {:?}", e))
    })?;

    Ok(PyTensor { inner: result })
}

/// Matrix multiplication
#[pyfunction]
pub fn matmul(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.matmul(other)
}

/// Batch matrix multiplication
#[pyfunction]
pub fn bmm(input: &PyTensor, other: &PyTensor) -> PyResult<PyTensor> {
    input.bmm(other)
}

/// Add matrix multiplication
#[pyfunction]
#[pyo3(signature = (input, mat1, mat2, beta=1.0, alpha=1.0))]
pub fn addmm(
    input: &PyTensor,
    mat1: &PyTensor,
    mat2: &PyTensor,
    beta: f32,
    alpha: f32,
) -> PyResult<PyTensor> {
    input.addmm(mat1, mat2, beta, alpha)
}

/// Reshape tensor
#[pyfunction]
pub fn reshape(input: &PyTensor, shape: Vec<isize>) -> PyResult<PyTensor> {
    input.reshape(shape)
}

/// View tensor (alias for reshape)
#[pyfunction]
pub fn view(input: &PyTensor, shape: Vec<isize>) -> PyResult<PyTensor> {
    input.view(shape)
}

/// Flatten tensor
#[pyfunction]
#[pyo3(signature = (input, start_dim=0, end_dim=-1))]
pub fn flatten(input: &PyTensor, start_dim: usize, end_dim: isize) -> PyResult<PyTensor> {
    input.flatten(start_dim, end_dim)
}

/// Squeeze tensor
#[pyfunction]
#[pyo3(signature = (input, dim=None))]
pub fn squeeze(input: &PyTensor, dim: Option<usize>) -> PyResult<PyTensor> {
    input.squeeze(dim)
}

/// Unsqueeze tensor
#[pyfunction]
pub fn unsqueeze(input: &PyTensor, dim: usize) -> PyResult<PyTensor> {
    input.unsqueeze(dim)
}

/// Transpose tensor
#[pyfunction]
pub fn transpose(input: &PyTensor, dim0: usize, dim1: usize) -> PyResult<PyTensor> {
    input.transpose(dim0, dim1)
}

/// Permute tensor
#[pyfunction]
pub fn permute(input: &PyTensor, dims: Vec<usize>) -> PyResult<PyTensor> {
    input.permute(dims)
}

/// 1D convolution
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None))]
pub fn conv1d(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    stride: Option<usize>,
    padding: Option<usize>,
) -> PyResult<PyTensor> {
    let result = nn::functional_api::conv1d(
        &input.inner,
        &weight.inner,
        bias.map(|b| &b.inner),
        stride.unwrap_or(1),
        padding.unwrap_or(0),
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Conv1d failed: {:?}", e))
    })?;

    Ok(PyTensor { inner: result })
}

/// 2D convolution
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None, dilation=None, groups=1))]
pub fn conv2d(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
    dilation: Option<(usize, usize)>,
    groups: usize,
) -> PyResult<PyTensor> {
    if dilation.unwrap_or((1, 1)) != (1, 1) {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "conv2d(dilation!=1) is not implemented",
        ));
    }
    if groups != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "conv2d(groups!=1) is not implemented",
        ));
    }

    let result = nn::functional_api::conv2d(
        &input.inner,
        &weight.inner,
        bias.map(|b| &b.inner),
        stride,
        padding,
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Conv2d failed: {:?}", e))
    })?;

    Ok(PyTensor { inner: result })
}

/// 2D transposed convolution (deconvolution)
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None, output_padding=None, groups=1, dilation=None))]
pub fn conv_transpose2d(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    stride: Option<(usize, usize)>,
    padding: Option<(usize, usize)>,
    output_padding: Option<(usize, usize)>,
    groups: usize,
    dilation: Option<(usize, usize)>,
) -> PyResult<PyTensor> {
    if dilation.unwrap_or((1, 1)) != (1, 1) {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "conv_transpose2d(dilation!=1) is not implemented",
        ));
    }
    if groups != 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "conv_transpose2d(groups!=1) is not implemented",
        ));
    }

    let result = nn::functional_api::conv_transpose_2d(
        &input.inner,
        &weight.inner,
        bias.map(|b| &b.inner),
        stride,
        padding,
        output_padding,
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "ConvTranspose2d failed: {:?}",
            e
        ))
    })?;

    Ok(PyTensor { inner: result })
}

/// 3D convolution
#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=None, padding=None))]
pub fn conv3d(
    input: &PyTensor,
    weight: &PyTensor,
    bias: Option<&PyTensor>,
    stride: Option<(usize, usize, usize)>,
    padding: Option<(usize, usize, usize)>,
) -> PyResult<PyTensor> {
    let result = nn::functional_api::conv3d(
        &input.inner,
        &weight.inner,
        bias.map(|b| &b.inner),
        stride.unwrap_or((1, 1, 1)),
        padding.unwrap_or((0, 0, 0)),
    )
    .map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Conv3d failed: {:?}", e))
    })?;

    Ok(PyTensor { inner: result })
}
