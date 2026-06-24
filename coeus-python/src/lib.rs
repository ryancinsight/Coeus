// ── Coeus Python bindings entry point ──

use pyo3::prelude::*;

#[cfg(feature = "mnemosyne-global")]
#[global_allocator]
static GLOBAL: mnemosyne::Mnemosyne = mnemosyne::Mnemosyne;

pub mod activations;
pub mod dist;
pub mod losses;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod tensor;

use dist::{PyMockCommunicator, PyTcpCommunicator, PyTcpMesh};
use tensor::PyTensor;

/// Shutdown the global Moirai executor.
#[pyfunction]
pub fn shutdown(py: Python<'_>) {
    py.allow_threads(|| {
        moirai::global().shutdown();
    });
}

/// PyCoeus extension module definition.
#[pymodule]
pub fn pycoeus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shutdown, m)?)?;
    m.add_class::<PyTensor>()?;
    m.add_class::<nn::PyLinear>()?;
    m.add_class::<tensor::PyStateDict>()?;
    m.add_class::<optim::PyLrScheduler>()?;
    m.add_class::<nn::PyConv1d>()?;
    m.add_class::<nn::PyConv2d>()?;
    m.add_class::<nn::PyConv3d>()?;
    m.add_class::<nn::PyLayerNorm>()?;
    m.add_class::<nn::PyRMSNorm>()?;
    m.add_class::<nn::PyBatchNorm3d>()?;
    m.add_class::<nn::PyAvgPool2d>()?;
    m.add_class::<nn::PyMaxPool2d>()?;
    m.add_class::<nn::PyAvgPool3d>()?;
    m.add_class::<nn::PyMaxPool3d>()?;
    m.add_class::<nn::PyEmbedding>()?;
    m.add_class::<nn::PyDropout>()?;
    m.add_class::<nn::PyBatchNorm1d>()?;
    m.add_class::<nn::PyBatchNorm2d>()?;
    m.add_class::<optim::PySGD>()?;
    m.add_class::<optim::PyAdam>()?;
    m.add_class::<optim::PyAdamW>()?;
    m.add_class::<optim::PyRMSProp>()?;
    m.add_class::<optim::PyAdaGrad>()?;
    m.add_class::<nn::PyGroupNorm>()?;
    m.add_class::<nn::PyInstanceNorm1d>()?;
    m.add_class::<nn::PyInstanceNorm2d>()?;
    m.add_class::<nn::PyMultiHeadAttention>()?;
    m.add_class::<nn::PyRotaryEmbedding>()?;
    m.add_class::<PyMockCommunicator>()?;
    m.add_class::<PyTcpMesh>()?;
    m.add_class::<PyTcpCommunicator>()?;

    m.add_function(wrap_pyfunction!(dist::create_mock_cluster, m)?)?;
    m.add_function(wrap_pyfunction!(dist::synchronize_gradients, m)?)?;

    m.add_function(wrap_pyfunction!(activations::relu, m)?)?;
    m.add_function(wrap_pyfunction!(activations::sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(activations::tanh, m)?)?;
    m.add_function(wrap_pyfunction!(activations::gelu, m)?)?;
    m.add_function(wrap_pyfunction!(activations::silu, m)?)?;
    m.add_function(wrap_pyfunction!(activations::mish, m)?)?;
    m.add_function(wrap_pyfunction!(activations::elu, m)?)?;
    m.add_function(wrap_pyfunction!(activations::softplus, m)?)?;
    m.add_function(wrap_pyfunction!(activations::gelu_tanh, m)?)?;
    m.add_function(wrap_pyfunction!(activations::leaky_relu, m)?)?;

    m.add_function(wrap_pyfunction!(losses::mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(losses::cross_entropy_loss, m)?)?;
    m.add_function(wrap_pyfunction!(losses::binary_cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(losses::nll_loss, m)?)?;
    m.add_function(wrap_pyfunction!(losses::huber_loss, m)?)?;
    m.add_function(wrap_pyfunction!(losses::cosine_embedding_loss, m)?)?;

    m.add_function(wrap_pyfunction!(ops::exp, m)?)?;
    m.add_function(wrap_pyfunction!(ops::log, m)?)?;
    m.add_function(wrap_pyfunction!(ops::log_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(ops::cat, m)?)?;
    m.add_function(wrap_pyfunction!(ops::stack, m)?)?;
    m.add_function(wrap_pyfunction!(ops::split, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sum_axis, m)?)?;
    m.add_function(wrap_pyfunction!(ops::mean_axis, m)?)?;
    m.add_function(wrap_pyfunction!(ops::pad, m)?)?;
    m.add_function(wrap_pyfunction!(ops::cumsum, m)?)?;
    m.add_function(wrap_pyfunction!(ops::matmul, m)?)?;
    m.add_function(wrap_pyfunction!(ops::abs, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sqrt, m)?)?;
    m.add_function(wrap_pyfunction!(ops::neg, m)?)?;
    m.add_function(wrap_pyfunction!(ops::clamp, m)?)?;
    m.add_function(wrap_pyfunction!(ops::max_axis, m)?)?;
    m.add_function(wrap_pyfunction!(ops::min_axis, m)?)?;
    m.add_function(wrap_pyfunction!(ops::log_sum_exp, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sum, m)?)?;
    m.add_function(wrap_pyfunction!(ops::mean, m)?)?;
    m.add_function(wrap_pyfunction!(ops::zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ones, m)?)?;
    m.add_function(wrap_pyfunction!(ops::full, m)?)?;
    m.add_function(wrap_pyfunction!(ops::arange, m)?)?;
    m.add_function(wrap_pyfunction!(ops::linspace, m)?)?;
    m.add_function(wrap_pyfunction!(ops::reshape, m)?)?;
    m.add_function(wrap_pyfunction!(ops::permute, m)?)?;
    m.add_function(wrap_pyfunction!(ops::t, m)?)?;
    m.add_function(wrap_pyfunction!(ops::pow, m)?)?;
    // Trigonometric
    m.add_function(wrap_pyfunction!(ops::sin, m)?)?;
    m.add_function(wrap_pyfunction!(ops::cos, m)?)?;
    // Shape ops
    m.add_function(wrap_pyfunction!(ops::flip, m)?)?;
    m.add_function(wrap_pyfunction!(ops::where_cond, m)?)?;
    m.add_function(wrap_pyfunction!(ops::softmax, m)?)?;
    // Constructors
    m.add_function(wrap_pyfunction!(ops::randn, m)?)?;
    m.add_function(wrap_pyfunction!(ops::zeros_like, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ones_like, m)?)?;
    m.add_function(wrap_pyfunction!(ops::eye, m)?)?;
    // Sorting / selection
    m.add_function(wrap_pyfunction!(ops::topk, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sort, m)?)?;
    // Statistical ops
    m.add_function(wrap_pyfunction!(ops::std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(ops::tensor_var, m)?)?;
    m.add_function(wrap_pyfunction!(ops::norm, m)?)?;
    // Comparison / selection
    m.add_function(wrap_pyfunction!(ops::eq, m)?)?;
    m.add_function(wrap_pyfunction!(ops::lt, m)?)?;
    m.add_function(wrap_pyfunction!(ops::gt, m)?)?;
    m.add_function(wrap_pyfunction!(ops::where_fn, m)?)?;

    Ok(())
}
