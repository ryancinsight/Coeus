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

use dist::{PyLocalCommunicator, PyTcpCommunicator, PyTcpMesh};
use tensor::{PyTensor, PyTensorIterator};

/// Shutdown the global Moirai executor.
#[pyfunction]
pub fn shutdown(py: Python<'_>) {
    py.allow_threads(|| {
        moirai::global().shutdown();
    });
}

/// Context manager that disables gradient tracking within its scope.
///
/// Usage:
/// ```python
/// with pycoeus.no_grad():
///     y = model(x)   # no gradients computed
/// ```
///
/// Implemented as a thin Python context manager; in the current version it is
/// a no-op marker since Coeus' backward pass is lazy (no grads accumulate
/// until `.backward()` is called).  Future versions will honour this flag for
/// in-place mutation and memory saving.
#[pyclass(name = "no_grad")]
pub struct NoGradCtx;

#[pymethods]
impl NoGradCtx {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __enter__(&self) {}

    fn __exit__(
        &self,
        _exc_type: pyo3::Bound<'_, pyo3::types::PyAny>,
        _exc_val: pyo3::Bound<'_, pyo3::types::PyAny>,
        _exc_tb: pyo3::Bound<'_, pyo3::types::PyAny>,
    ) -> bool {
        false // do not suppress exceptions
    }
}

/// PyCoeus extension module definition.
#[pymodule]
pub fn pycoeus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(shutdown, m)?)?;
    m.add_class::<PyTensor>()?;
    m.add_class::<PyTensorIterator>()?;
    // no_grad context manager
    m.add_class::<NoGradCtx>()?;
    m.add_class::<nn::PyLinear>()?;
    m.add_class::<tensor::PyStateDict>()?;
    m.add_class::<optim::PyLrScheduler>()?;
    m.add_class::<nn::PyConv1d>()?;
    m.add_class::<nn::PyConv2d>()?;
    m.add_class::<nn::PyConv3d>()?;
    m.add_class::<nn::PyConvTranspose1d>()?;
    m.add_class::<nn::PyConvTranspose2d>()?;
    m.add_class::<nn::PyLayerNorm>()?;
    m.add_class::<nn::PyRMSNorm>()?;
    m.add_class::<nn::PyBatchNorm3d>()?;
    m.add_class::<nn::PyAvgPool2d>()?;
    m.add_class::<nn::PyMaxPool2d>()?;
    m.add_class::<nn::PyAvgPool3d>()?;
    m.add_class::<nn::PyMaxPool3d>()?;
    m.add_class::<nn::PyGlobalAvgPool1d>()?;
    m.add_class::<nn::PyGlobalAvgPool2d>()?;
    m.add_class::<nn::PyGlobalAvgPool3d>()?;
    m.add_class::<nn::PyGlobalMaxPool2d>()?;
    m.add_class::<nn::PyGlobalMaxPool3d>()?;
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
    m.add_class::<nn::PyFeedForward>()?;
    m.add_class::<PyLocalCommunicator>()?;
    m.add_class::<PyTcpMesh>()?;
    m.add_class::<PyTcpCommunicator>()?;

    m.add_function(wrap_pyfunction!(dist::create_local_cluster, m)?)?;
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
    m.add_function(wrap_pyfunction!(ops::recip, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sign, m)?)?;
    m.add_function(wrap_pyfunction!(ops::floor, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ceil, m)?)?;
    m.add_function(wrap_pyfunction!(ops::round, m)?)?;
    m.add_function(wrap_pyfunction!(ops::trunc, m)?)?;
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
    m.add_function(wrap_pyfunction!(ops::vector_norm, m)?)?;
    // Comparison / selection
    m.add_function(wrap_pyfunction!(ops::eq, m)?)?;
    m.add_function(wrap_pyfunction!(ops::lt, m)?)?;
    m.add_function(wrap_pyfunction!(ops::gt, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ge, m)?)?;
    m.add_function(wrap_pyfunction!(ops::le, m)?)?;
    m.add_function(wrap_pyfunction!(ops::ne, m)?)?;
    m.add_function(wrap_pyfunction!(ops::where_fn, m)?)?;
    // Indexing ops
    m.add_function(wrap_pyfunction!(ops::gather, m)?)?;
    m.add_function(wrap_pyfunction!(ops::index_select, m)?)?;
    m.add_function(wrap_pyfunction!(ops::einsum, m)?)?;
    m.add_function(wrap_pyfunction!(ops::scatter_add, m)?)?;
    m.add_function(wrap_pyfunction!(ops::repeat_interleave, m)?)?;
    // Spatial resize
    m.add_function(wrap_pyfunction!(ops::interpolate, m)?)?;
    // Shape extras
    m.add_function(wrap_pyfunction!(ops::unsqueeze, m)?)?;
    m.add_function(wrap_pyfunction!(ops::squeeze, m)?)?;
    m.add_function(wrap_pyfunction!(ops::flatten, m)?)?;
    // Selection
    m.add_function(wrap_pyfunction!(ops::argmax, m)?)?;
    m.add_function(wrap_pyfunction!(ops::argmin, m)?)?;
    // broadcast / masked_fill / nonzero
    m.add_function(wrap_pyfunction!(ops::broadcast_to, m)?)?;
    m.add_function(wrap_pyfunction!(ops::masked_fill, m)?)?;
    m.add_function(wrap_pyfunction!(ops::nonzero, m)?)?;
    // Triangular masking / roll
    m.add_function(wrap_pyfunction!(ops::tril, m)?)?;
    m.add_function(wrap_pyfunction!(ops::triu, m)?)?;
    m.add_function(wrap_pyfunction!(ops::roll, m)?)?;
    // meshgrid / tile
    m.add_function(wrap_pyfunction!(ops::meshgrid, m)?)?;
    m.add_function(wrap_pyfunction!(ops::tile, m)?)?;
    // Functional nn (stateless)
    m.add_function(wrap_pyfunction!(ops::linear, m)?)?;
    m.add_function(wrap_pyfunction!(ops::layer_norm, m)?)?;
    m.add_function(wrap_pyfunction!(ops::dropout, m)?)?;
    // diag / diagonal / cumprod
    m.add_function(wrap_pyfunction!(ops::diag, m)?)?;
    m.add_function(wrap_pyfunction!(ops::diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(ops::cumprod, m)?)?;
    // nn.functional aliases (F.*)
    m.add_function(wrap_pyfunction!(ops::f_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(ops::f_log_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(ops::f_relu, m)?)?;
    m.add_function(wrap_pyfunction!(ops::f_sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(ops::f_tanh, m)?)?;
    m.add_function(wrap_pyfunction!(ops::f_gelu, m)?)?;
    m.add_function(wrap_pyfunction!(ops::f_silu, m)?)?;
    m.add_function(wrap_pyfunction!(ops::f_mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(ops::f_binary_cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(ops::f_cross_entropy, m)?)?;
    // amax / amin / prod
    m.add_function(wrap_pyfunction!(ops::amax, m)?)?;
    m.add_function(wrap_pyfunction!(ops::amin, m)?)?;
    m.add_function(wrap_pyfunction!(ops::prod, m)?)?;

    Ok(())
}
