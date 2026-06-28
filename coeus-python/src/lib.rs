// ── Coeus Python bindings entry point ──
#![deny(missing_docs)]
//! PyO3 Python bindings for the Coeus tensor library.
use pyo3::prelude::*;

#[cfg(feature = "mnemosyne-global")]
#[global_allocator]
static GLOBAL: mnemosyne::Mnemosyne = mnemosyne::Mnemosyne;

/// Activation functions (ReLU, GELU, SiLU, etc.).
pub mod activations;
/// Distributed training communicators.
pub mod dist;
mod grad_mode;
/// Weight initialization functions (uniform, normal, xavier, kaiming).
pub mod init;
/// Loss functions (MSE, cross-entropy, etc.).
pub mod losses;
/// Neural network layers (Linear, Conv, Norm, Attention, Transformer, RNN, Pool, etc.).
pub mod nn;
/// Tensor operations (arithmetic, reductions, shaping, functional nn).
pub mod ops;
/// Optimizers and learning rate schedulers.
pub mod optim;
/// Tensor type and supporting utilities (state dict, iterator).
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
/// The context manager tracks nested scopes on the current Python thread. Values
/// returned through PyO3 operation wrappers inside the scope are detached from
/// the autograd graph; explicit tensor factories still honor `requires_grad`.
#[pyclass(name = "no_grad")]
pub struct NoGradCtx {
    active: std::sync::atomic::AtomicBool,
}

#[pymethods]
impl NoGradCtx {
    #[new]
    fn new() -> Self {
        Self {
            active: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn __enter__(&self) {
        if !self.active.swap(true, std::sync::atomic::Ordering::AcqRel) {
            grad_mode::push_no_grad();
        }
    }

    fn __exit__(
        &self,
        _exc_type: pyo3::Bound<'_, pyo3::types::PyAny>,
        _exc_val: pyo3::Bound<'_, pyo3::types::PyAny>,
        _exc_tb: pyo3::Bound<'_, pyo3::types::PyAny>,
    ) -> bool {
        if self.active.swap(false, std::sync::atomic::Ordering::AcqRel) {
            grad_mode::pop_no_grad();
        }
        false // do not suppress exceptions
    }
}

impl Drop for NoGradCtx {
    fn drop(&mut self) {
        if self.active.swap(false, std::sync::atomic::Ordering::AcqRel) {
            grad_mode::pop_no_grad();
        }
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
    m.add_class::<nn::PyConvTranspose3d>()?;
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
    m.add_class::<nn::PyBilinear>()?;
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
    m.add_class::<nn::PyInstanceNorm3d>()?;
    m.add_class::<nn::PyMultiHeadAttention>()?;
    m.add_class::<nn::PyRotaryEmbedding>()?;
    m.add_class::<nn::PyFeedForward>()?;
    m.add_class::<nn::PyTransformerDecoderLayer>()?;
    m.add_class::<nn::PyTransformerDecoder>()?;
    m.add_class::<nn::PyTransformerEncoderLayer>()?;
    m.add_class::<nn::PyTransformerEncoder>()?;
    m.add_class::<nn::PyTransformer>()?;
    m.add_class::<nn::PySinusoidalEncoding>()?;
    m.add_class::<nn::PyScaledDotProductAttention>()?;
    m.add_class::<nn::PySequential>()?;
    m.add_class::<nn::PyModuleList>()?;
    m.add_class::<nn::PyModule>()?;
    m.add_class::<nn::PyLSTMCell>()?;
    m.add_class::<nn::PyGRUCell>()?;
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
    m.add_function(wrap_pyfunction!(activations::glu, m)?)?;
    m.add_function(wrap_pyfunction!(activations::masked_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(activations::causal_softmax, m)?)?;
    // G-037 extended activation family
    m.add_function(wrap_pyfunction!(activations::hardtanh, m)?)?;
    m.add_function(wrap_pyfunction!(activations::hardsigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(activations::hardswish, m)?)?;
    m.add_function(wrap_pyfunction!(activations::hardshrink, m)?)?;
    m.add_function(wrap_pyfunction!(activations::softshrink, m)?)?;
    m.add_function(wrap_pyfunction!(activations::softsign, m)?)?;
    m.add_function(wrap_pyfunction!(activations::threshold, m)?)?;
    m.add_function(wrap_pyfunction!(activations::celu, m)?)?;
    m.add_function(wrap_pyfunction!(activations::prelu, m)?)?;

    m.add_function(wrap_pyfunction!(losses::mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(losses::cross_entropy_loss, m)?)?;
    m.add_function(wrap_pyfunction!(losses::binary_cross_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(losses::nll_loss, m)?)?;
    m.add_function(wrap_pyfunction!(losses::huber_loss, m)?)?;
    m.add_function(wrap_pyfunction!(losses::kl_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(losses::margin_ranking_loss, m)?)?;
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
    m.add_function(wrap_pyfunction!(ops::logspace, m)?)?;
    m.add_function(wrap_pyfunction!(ops::geomspace, m)?)?;
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
    m.add_function(wrap_pyfunction!(ops::rand, m)?)?;
    m.add_function(wrap_pyfunction!(ops::randint, m)?)?;
    m.add_function(wrap_pyfunction!(ops::bernoulli, m)?)?;
    // Sorting / selection
    m.add_function(wrap_pyfunction!(ops::topk, m)?)?;
    m.add_function(wrap_pyfunction!(ops::sort, m)?)?;
    // Statistical ops
    m.add_function(wrap_pyfunction!(ops::std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(ops::tensor_var, m)?)?;
    m.add_function(wrap_pyfunction!(ops::var_mean, m)?)?;
    m.add_function(wrap_pyfunction!(ops::std_mean, m)?)?;
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
    m.add_function(wrap_pyfunction!(ops::index_put, m)?)?;
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
    m.add_function(wrap_pyfunction!(ops::broadcast_tensors, m)?)?;
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
    m.add_function(wrap_pyfunction!(ops::bilinear, m)?)?;
    m.add_function(wrap_pyfunction!(ops::layer_norm, m)?)?;
    m.add_function(wrap_pyfunction!(ops::batch_norm_1d, m)?)?;
    m.add_function(wrap_pyfunction!(ops::rms_norm, m)?)?;
    m.add_function(wrap_pyfunction!(ops::group_norm, m)?)?;
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
    // vector arithmetic (dot / cross)
    m.add_function(wrap_pyfunction!(ops::dot, m)?)?;
    m.add_function(wrap_pyfunction!(ops::cross, m)?)?;
    // matrix norm (Frobenius; ord!='fro' is a ValueError)
    m.add_function(wrap_pyfunction!(ops::matrix_norm, m)?)?;
    // Functional attention
    m.add_function(wrap_pyfunction!(ops::scaled_dot_product_attention, m)?)?;
    // Batch matmul / outer product
    m.add_function(wrap_pyfunction!(ops::bmm, m)?)?;
    m.add_function(wrap_pyfunction!(ops::outer, m)?)?;
    // Encoding / selection
    m.add_function(wrap_pyfunction!(ops::one_hot, m)?)?;
    m.add_function(wrap_pyfunction!(ops::masked_select, m)?)?;
    // Chunking
    m.add_function(wrap_pyfunction!(ops::chunk, m)?)?;
    // Normalization
    m.add_function(wrap_pyfunction!(ops::normalize, m)?)?;
    // Comparison / closeness
    m.add_function(wrap_pyfunction!(ops::isclose, m)?)?;
    m.add_function(wrap_pyfunction!(ops::allclose, m)?)?;
    m.add_function(wrap_pyfunction!(ops::nan_to_num, m)?)?;
    // Gradient utilities
    m.add_function(wrap_pyfunction!(ops::clip_grad_norm_, m)?)?;
    m.add_function(wrap_pyfunction!(ops::clip_grad_value_, m)?)?;

    // ── init sub-module (weight initialization functions) ──
    let init_mod = PyModule::new(m.py(), "init")?;
    init_mod.add_function(wrap_pyfunction!(init::uniform_, &init_mod)?)?;
    init_mod.add_function(wrap_pyfunction!(init::normal_, &init_mod)?)?;
    init_mod.add_function(wrap_pyfunction!(init::constant_, &init_mod)?)?;
    init_mod.add_function(wrap_pyfunction!(init::zeros_, &init_mod)?)?;
    init_mod.add_function(wrap_pyfunction!(init::ones_, &init_mod)?)?;
    init_mod.add_function(wrap_pyfunction!(init::xavier_uniform_, &init_mod)?)?;
    init_mod.add_function(wrap_pyfunction!(init::xavier_normal_, &init_mod)?)?;
    init_mod.add_function(wrap_pyfunction!(init::kaiming_uniform_, &init_mod)?)?;
    init_mod.add_function(wrap_pyfunction!(init::kaiming_normal_, &init_mod)?)?;
    m.add_submodule(&init_mod)?;

    Ok(())
}
