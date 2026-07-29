use crate::{nn::error::map_module_error, tensor::PyTensor};
use pyo3::prelude::*;

/// Python-exposed Adaptive 1D Average Pooling layer.
///
/// Equivalent to `torch.nn.AdaptiveAvgPool1d(output_size)`.
#[pyclass(name = "AdaptiveAvgPool1d")]
pub struct PyAdaptiveAvgPool1d {
    /// Target spatial output length.
    #[pyo3(get)]
    pub output_size: usize,
}

#[pymethods]
impl PyAdaptiveAvgPool1d {
    #[new]
    /// Create an `AdaptiveAvgPool1d` with the given target output length.
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }

    /// Forward pass: `[N, C, L]` → `[N, C, output_size]`.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let output_size = self.output_size;
        let pool = coeus_nn::AdaptiveAvgPool1d::<f64, coeus_core::MoiraiBackend>::new(output_size);
        let inner = py.allow_threads(move || pool.forward(&input_var));
        inner.map(PyTensor::from_var).map_err(map_module_error)
    }

    /// Return an empty state dict (no learnable parameters).
    pub fn state_dict(&self) -> crate::tensor::PyStateDict {
        crate::tensor::PyStateDict {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    /// Zero gradients (no-op for pooling layers).
    pub fn zero_grad(&self) {}
}

/// Python-exposed Adaptive 2D Average Pooling layer.
///
/// Equivalent to `torch.nn.AdaptiveAvgPool2d((out_h, out_w))`.
#[pyclass(name = "AdaptiveAvgPool2d")]
pub struct PyAdaptiveAvgPool2d {
    /// Target output height.
    #[pyo3(get)]
    pub out_h: usize,
    /// Target output width.
    #[pyo3(get)]
    pub out_w: usize,
}

#[pymethods]
impl PyAdaptiveAvgPool2d {
    #[new]
    #[pyo3(signature = (out_h, out_w = None))]
    /// Create an `AdaptiveAvgPool2d` pooling to `(out_h, out_w)`.
    ///
    /// Passing only one size `n` gives a square `(n, n)` output.
    pub fn new(out_h: usize, out_w: Option<usize>) -> Self {
        Self {
            out_h,
            out_w: out_w.unwrap_or(out_h),
        }
    }

    /// Forward pass: `[N, C, H, W]` → `[N, C, out_h, out_w]`.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let out_h = self.out_h;
        let out_w = self.out_w;
        let pool = coeus_nn::AdaptiveAvgPool2d::<f64, coeus_core::MoiraiBackend>::new(out_h, out_w);
        let inner = py.allow_threads(move || pool.forward(&input_var));
        inner.map(PyTensor::from_var).map_err(map_module_error)
    }

    /// Return an empty state dict (no learnable parameters).
    pub fn state_dict(&self) -> crate::tensor::PyStateDict {
        crate::tensor::PyStateDict {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    /// Zero gradients (no-op for pooling layers).
    pub fn zero_grad(&self) {}
}

/// Python-exposed Adaptive 1D Max Pooling layer.
///
/// Equivalent to `torch.nn.AdaptiveMaxPool1d(output_size)`. Differentiable.
#[pyclass(name = "AdaptiveMaxPool1d")]
pub struct PyAdaptiveMaxPool1d {
    /// Target spatial output length.
    #[pyo3(get)]
    pub output_size: usize,
}

#[pymethods]
impl PyAdaptiveMaxPool1d {
    #[new]
    /// Create an `AdaptiveMaxPool1d` with the given target output length.
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }

    /// Forward pass: `[N, C, L]` → `[N, C, output_size]`.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let output_size = self.output_size;
        let pool = coeus_nn::AdaptiveMaxPool1d::<f64, coeus_core::MoiraiBackend>::new(output_size);
        let inner = py.allow_threads(move || pool.forward(&input_var));
        inner.map(PyTensor::from_var).map_err(map_module_error)
    }

    /// Return an empty state dict (no learnable parameters).
    pub fn state_dict(&self) -> crate::tensor::PyStateDict {
        crate::tensor::PyStateDict {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    /// Zero gradients (no-op for pooling layers).
    pub fn zero_grad(&self) {}
}

/// Python-exposed Adaptive 2D Max Pooling layer.
///
/// Equivalent to `torch.nn.AdaptiveMaxPool2d((out_h, out_w))`. Differentiable.
#[pyclass(name = "AdaptiveMaxPool2d")]
pub struct PyAdaptiveMaxPool2d {
    /// Target output height.
    #[pyo3(get)]
    pub out_h: usize,
    /// Target output width.
    #[pyo3(get)]
    pub out_w: usize,
}

#[pymethods]
impl PyAdaptiveMaxPool2d {
    #[new]
    #[pyo3(signature = (out_h, out_w = None))]
    /// Create an `AdaptiveMaxPool2d` pooling to `(out_h, out_w)`.
    ///
    /// Passing only one size `n` gives a square `(n, n)` output.
    pub fn new(out_h: usize, out_w: Option<usize>) -> Self {
        Self {
            out_h,
            out_w: out_w.unwrap_or(out_h),
        }
    }

    /// Forward pass: `[N, C, H, W]` → `[N, C, out_h, out_w]`.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let out_h = self.out_h;
        let out_w = self.out_w;
        let pool = coeus_nn::AdaptiveMaxPool2d::<f64, coeus_core::MoiraiBackend>::new(out_h, out_w);
        let inner = py.allow_threads(move || pool.forward(&input_var));
        inner.map(PyTensor::from_var).map_err(map_module_error)
    }

    /// Return an empty state dict (no learnable parameters).
    pub fn state_dict(&self) -> crate::tensor::PyStateDict {
        crate::tensor::PyStateDict {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    /// Zero gradients (no-op for pooling layers).
    pub fn zero_grad(&self) {}
}
