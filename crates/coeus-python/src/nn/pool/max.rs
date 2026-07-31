use crate::{
    nn::error::map_module_error,
    tensor::{PyStateDict, PyTensor},
};
use pyo3::prelude::*;

/// Python-exposed 1D Max Pooling layer.
#[pyclass(name = "MaxPool1d")]
pub struct PyMaxPool1d {
    /// Pooling window length.
    #[pyo3(get)]
    pub kernel_size: usize,
    /// Pooling stride.
    #[pyo3(get)]
    pub stride: usize,
    /// Zero-padding length.
    #[pyo3(get)]
    pub padding: usize,
    /// Dilation factor.
    #[pyo3(get)]
    pub dilation: usize,
}

#[pymethods]
impl PyMaxPool1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride = None, padding = 0, dilation = 1))]
    /// Create a MaxPool1d layer.
    pub fn new(kernel_size: usize, stride: Option<usize>, padding: usize, dilation: usize) -> Self {
        Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            padding,
            dilation,
        }
    }

    /// Forward pass: `[N, C, L]` → `[N, C, L_out]`.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let pool = coeus_nn::pool::MaxPool1d::<f64, coeus_core::MoiraiBackend>::with_params(
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        );
        let inner = py.allow_threads(move || pool.forward(&input_var));
        inner.map(PyTensor::from_var).map_err(map_module_error)
    }

    /// Return an empty state dict (no learnable parameters).
    pub fn state_dict(&self) -> PyStateDict {
        PyStateDict {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    /// Zero gradients (no-op for pooling layers).
    pub fn zero_grad(&self) {}
}

/// Python-exposed 2D Max Pooling layer.
#[pyclass(name = "MaxPool2d")]
pub struct PyMaxPool2d {
    /// Square pooling kernel side length.
    #[pyo3(get)]
    pub kernel_size: usize,
    /// Pooling stride (defaults to `kernel_size` when not specified).
    #[pyo3(get)]
    pub stride: usize,
    /// Zero-padding applied to spatial dimensions.
    #[pyo3(get)]
    pub padding: usize,
    /// Dilation factor for the pooling kernel.
    #[pyo3(get)]
    pub dilation: usize,
}

#[pymethods]
impl PyMaxPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride = None, padding = 0, dilation = 1))]
    /// Create a MaxPool2d layer.
    pub fn new(kernel_size: usize, stride: Option<usize>, padding: usize, dilation: usize) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
        }
    }

    /// Forward pass through the MaxPool2d layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let pool = coeus_nn::pool::MaxPool2d::<f64, coeus_core::MoiraiBackend>::with_params(
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        );

        let inner = py.allow_threads(move || pool.forward(&input_var));
        inner.map(PyTensor::from_var).map_err(map_module_error)
    }

    fn state_dict(&self) -> PyStateDict {
        PyStateDict {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    fn load_state_dict(&self, _state_dict: &PyStateDict) -> PyResult<()> {
        Ok(())
    }

    /// Return the list of learnable parameters.
    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, _py: Python<'_>) {}
}

/// Python-exposed 3D Max Pooling layer.
#[pyclass(name = "MaxPool3d")]
pub struct PyMaxPool3d {
    /// Cubic pooling kernel side length.
    #[pyo3(get)]
    pub kernel_size: usize,
    /// Pooling stride (defaults to `kernel_size` when not specified).
    #[pyo3(get)]
    pub stride: usize,
    /// Zero-padding applied to spatial dimensions.
    #[pyo3(get)]
    pub padding: usize,
    /// Dilation factor for the pooling kernel.
    #[pyo3(get)]
    pub dilation: usize,
}

#[pymethods]
impl PyMaxPool3d {
    #[new]
    #[pyo3(signature = (kernel_size, stride = None, padding = 0, dilation = 1))]
    /// Create a MaxPool3d layer.
    pub fn new(kernel_size: usize, stride: Option<usize>, padding: usize, dilation: usize) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
        }
    }

    /// Forward pass through the MaxPool3d layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let pool = coeus_nn::pool::MaxPool3d::<f64, coeus_core::MoiraiBackend>::with_params(
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        );

        let inner = py.allow_threads(move || pool.forward(&input_var));
        inner.map(PyTensor::from_var).map_err(map_module_error)
    }

    fn state_dict(&self) -> PyStateDict {
        PyStateDict {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    fn load_state_dict(&self, _state_dict: &PyStateDict) -> PyResult<()> {
        Ok(())
    }

    /// Return the list of learnable parameters.
    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, _py: Python<'_>) {}
}
