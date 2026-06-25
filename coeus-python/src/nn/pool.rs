use crate::tensor::{PyStateDict, PyTensor};
use pyo3::prelude::*;

/// Python-exposed 2D Average Pooling layer.
#[pyclass(name = "AvgPool2d")]
pub struct PyAvgPool2d {
    #[pyo3(get)]
    pub kernel_size: usize,
    #[pyo3(get)]
    pub stride: usize,
    #[pyo3(get)]
    pub padding: usize,
    #[pyo3(get)]
    pub dilation: usize,
}

#[pymethods]
impl PyAvgPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride = None, padding = 0, dilation = 1))]
    pub fn new(kernel_size: usize, stride: Option<usize>, padding: usize, dilation: usize) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
        }
    }

    /// Forward pass through the AvgPool2d layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let pool = coeus_nn::pool::AvgPool2d::<f64, coeus_core::MoiraiBackend>::with_params(
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        );

        let inner = py.allow_threads(move || pool.forward(&input_var));
        Ok(PyTensor::from_var(inner))
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

/// Python-exposed 2D Max Pooling layer.
#[pyclass(name = "MaxPool2d")]
pub struct PyMaxPool2d {
    #[pyo3(get)]
    pub kernel_size: usize,
    #[pyo3(get)]
    pub stride: usize,
    #[pyo3(get)]
    pub padding: usize,
    #[pyo3(get)]
    pub dilation: usize,
}

#[pymethods]
impl PyMaxPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride = None, padding = 0, dilation = 1))]
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
        Ok(PyTensor::from_var(inner))
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

/// Python-exposed 3D Average Pooling layer.
#[pyclass(name = "AvgPool3d")]
pub struct PyAvgPool3d {
    #[pyo3(get)]
    pub kernel_size: usize,
    #[pyo3(get)]
    pub stride: usize,
    #[pyo3(get)]
    pub padding: usize,
    #[pyo3(get)]
    pub dilation: usize,
}

#[pymethods]
impl PyAvgPool3d {
    #[new]
    #[pyo3(signature = (kernel_size, stride = None, padding = 0, dilation = 1))]
    pub fn new(kernel_size: usize, stride: Option<usize>, padding: usize, dilation: usize) -> Self {
        let stride = stride.unwrap_or(kernel_size);
        Self {
            kernel_size,
            stride,
            padding,
            dilation,
        }
    }

    /// Forward pass through the AvgPool3d layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let pool = coeus_nn::pool::AvgPool3d::<f64, coeus_core::MoiraiBackend>::with_params(
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        );

        let inner = py.allow_threads(move || pool.forward(&input_var));
        Ok(PyTensor::from_var(inner))
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
    #[pyo3(get)]
    pub kernel_size: usize,
    #[pyo3(get)]
    pub stride: usize,
    #[pyo3(get)]
    pub padding: usize,
    #[pyo3(get)]
    pub dilation: usize,
}

#[pymethods]
impl PyMaxPool3d {
    #[new]
    #[pyo3(signature = (kernel_size, stride = None, padding = 0, dilation = 1))]
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
        Ok(PyTensor::from_var(inner))
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

/// Python-exposed Global Average Pooling 1D (reduces `[N, C, L]` → `[N, C, 1]`).
#[pyclass(name = "GlobalAvgPool1d")]
#[derive(Default)]
pub struct PyGlobalAvgPool1d;

#[pymethods]
impl PyGlobalAvgPool1d {
    #[new]
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let result = py.allow_threads(|| {
            coeus_nn::GlobalAvgPool1d::<f64, coeus_core::MoiraiBackend>::new().forward(&input_var)
        });
        Ok(PyTensor { inner: result })
    }

    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    pub fn zero_grad(&self, _py: Python<'_>) {}
}

/// Python-exposed Global Average Pooling 2D (reduces `[N, C, H, W]` → `[N, C, 1, 1]`).
#[pyclass(name = "GlobalAvgPool2d")]
#[derive(Default)]
pub struct PyGlobalAvgPool2d;

#[pymethods]
impl PyGlobalAvgPool2d {
    #[new]
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let result = py.allow_threads(|| {
            coeus_nn::GlobalAvgPool2d::<f64, coeus_core::MoiraiBackend>::new().forward(&input_var)
        });
        Ok(PyTensor { inner: result })
    }

    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    pub fn zero_grad(&self, _py: Python<'_>) {}
}

/// Python-exposed Global Average Pooling 3D (reduces `[N, C, D, H, W]` → `[N, C, 1, 1, 1]`).
#[pyclass(name = "GlobalAvgPool3d")]
#[derive(Default)]
pub struct PyGlobalAvgPool3d;

#[pymethods]
impl PyGlobalAvgPool3d {
    #[new]
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let result = py.allow_threads(|| {
            coeus_nn::GlobalAvgPool3d::<f64, coeus_core::MoiraiBackend>::new().forward(&input_var)
        });
        Ok(PyTensor { inner: result })
    }

    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    pub fn zero_grad(&self, _py: Python<'_>) {}
}

/// Python-exposed Global Max Pooling 2D (reduces `[N, C, H, W]` → `[N, C, 1, 1]`).
#[pyclass(name = "GlobalMaxPool2d")]
#[derive(Default)]
pub struct PyGlobalMaxPool2d;

#[pymethods]
impl PyGlobalMaxPool2d {
    #[new]
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let result = py.allow_threads(|| {
            coeus_nn::GlobalMaxPool2d::<f64, coeus_core::MoiraiBackend>::new().forward(&input_var)
        });
        Ok(PyTensor { inner: result })
    }

    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    pub fn zero_grad(&self, _py: Python<'_>) {}
}

/// Python-exposed Global Max Pooling 3D (reduces `[N, C, D, H, W]` → `[N, C, 1, 1, 1]`).
#[pyclass(name = "GlobalMaxPool3d")]
#[derive(Default)]
pub struct PyGlobalMaxPool3d;

#[pymethods]
impl PyGlobalMaxPool3d {
    #[new]
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let result = py.allow_threads(|| {
            coeus_nn::GlobalMaxPool3d::<f64, coeus_core::MoiraiBackend>::new().forward(&input_var)
        });
        Ok(PyTensor { inner: result })
    }

    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    pub fn zero_grad(&self, _py: Python<'_>) {}
}
