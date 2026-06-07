use pyo3::prelude::*;
use crate::tensor::{PyTensor, PyStateDict};

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
        Self { kernel_size, stride, padding, dilation }
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
        Ok(PyTensor { inner })
    }

    fn state_dict(&self) -> PyStateDict {
        PyStateDict { inner: coeus_tensor::checkpoint::StateDict::new() }
    }

    fn load_state_dict(&self, _state_dict: &PyStateDict) -> PyResult<()> {
        Ok(())
    }
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
        Self { kernel_size, stride, padding, dilation }
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
        Ok(PyTensor { inner })
    }

    fn state_dict(&self) -> PyStateDict {
        PyStateDict { inner: coeus_tensor::checkpoint::StateDict::new() }
    }

    fn load_state_dict(&self, _state_dict: &PyStateDict) -> PyResult<()> {
        Ok(())
    }
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
        Self { kernel_size, stride, padding, dilation }
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
        Ok(PyTensor { inner })
    }

    fn state_dict(&self) -> PyStateDict {
        PyStateDict { inner: coeus_tensor::checkpoint::StateDict::new() }
    }

    fn load_state_dict(&self, _state_dict: &PyStateDict) -> PyResult<()> {
        Ok(())
    }
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
        Self { kernel_size, stride, padding, dilation }
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
        Ok(PyTensor { inner })
    }

    fn state_dict(&self) -> PyStateDict {
        PyStateDict { inner: coeus_tensor::checkpoint::StateDict::new() }
    }

    fn load_state_dict(&self, _state_dict: &PyStateDict) -> PyResult<()> {
        Ok(())
    }
}
