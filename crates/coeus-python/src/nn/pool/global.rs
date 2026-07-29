use crate::{nn::error::map_module_error, tensor::PyTensor};
use pyo3::prelude::*;

/// Python-exposed Global Average Pooling 1D (reduces `[N, C, L]` → `[N, C, 1]`).
#[pyclass(name = "GlobalAvgPool1d")]
#[derive(Default)]
pub struct PyGlobalAvgPool1d;

#[pymethods]
impl PyGlobalAvgPool1d {
    #[new]
    /// Create a GlobalAvgPool1d layer.
    pub fn new() -> Self {
        Self
    }

    /// Forward pass: reduce `[N, C, L]` → `[N, C, 1]` by global average pooling.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let result = py.allow_threads(|| {
            coeus_nn::GlobalAvgPool1d::<f64, coeus_core::MoiraiBackend>::new().forward(&input_var)
        });
        result
            .map(|inner| PyTensor { inner })
            .map_err(map_module_error)
    }

    /// Return the list of learnable parameters (always empty — no parameters).
    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    /// Zero gradients of all parameters (no-op).
    pub fn zero_grad(&self, _py: Python<'_>) {}
}

/// Python-exposed Global Average Pooling 2D (reduces `[N, C, H, W]` → `[N, C, 1, 1]`).
#[pyclass(name = "GlobalAvgPool2d")]
#[derive(Default)]
pub struct PyGlobalAvgPool2d;

#[pymethods]
impl PyGlobalAvgPool2d {
    #[new]
    /// Create a GlobalAvgPool2d layer.
    pub fn new() -> Self {
        Self
    }

    /// Forward pass: reduce `[N, C, H, W]` → `[N, C, 1, 1]` by global average pooling.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let result = py.allow_threads(|| {
            coeus_nn::GlobalAvgPool2d::<f64, coeus_core::MoiraiBackend>::new().forward(&input_var)
        });
        result
            .map(|inner| PyTensor { inner })
            .map_err(map_module_error)
    }

    /// Return the list of learnable parameters (always empty — no parameters).
    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    /// Zero gradients of all parameters (no-op).
    pub fn zero_grad(&self, _py: Python<'_>) {}
}

/// Python-exposed Global Average Pooling 3D (reduces `[N, C, D, H, W]` → `[N, C, 1, 1, 1]`).
#[pyclass(name = "GlobalAvgPool3d")]
#[derive(Default)]
pub struct PyGlobalAvgPool3d;

#[pymethods]
impl PyGlobalAvgPool3d {
    #[new]
    /// Create a GlobalAvgPool3d layer.
    pub fn new() -> Self {
        Self
    }

    /// Forward pass: reduce `[N, C, D, H, W]` → `[N, C, 1, 1, 1]` by global average pooling.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let result = py.allow_threads(|| {
            coeus_nn::GlobalAvgPool3d::<f64, coeus_core::MoiraiBackend>::new().forward(&input_var)
        });
        result
            .map(|inner| PyTensor { inner })
            .map_err(map_module_error)
    }

    /// Return the list of learnable parameters (always empty — no parameters).
    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    /// Zero gradients of all parameters (no-op).
    pub fn zero_grad(&self, _py: Python<'_>) {}
}

/// Python-exposed Global Max Pooling 2D (reduces `[N, C, H, W]` → `[N, C, 1, 1]`).
#[pyclass(name = "GlobalMaxPool2d")]
#[derive(Default)]
pub struct PyGlobalMaxPool2d;

#[pymethods]
impl PyGlobalMaxPool2d {
    #[new]
    /// Create a GlobalMaxPool2d layer.
    pub fn new() -> Self {
        Self
    }

    /// Forward pass: reduce `[N, C, H, W]` → `[N, C, 1, 1]` by global max pooling.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let result = py.allow_threads(|| {
            coeus_nn::GlobalMaxPool2d::<f64, coeus_core::MoiraiBackend>::new().forward(&input_var)
        });
        result
            .map(|inner| PyTensor { inner })
            .map_err(map_module_error)
    }

    /// Return the list of learnable parameters (always empty — no parameters).
    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    /// Zero gradients of all parameters (no-op).
    pub fn zero_grad(&self, _py: Python<'_>) {}
}

/// Python-exposed Global Max Pooling 3D (reduces `[N, C, D, H, W]` → `[N, C, 1, 1, 1]`).
#[pyclass(name = "GlobalMaxPool3d")]
#[derive(Default)]
pub struct PyGlobalMaxPool3d;

#[pymethods]
impl PyGlobalMaxPool3d {
    #[new]
    /// Create a GlobalMaxPool3d layer.
    pub fn new() -> Self {
        Self
    }

    /// Forward pass: reduce `[N, C, D, H, W]` → `[N, C, 1, 1, 1]` by global max pooling.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let input_var = input.inner.clone();
        let result = py.allow_threads(|| {
            coeus_nn::GlobalMaxPool3d::<f64, coeus_core::MoiraiBackend>::new().forward(&input_var)
        });
        result
            .map(|inner| PyTensor { inner })
            .map_err(map_module_error)
    }

    /// Return the list of learnable parameters (always empty — no parameters).
    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    /// Zero gradients of all parameters (no-op).
    pub fn zero_grad(&self, _py: Python<'_>) {}
}
