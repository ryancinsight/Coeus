use backend::CpuBackend;
use dtype::float::Float32;
use nn::{BatchNorm2d, Conv2D, Dropout, Linear, Module, ReLU};
use storage::DenseStorage;
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, Py, PyErr, PyResult};

use crate::tensor::PyTensor;

/// Sequential container for chaining modules
#[pyclass(name = "Sequential", module = "_coeus", unsendable)]
pub struct PySequential {
    pub inner: coeus_nn::Sequential<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PySequential {
    #[new]
    fn new() -> PyResult<Self> {
        let sequential = coeus_nn::Sequential::new();
        Ok(PySequential { inner: sequential })
    }

    #[allow(deprecated)]
    fn add_module(&mut self, name: String, module: Py<PyAny>) -> PyResult<()> {
        // Dynamic module composition with trait object support
        // Supports common neural network modules that can be identified by type
        let module_str = format!("{:?}", module);

        pyo3::Python::with_gil(|_py| {
            // Try to identify the module type and create appropriate Rust module
            // This implementation provides basic dynamic composition for core NN modules

            if module_str.contains("ReLU") {
                let relu = ReLU;
                self.inner.add_module(name, relu);
                Ok(())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Dynamic module composition not supported for: {}. Use specific add_* methods (add_linear, add_conv2d, add_activation)", module_str)))
            }
        })
    }

    /// Add a Linear layer to the sequential model
    #[pyo3(signature = (name, in_features, out_features))]
    fn add_linear(
        &mut self,
        name: String,
        in_features: usize,
        out_features: usize,
    ) -> PyResult<()> {
        let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_features,
            out_features,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Sequential operation failed: {:?}", e)))?;
        self.inner.add_module(name, linear);
        Ok(())
    }

    /// Add a ReLU activation to the sequential model
    fn add_relu(&mut self, name: String) -> PyResult<()> {
        let relu = ReLU;
        self.inner.add_module(name, relu);
        Ok(())
    }

    /// Add a Conv2D layer to the sequential model
    #[pyo3(signature = (name, in_channels, out_channels, kernel_size, stride=None, padding=None, bias=None))]
    fn add_conv2d(
        &mut self,
        name: String,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        bias: Option<bool>,
    ) -> PyResult<()> {
        let conv = Conv2D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Sequential operation failed: {:?}", e)))?;
        self.inner.add_module(name, conv);
        Ok(())
    }

    /// Add a Dropout layer to the sequential model
    #[pyo3(signature = (name, p=0.5))]
    fn add_dropout(&mut self, name: String, p: Option<f64>) -> PyResult<()> {
        let p_val = p.unwrap_or(0.5);
        let dropout = Dropout::new(p_val);
        self.inner.add_module(name, dropout);
        Ok(())
    }

    /// Add a BatchNorm2d layer to the sequential model
    #[pyo3(signature = (name, num_features, eps=1e-5, momentum=0.1))]
    fn add_batch_norm2d(
        &mut self,
        name: String,
        num_features: usize,
        eps: Option<f64>,
        momentum: Option<f64>,
    ) -> PyResult<()> {
        let eps_val = eps.unwrap_or(1e-5);
        let momentum_val = momentum.unwrap_or(0.1);
        let batchnorm =
            BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::default(),
                num_features,
                eps_val,
                momentum_val,
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Sequential operation failed: {:?}", e)))?;
        self.inner.add_module(name, batchnorm);
        Ok(())
    }

    /// Add a simple identity/placeholder layer (for future normalization implementation)
    #[pyo3(signature = (_name, _normalized_shape))]
    fn add_layer_norm(&mut self, _name: String, _normalized_shape: Vec<usize>) -> PyResult<()> {
        // LayerNorm implementation requires feature flag - placeholder for now
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "LayerNorm layer - requires feature flag enablement in nn crate",
        ))
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self.inner.forward(&input.inner).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }

    #[getter]
    fn bias(&self) -> PyResult<Option<PyTensor>> {
        // Sequential doesn't have direct bias access
        Ok(None)
    }
}

/// Linear (fully connected) neural network layer Python binding
#[pyclass(name = "Linear", module = "_coeus", unsendable)]
pub struct PyLinear {
    pub inner: Linear<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyLinear {
    #[new]
    #[pyo3(signature = (in_features, out_features))]
    fn new(in_features: usize, out_features: usize) -> PyResult<Self> {
        let linear = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_features,
            out_features,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create Linear layer: {:?}",
                e
            ))
        })?;
        Ok(PyLinear { inner: linear })
    }

    #[getter]
    fn modules(&self) -> PyResult<Vec<String>> {
        let names = self.inner.child_module_names();
        let string_names = names.into_iter().map(|(_, name)| name).collect();
        Ok(string_names)
    }

    fn len(&self) -> usize {
        // Return number of parameters: weight + bias (both always present)
        2
    }

    #[getter]
    fn bias(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.bias.data().clone(),
        })
    }

    fn train(&mut self, mode: bool) {
        Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::train(&mut self.inner, mode);
    }
}

/// Conv2D layer Python binding
#[pyclass(name = "Conv2D", module = "_coeus", unsendable)]
pub struct PyConv2D {
    pub inner: Conv2D<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyConv2D {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=None, padding=None, bias=None))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
        bias: Option<bool>,
    ) -> PyResult<Self> {
        let conv = Conv2D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create Conv2D layer: {:?}",
                e
            ))
        })?;
        Ok(PyConv2D { inner: conv })
    }

    /// Alternative forward method for single-value kernel_size
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=None, padding=None, bias=None))]
    #[staticmethod]
    fn create(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        bias: Option<bool>,
    ) -> PyResult<Self> {
        Self::new(
            in_channels,
            out_channels,
            (kernel_size, kernel_size),
            stride.map(|s| (s, s)),
            padding.map(|p| (p, p)),
            bias,
        )
    }

    #[getter]
    fn weight(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.weight().data().clone(),
        })
    }

    #[getter]
    fn bias(&self) -> PyResult<Option<PyTensor>> {
        if let Some(bias_param) = self.inner.bias() {
            Ok(Some(PyTensor {
                inner: bias_param.data().clone(),
            }))
        } else {
            Ok(None)
        }
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self.inner.forward(&input.inner).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }

    fn train(&mut self, mode: bool) {
        Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::train(&mut self.inner, mode);
    }

    /// Get output size for given input dimensions
    #[pyo3(signature = (height, width))]
    fn output_size(&self, height: usize, width: usize) -> (usize, usize) {
        self.inner.output_size(height, width)
    }
}

/// BatchNorm2d layer Python binding
#[pyclass(name = "BatchNorm2d", module = "_coeus", unsendable)]
pub struct PyBatchNorm2d {
    pub inner: BatchNorm2d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyBatchNorm2d {
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1))]
    fn new(num_features: usize, eps: Option<f64>, momentum: Option<f64>) -> PyResult<Self> {
        let eps_val = eps.unwrap_or(1e-5);
        let momentum_val = momentum.unwrap_or(0.1);
        let batchnorm =
            BatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new_with_backend(
                CpuBackend::default(),
                num_features,
                eps_val,
                momentum_val,
            )
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to create BatchNorm2d layer: {:?}",
                    e
                ))
            })?;
        Ok(PyBatchNorm2d { inner: batchnorm })
    }

    #[getter]
    fn num_features(&self) -> usize {
        self.inner.num_features
    }

    #[getter]
    fn eps(&self) -> f64 {
        self.inner.eps
    }

    #[getter]
    fn momentum(&self) -> f64 {
        self.inner.momentum
    }

    #[getter]
    fn training(&self) -> bool {
        self.inner.training
    }

    #[getter]
    fn weight(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.weight.data().clone(),
        })
    }

    #[getter]
    fn bias(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.bias.data().clone(),
        })
    }

    #[getter]
    fn running_mean(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.running_mean(),
        })
    }

    #[getter]
    fn running_var(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.running_var(),
        })
    }

    fn train(&mut self, mode: bool) {
        Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::train(&mut self.inner, mode);
    }
}

/// Dropout layer Python binding
#[pyclass(name = "Dropout", module = "_coeus", unsendable)]
pub struct PyDropout {
    pub inner: Dropout,
}

#[pymethods]
impl PyDropout {
    #[new]
    #[pyo3(signature = (p=0.5))]
    fn new(p: Option<f64>) -> PyResult<Self> {
        let p_val = p.unwrap_or(0.5);
        let dropout = Dropout::new(p_val);
        Ok(PyDropout { inner: dropout })
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = self.inner.forward(&input.inner).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }

    #[getter]
    fn p(&self) -> f64 {
        self.inner.p as f64
    }

    #[getter]
    fn training(&self) -> bool {
        self.inner.training
    }

    fn train(&mut self, mode: bool) {
        self.inner.training = mode;
    }
}
