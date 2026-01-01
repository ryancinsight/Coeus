use backend::CpuBackend;
use dtype::float::Float32;
use nn::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AvgPool1d, AvgPool2d, BatchNorm2d, Conv2D, Dropout, GeLU,
    LayerNorm, Linear, MaxPool1d, MaxPool2d, Module, PReLU, ReLU, SiLU, GRU, LSTM, RNN,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::{pyclass, pymethods, Py, PyErr, PyResult};
use storage::DenseStorage;

use crate::tensor::PyTensor;

/// Sequential container for chaining modules
#[pyclass(name = "Sequential", module = "_coeus", unsendable)]
pub struct PySequential {
    pub inner: nn::Sequential<
        backend::CpuBackend<dtype::float::Float32>,
        storage::DenseStorage<dtype::float::Float32>,
        dtype::float::Float32,
    >,
}

#[pymethods]
impl PySequential {
    #[new]
    fn new() -> PyResult<Self> {
        let sequential = nn::Sequential::new();
        Ok(PySequential { inner: sequential })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[allow(deprecated)]
    fn add_module(&mut self, name: String, module: Py<PyAny>) -> PyResult<()> {
        let _ = (name, module);
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Sequential.add_module is not yet implemented; use add_linear/add_conv2d/add_relu/etc",
        ))
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
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Sequential operation failed: {:?}",
                e
            ))
        })?;
        self.inner.add_module(name, linear);
        Ok(())
    }

    /// Add a ReLU activation to the sequential model
    fn add_relu(&mut self, name: String) -> PyResult<()> {
        let relu = ReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
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
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Sequential operation failed: {:?}",
                e
            ))
        })?;
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
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Sequential operation failed: {:?}",
                    e
                ))
            })?;
        self.inner.add_module(name, batchnorm);
        Ok(())
    }

    /// Add a LayerNorm layer to the sequential model
    #[pyo3(signature = (name, normalized_shape, eps=1e-5))]
    fn add_layer_norm(
        &mut self,
        name: String,
        normalized_shape: Vec<usize>,
        eps: Option<f64>,
    ) -> PyResult<()> {
        let eps_val = eps.unwrap_or(1e-5);
        let layernorm = LayerNorm::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            normalized_shape,
            eps_val,
        );
        self.inner.add_module(name, layernorm);
        Ok(())
    }

    /// Add a RNN layer to the sequential model
    #[pyo3(signature = (name, input_size, hidden_size, num_layers=1, bias=true, batch_first=false, bidirectional=false))]
    fn add_rnn(
        &mut self,
        name: String,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        bidirectional: bool,
    ) -> PyResult<()> {
        let rnn = RNN::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            bidirectional,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create RNN: {:?}",
                e
            ))
        })?;
        self.inner.add_module(name, rnn);
        Ok(())
    }

    /// Add a LSTM layer to the sequential model
    #[pyo3(signature = (name, input_size, hidden_size, num_layers=1, bias=true, batch_first=false, bidirectional=false))]
    fn add_lstm(
        &mut self,
        name: String,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        bidirectional: bool,
    ) -> PyResult<()> {
        let lstm = LSTM::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            bidirectional,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create LSTM: {:?}",
                e
            ))
        })?;
        self.inner.add_module(name, lstm);
        Ok(())
    }

    /// Add a GRU layer to the sequential model
    #[pyo3(signature = (name, input_size, hidden_size, num_layers=1, bias=true, batch_first=false, bidirectional=false))]
    fn add_gru(
        &mut self,
        name: String,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        bidirectional: bool,
    ) -> PyResult<()> {
        let gru = GRU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            bidirectional,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create GRU: {:?}",
                e
            ))
        })?;
        self.inner.add_module(name, gru);
        Ok(())
    }

    /// Add a GeLU activation to the sequential model
    fn add_gelu(&mut self, name: String) -> PyResult<()> {
        let gelu = GeLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        self.inner.add_module(name, gelu);
        Ok(())
    }

    /// Add a SiLU activation to the sequential model
    fn add_silu(&mut self, name: String) -> PyResult<()> {
        let silu = SiLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new();
        self.inner.add_module(name, silu);
        Ok(())
    }

    /// Add a PReLU activation to the sequential model
    #[pyo3(signature = (name, num_parameters=1, init=0.25))]
    fn add_prelu(&mut self, name: String, num_parameters: usize, init: f32) -> PyResult<()> {
        let prelu = PReLU::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            num_parameters,
            Some(Float32::new(init)),
        );
        self.inner.add_module(name, prelu);
        Ok(())
    }

    /// Add a Sigmoid activation to the sequential model
    fn add_sigmoid(&mut self, name: String) -> PyResult<()> {
        let _ = name;
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "Sigmoid module is not yet implemented; use functional sigmoid instead",
        ))
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
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

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let params = self.inner.parameters();
        let py_params = params
            .into_iter()
            .map(|p| PyTensor {
                inner: p.data().clone(),
            })
            .collect();
        Ok(py_params)
    }

    fn named_parameters(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let params = self.inner.parameters();
        let py_params = params
            .into_iter()
            .map(|p| {
                (
                    p.name().to_string(),
                    PyTensor {
                        inner: p.data().clone(),
                    },
                )
            })
            .collect();
        Ok(py_params)
    }

    fn state_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        // Add parameters
        for (name, tensor) in self.named_parameters()? {
            dict.set_item(name, tensor)?;
        }
        // Add buffers
        for (name, tensor) in self.named_buffers()? {
            dict.set_item(name, tensor)?;
        }
        Ok(dict.unbind().into())
    }

    fn load_state_dict(&mut self, state_dict: Bound<PyDict>) -> PyResult<()> {
        let params = self.inner.parameters(); // Get params with hierarchical names
        for mut param in params {
            let name = param.name();
            if let Some(item) = state_dict.get_item(name)? {
                let pytensor: PyTensor = item.extract()?;
                *param.data_mut() = pytensor.inner.clone();
            }
        }

        // Load buffers
        // We iterate named_buffers to get names, look them up in state_dict, and call load_buffer
        let buffers = self.inner.named_buffers(); // returns Vec<(String, Tensor)> clones
        for (name, _) in buffers {
            if let Some(item) = state_dict.get_item(&name)? {
                let pytensor: PyTensor = item.extract()?;
                self.inner
                    .load_buffer(&name, &pytensor.inner)
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to load buffer {}: {:?}",
                            name, e
                        ))
                    })?;
            }
        }
        Ok(())
    }

    fn named_buffers(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let buffers = self.inner.named_buffers();
        let py_buffers = buffers
            .into_iter()
            .map(|(n, t)| (n, PyTensor { inner: t }))
            .collect();
        Ok(py_buffers)
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

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(
            &self.inner,
            &input.inner,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let params = self.inner.parameters();
        let py_params = params
            .into_iter()
            .map(|p| PyTensor {
                inner: p.data().clone(),
            })
            .collect();
        Ok(py_params)
    }

    fn named_parameters(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let params = self.inner.parameters();
        let py_params = params
            .into_iter()
            .map(|p| {
                (
                    p.name().to_string(),
                    PyTensor {
                        inner: p.data().clone(),
                    },
                )
            })
            .collect();
        Ok(py_params)
    }

    fn state_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (name, tensor) in self.named_parameters()? {
            dict.set_item(name, tensor)?;
        }
        Ok(dict.unbind().into())
    }

    fn load_state_dict(&mut self, state_dict: Bound<PyDict>) -> PyResult<()> {
        let params = self.inner.parameters();
        for mut param in params {
            let name = param.name();
            if let Some(item) = state_dict.get_item(name)? {
                let pytensor: PyTensor = item.extract()?;
                *param.data_mut() = pytensor.inner.clone();
            }
        }
        Ok(())
    }
}

/// Embedding layer Python binding
#[pyclass(name = "Embedding", module = "_coeus", unsendable)]
pub struct PyEmbedding {
    pub inner: nn::embedding::Embedding<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    #[pyo3(signature = (num_embeddings, embedding_dim, padding_idx=None))]
    fn new(
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
    ) -> PyResult<Self> {
        let embedding = nn::embedding::Embedding::<
            CpuBackend<Float32>,
            DenseStorage<Float32>,
            Float32,
        >::new(num_embeddings, embedding_dim, padding_idx)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create Embedding layer: {:?}",
                e
            ))
        })?;
        Ok(PyEmbedding { inner: embedding })
    }

    #[getter]
    fn weight(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.weight.data().clone(),
        })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(
            &self.inner,
            &input.inner,
        )
        .map_err(|e| {
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

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
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

    fn output_size(&self, height: usize, width: usize) -> (usize, usize) {
        self.inner.output_size(height, width)
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let params = self.inner.parameters();
        let py_params = params
            .into_iter()
            .map(|p| PyTensor {
                inner: p.data().clone(),
            })
            .collect();
        Ok(py_params)
    }

    fn named_parameters(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let params = self.inner.parameters();
        let py_params = params
            .into_iter()
            .map(|p| {
                (
                    p.name().to_string(),
                    PyTensor {
                        inner: p.data().clone(),
                    },
                )
            })
            .collect();
        Ok(py_params)
    }

    fn named_buffers(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let buffers = self.inner.named_buffers();
        let py_buffers = buffers
            .into_iter()
            .map(|(n, t)| (n, PyTensor { inner: t }))
            .collect();
        Ok(py_buffers)
    }

    fn state_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (name, tensor) in self.named_parameters()? {
            dict.set_item(name, tensor)?;
        }
        for (name, tensor) in self.named_buffers()? {
            dict.set_item(name, tensor)?;
        }
        Ok(dict.unbind().into())
    }

    fn load_state_dict(&mut self, state_dict: Bound<PyDict>) -> PyResult<()> {
        let params = self.inner.parameters();
        for mut param in params {
            let name = param.name();
            if let Some(item) = state_dict.get_item(name)? {
                let pytensor: PyTensor = item.extract()?;
                *param.data_mut() = pytensor.inner.clone();
            }
        }
        let buffers = self.inner.named_buffers();
        for (name, _) in buffers {
            if let Some(item) = state_dict.get_item(&name)? {
                let pytensor: PyTensor = item.extract()?;
                self.inner
                    .load_buffer(&name, &pytensor.inner)
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to load buffer {}: {:?}",
                            name, e
                        ))
                    })?;
            }
        }
        Ok(())
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

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let params = self.inner.parameters();
        let py_params = params
            .into_iter()
            .map(|p| PyTensor {
                inner: p.data().clone(),
            })
            .collect();
        Ok(py_params)
    }

    fn named_parameters(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let params = self.inner.parameters();
        let py_params = params
            .into_iter()
            .map(|p| {
                (
                    p.name().to_string(),
                    PyTensor {
                        inner: p.data().clone(),
                    },
                )
            })
            .collect();
        Ok(py_params)
    }

    fn named_buffers(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let buffers = self.inner.named_buffers();
        let py_buffers = buffers
            .into_iter()
            .map(|(n, t)| (n, PyTensor { inner: t }))
            .collect();
        Ok(py_buffers)
    }

    fn state_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (name, tensor) in self.named_parameters()? {
            dict.set_item(name, tensor)?;
        }
        for (name, tensor) in self.named_buffers()? {
            dict.set_item(name, tensor)?;
        }
        Ok(dict.unbind().into())
    }

    fn load_state_dict(&mut self, state_dict: Bound<PyDict>) -> PyResult<()> {
        let params = self.inner.parameters();
        for mut param in params {
            let name = param.name();
            if let Some(item) = state_dict.get_item(name)? {
                let pytensor: PyTensor = item.extract()?;
                *param.data_mut() = pytensor.inner.clone();
            }
        }
        let buffers = self.inner.named_buffers();
        for (name, _) in buffers {
            if let Some(item) = state_dict.get_item(&name)? {
                let pytensor: PyTensor = item.extract()?;
                self.inner
                    .load_buffer(&name, &pytensor.inner)
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to load buffer {}: {:?}",
                            name, e
                        ))
                    })?;
            }
        }
        Ok(())
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

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
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
        self.inner.p
    }

    #[getter]
    fn training(&self) -> bool {
        self.inner.training
    }

    fn train(&mut self, mode: bool) {
        self.inner.training = mode;
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        Ok(Vec::new())
    }

    fn named_parameters(&self) -> PyResult<Vec<(String, PyTensor)>> {
        Ok(Vec::new())
    }

    fn named_buffers(&self) -> PyResult<Vec<(String, PyTensor)>> {
        Ok(Vec::new())
    }

    fn state_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        Ok(dict.unbind().into())
    }

    fn load_state_dict(&mut self, _state_dict: Bound<PyDict>) -> PyResult<()> {
        Ok(())
    }
}

/// MaxPool1d layer Python binding
#[pyclass(name = "MaxPool1d", module = "_coeus", unsendable)]
pub struct PyMaxPool1d {
    pub inner: MaxPool1d,
}

#[pymethods]
impl PyMaxPool1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=0))]
    fn new(
        kernel_size: Bound<PyAny>,
        stride: Option<Bound<PyAny>>,
        padding: usize,
    ) -> PyResult<Self> {
        let k = if let Ok(s) = kernel_size.extract::<usize>() {
            s
        } else {
            kernel_size.extract::<usize>()?
        };
        let s = if let Some(st) = stride {
            Some(st.extract::<usize>()?)
        } else {
            None
        };
        Ok(PyMaxPool1d {
            inner: MaxPool1d::new(k, s, padding),
        })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
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
}

/// MaxPool2d layer Python binding
#[pyclass(name = "MaxPool2d", module = "_coeus", unsendable)]
pub struct PyMaxPool2d {
    pub inner: MaxPool2d,
}

#[pymethods]
impl PyMaxPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None))]
    fn new(
        kernel_size: Bound<PyAny>,
        stride: Option<Bound<PyAny>>,
        padding: Option<Bound<PyAny>>,
    ) -> PyResult<Self> {
        let k = if let Ok(s) = kernel_size.extract::<usize>() {
            (s, s)
        } else {
            kernel_size.extract::<(usize, usize)>()?
        };
        let s = if let Some(st) = stride {
            if let Ok(s_val) = st.extract::<usize>() {
                Some((s_val, s_val))
            } else {
                Some(st.extract::<(usize, usize)>()?)
            }
        } else {
            None
        };
        let p = if let Some(padding) = padding {
            if let Ok(p_val) = padding.extract::<usize>() {
                (p_val, p_val)
            } else {
                padding.extract::<(usize, usize)>()?
            }
        } else {
            (0, 0)
        };
        Ok(PyMaxPool2d {
            inner: MaxPool2d::new(k, s, p),
        })
    }

    #[pyo3(signature = (kernel_size, stride=None, padding=0))]
    #[staticmethod]
    fn create(kernel_size: usize, stride: Option<usize>, padding: usize) -> Self {
        PyMaxPool2d {
            inner: MaxPool2d::new(
                (kernel_size, kernel_size),
                stride.map(|s| (s, s)),
                (padding, padding),
            ),
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
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
}

/// AvgPool1d layer Python binding
#[pyclass(name = "AvgPool1d", module = "_coeus", unsendable)]
pub struct PyAvgPool1d {
    pub inner: AvgPool1d,
}

#[pymethods]
impl PyAvgPool1d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=0))]
    fn new(
        kernel_size: Bound<PyAny>,
        stride: Option<Bound<PyAny>>,
        padding: usize,
    ) -> PyResult<Self> {
        let k = if let Ok(s) = kernel_size.extract::<usize>() {
            s
        } else {
            kernel_size.extract::<usize>()?
        };
        let s = if let Some(st) = stride {
            Some(st.extract::<usize>()?)
        } else {
            None
        };
        Ok(PyAvgPool1d {
            inner: AvgPool1d::new(k, s, padding),
        })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
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
}

/// AvgPool2d layer Python binding
#[pyclass(name = "AvgPool2d", module = "_coeus", unsendable)]
pub struct PyAvgPool2d {
    pub inner: AvgPool2d,
}

#[pymethods]
impl PyAvgPool2d {
    #[new]
    #[pyo3(signature = (kernel_size, stride=None, padding=None))]
    fn new(
        kernel_size: Bound<PyAny>,
        stride: Option<Bound<PyAny>>,
        padding: Option<Bound<PyAny>>,
    ) -> PyResult<Self> {
        let k = if let Ok(s) = kernel_size.extract::<usize>() {
            (s, s)
        } else {
            kernel_size.extract::<(usize, usize)>()?
        };
        let s = if let Some(st) = stride {
            if let Ok(s_val) = st.extract::<usize>() {
                Some((s_val, s_val))
            } else {
                Some(st.extract::<(usize, usize)>()?)
            }
        } else {
            None
        };
        let p = if let Some(padding) = padding {
            if let Ok(p_val) = padding.extract::<usize>() {
                (p_val, p_val)
            } else {
                padding.extract::<(usize, usize)>()?
            }
        } else {
            (0, 0)
        };
        Ok(PyAvgPool2d {
            inner: AvgPool2d::new(k, s, p),
        })
    }

    #[pyo3(signature = (kernel_size, stride=None, padding=0))]
    #[staticmethod]
    fn create(kernel_size: usize, stride: Option<usize>, padding: usize) -> Self {
        PyAvgPool2d {
            inner: AvgPool2d::new(
                (kernel_size, kernel_size),
                stride.map(|s| (s, s)),
                (padding, padding),
            ),
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
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
}

/// AdaptiveAvgPool1d layer Python binding
#[pyclass(name = "AdaptiveAvgPool1d", module = "_coeus", unsendable)]
pub struct PyAdaptiveAvgPool1d {
    pub inner: AdaptiveAvgPool1d,
}

#[pymethods]
impl PyAdaptiveAvgPool1d {
    #[new]
    #[pyo3(signature = (output_size))]
    fn new(output_size: Bound<PyAny>) -> PyResult<Self> {
        let size = if let Ok(s) = output_size.extract::<usize>() {
            s
        } else {
            output_size.extract::<usize>()?
        };
        Ok(PyAdaptiveAvgPool1d {
            inner: AdaptiveAvgPool1d::new(size),
        })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
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
}

/// AdaptiveAvgPool2d layer Python binding
#[pyclass(name = "AdaptiveAvgPool2d", module = "_coeus", unsendable)]
pub struct PyAdaptiveAvgPool2d {
    pub inner: AdaptiveAvgPool2d,
}

#[pymethods]
impl PyAdaptiveAvgPool2d {
    #[new]
    #[pyo3(signature = (output_size))]
    fn new(output_size: Bound<PyAny>) -> PyResult<Self> {
        let size = if let Ok(s) = output_size.extract::<usize>() {
            (s, s)
        } else {
            output_size.extract::<(usize, usize)>()?
        };
        Ok(PyAdaptiveAvgPool2d {
            inner: AdaptiveAvgPool2d::new(size),
        })
    }

    #[pyo3(signature = (output_size))]
    #[staticmethod]
    fn create(output_size: usize) -> Self {
        PyAdaptiveAvgPool2d {
            inner: AdaptiveAvgPool2d::new((output_size, output_size)),
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
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
}

/// LayerNorm layer Python binding
#[pyclass(name = "LayerNorm", module = "_coeus", unsendable)]
pub struct PyLayerNorm {
    pub inner: LayerNorm<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyLayerNorm {
    #[new]
    #[pyo3(signature = (normalized_shape, eps=1e-5))]
    fn new(normalized_shape: Bound<PyAny>, eps: Option<f64>) -> PyResult<Self> {
        let eps_val = eps.unwrap_or(1e-5);
        let shape = if let Ok(s) = normalized_shape.extract::<usize>() {
            vec![s]
        } else {
            normalized_shape.extract::<Vec<usize>>()?
        };
        Ok(PyLayerNorm {
            inner: LayerNorm::new(shape, eps_val),
        })
    }

    #[getter]
    fn normalized_shape(&self) -> Vec<usize> {
        self.inner.normalized_shape.clone()
    }

    #[getter]
    fn eps(&self) -> f64 {
        self.inner.eps
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

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(
            &self.inner,
            &input.inner,
        )
        .map_err(|e| {
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

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let params = self.inner.parameters();
        let py_params = params
            .into_iter()
            .map(|p| PyTensor {
                inner: p.data().clone(),
            })
            .collect();
        Ok(py_params)
    }

    fn named_parameters(&self) -> PyResult<Vec<(String, PyTensor)>> {
        let params = self.inner.parameters();
        let py_params = params
            .into_iter()
            .map(|p| {
                (
                    p.name().to_string(),
                    PyTensor {
                        inner: p.data().clone(),
                    },
                )
            })
            .collect();
        Ok(py_params)
    }

    fn state_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let dict = PyDict::new(py);
        for (name, tensor) in self.named_parameters()? {
            dict.set_item(name, tensor)?;
        }
        Ok(dict.unbind().into())
    }

    fn load_state_dict(&mut self, state_dict: Bound<PyDict>) -> PyResult<()> {
        let params = self.inner.parameters();
        for mut param in params {
            let name = param.name();
            if let Some(item) = state_dict.get_item(name)? {
                let pytensor: PyTensor = item.extract()?;
                *param.data_mut() = pytensor.inner.clone();
            }
        }
        Ok(())
    }
}

/// ReLU activation Python binding
#[pyclass(name = "ReLU", module = "_coeus", unsendable)]
pub struct PyReLU {
    pub inner: ReLU<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyReLU {
    #[new]
    fn new() -> Self {
        PyReLU { inner: ReLU::new() }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(
            &self.inner,
            &input.inner,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }
}

/// GeLU activation Python binding
#[pyclass(name = "GELU", module = "_coeus", unsendable)]
pub struct PyGeLU {
    pub inner: GeLU<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyGeLU {
    #[new]
    fn new() -> Self {
        PyGeLU { inner: GeLU::new() }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(
            &self.inner,
            &input.inner,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }
}

/// SiLU activation Python binding
#[pyclass(name = "SiLU", module = "_coeus", unsendable)]
pub struct PySiLU {
    pub inner: SiLU<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PySiLU {
    #[new]
    fn new() -> Self {
        PySiLU { inner: SiLU::new() }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(
            &self.inner,
            &input.inner,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }
}

/// PReLU activation Python binding
#[pyclass(name = "PReLU", module = "_coeus", unsendable)]
pub struct PyPReLU {
    pub inner: PReLU<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyPReLU {
    #[new]
    #[pyo3(signature = (num_parameters=1, init=0.25))]
    fn new(num_parameters: usize, init: f32) -> Self {
        PyPReLU {
            inner: PReLU::new(num_parameters, Some(Float32::new(init))),
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        let output = Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::forward(
            &self.inner,
            &input.inner,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Forward pass failed: {:?}",
                e
            ))
        })?;
        Ok(PyTensor { inner: output })
    }

    #[getter]
    fn weight(&self) -> PyResult<PyTensor> {
        Ok(PyTensor {
            inner: self.inner.weight.data().clone(),
        })
    }
}

/// Tanh activation Python binding
#[pyclass(name = "Tanh", module = "_coeus", unsendable)]
pub struct PyTanh {}

#[pymethods]
impl PyTanh {
    #[new]
    fn new() -> Self {
        PyTanh {}
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        crate::functional::tanh(input)
    }
}

/// Sigmoid activation Python binding
#[pyclass(name = "Sigmoid", module = "_coeus", unsendable)]
pub struct PySigmoid {}

#[pymethods]
impl PySigmoid {
    #[new]
    fn new() -> Self {
        PySigmoid {}
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        crate::functional::sigmoid(input)
    }
}

/// RNN layer Python binding
#[pyclass(name = "RNN", module = "_coeus", unsendable)]
pub struct PyRNN {
    pub inner: RNN<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyRNN {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=1, bias=true, batch_first=false, bidirectional=false))]
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        bidirectional: bool,
    ) -> PyResult<Self> {
        let rnn = RNN::new(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            bidirectional,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create RNN: {:?}",
                e
            ))
        })?;
        Ok(PyRNN { inner: rnn })
    }

    #[pyo3(signature = (input, hidden=None))]
    fn __call__(
        &self,
        input: &PyTensor,
        hidden: Option<&PyTensor>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        self.forward(input, hidden)
    }

    fn forward(
        &self,
        input: &PyTensor,
        hidden: Option<&PyTensor>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (output, h_n) = self
            .inner
            .forward_with_hidden(&input.inner, hidden.map(|h| &h.inner))
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Forward pass failed: {:?}",
                    e
                ))
            })?;
        Ok((PyTensor { inner: output }, PyTensor { inner: h_n }))
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let params =
            Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::parameters(&self.inner);
        let py_params = params
            .into_iter()
            .map(|p| PyTensor {
                inner: p.data().clone(),
            })
            .collect();
        Ok(py_params)
    }
}

/// LSTM layer Python binding
#[pyclass(name = "LSTM", module = "_coeus", unsendable)]
pub struct PyLSTM {
    pub inner: LSTM<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyLSTM {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=1, bias=true, batch_first=false, bidirectional=false))]
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        bidirectional: bool,
    ) -> PyResult<Self> {
        let lstm = LSTM::new(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            bidirectional,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create LSTM: {:?}",
                e
            ))
        })?;
        Ok(PyLSTM { inner: lstm })
    }

    #[pyo3(signature = (input, state=None))]
    fn __call__(
        &self,
        input: &PyTensor,
        state: Option<(PyRef<PyTensor>, PyRef<PyTensor>)>,
    ) -> PyResult<(PyTensor, (PyTensor, PyTensor))> {
        self.forward(input, state)
    }

    fn forward(
        &self,
        input: &PyTensor,
        state: Option<(PyRef<PyTensor>, PyRef<PyTensor>)>,
    ) -> PyResult<(PyTensor, (PyTensor, PyTensor))> {
        let (output, (h_n, c_n)) = self
            .inner
            .forward(
                &input.inner,
                state.as_ref().map(|(h, c)| (&h.inner, &c.inner)),
            )
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Forward pass failed: {:?}",
                    e
                ))
            })?;
        Ok((
            PyTensor { inner: output },
            (PyTensor { inner: h_n }, PyTensor { inner: c_n }),
        ))
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let params =
            Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::parameters(&self.inner);
        let py_params = params
            .into_iter()
            .map(|p| PyTensor {
                inner: p.data().clone(),
            })
            .collect();
        Ok(py_params)
    }
}

/// GRU layer Python binding
#[pyclass(name = "GRU", module = "_coeus", unsendable)]
pub struct PyGRU {
    pub inner: GRU<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyGRU {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=1, bias=true, batch_first=false, bidirectional=false))]
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        bidirectional: bool,
    ) -> PyResult<Self> {
        let gru = GRU::new(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            bidirectional,
        )
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to create GRU: {:?}",
                e
            ))
        })?;
        Ok(PyGRU { inner: gru })
    }

    #[pyo3(signature = (input, hidden=None))]
    fn __call__(
        &self,
        input: &PyTensor,
        hidden: Option<&PyTensor>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        self.forward(input, hidden)
    }

    fn forward(
        &self,
        input: &PyTensor,
        hidden: Option<&PyTensor>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let (output, h_n) = self
            .inner
            .forward_with_hidden(&input.inner, hidden.map(|h| &h.inner))
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Forward pass failed: {:?}",
                    e
                ))
            })?;
        Ok((PyTensor { inner: output }, PyTensor { inner: h_n }))
    }

    fn parameters(&self) -> PyResult<Vec<PyTensor>> {
        let params =
            Module::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::parameters(&self.inner);
        let py_params = params
            .into_iter()
            .map(|p| PyTensor {
                inner: p.data().clone(),
            })
            .collect();
        Ok(py_params)
    }
}
