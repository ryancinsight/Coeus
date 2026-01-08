use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{pyclass, pymethods, Py, PyErr, PyResult};

use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use crate::tensor::PyTensor;
use tensor::Tensor;

use nn::containers::sequential::Sequential;
use nn::modules::linear::Linear;
use nn::modules::convolution::Conv2D;
use nn::modules::normalization::BatchNorm2d;
use nn::modules::activation::{ReLU, GeLU, SiLU};
//use nn::modules::regularization::dropout::Dropout;
use nn::modules::normalization::LayerNorm;

use nn::core::module::Module;

// Import binding classes for type extraction
use crate::nn::linear::PyLinear;
use crate::nn::conv::PyConv2D;
use crate::nn::normalization::PyBatchNorm2d; // PyLayerNorm not yet implemented
use crate::nn::activations::{PyReLU, PyGeLU, PySiLU};
// Note: PyDropout and PyLayerNorm need to be implemented in their respective files or added if missing.
// I haven't implemented PyDropout yet. I'll skip extracting it for now or implement it quickly.

#[pyclass(name = "Sequential", module = "nn", unsendable)]
#[derive(Clone)]
pub struct PySequential {
    pub inner: Sequential<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PySequential {
    #[new]
    #[pyo3(signature = (*args))]
    fn new(args: &Bound<PyTuple>) -> PyResult<Self> {
        let mut sequential = Sequential::new();

        for (i, arg) in args.iter().enumerate() {
            let name = i.to_string();

            // Try to downcast to known module types
            if let Ok(m) = arg.extract::<PyLinear>() {
                sequential.add_module(name, m.inner.clone());
            } else if let Ok(m) = arg.extract::<PyConv2D>() {
                sequential.add_module(name, m.inner.clone());
            } else if let Ok(m) = arg.extract::<PyReLU>() {
                sequential.add_module(name, m.inner.clone());
            } else if let Ok(m) = arg.extract::<PyGeLU>() {
                sequential.add_module(name, m.inner.clone());
            } else if let Ok(m) = arg.extract::<PySiLU>() {
                sequential.add_module(name, m.inner.clone());
            } else if let Ok(m) = arg.extract::<PyBatchNorm2d>() {
                 sequential.add_module(name, m.inner.clone());
            //} else if let Ok(m) = arg.extract::<PyLayerNorm>() {
            //    sequential.add_module(name, m.inner.clone());
            } else if let Ok(m) = arg.extract::<PySequential>() {
                sequential.add_module(name, m.inner.clone());
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "Sequential: Argument {} is not a supported Module (or use add_module). Supported: Linear, Conv2D, ReLU, GeLU, SiLU, BatchNorm2d.",
                    i
                )));
            }
        }
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

    /*
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
    */

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
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySequential>()?;
    Ok(())
}
