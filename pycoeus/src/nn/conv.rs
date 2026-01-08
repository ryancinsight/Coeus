use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyErr, PyResult};
use pyo3::types::{PyTuple, PyAny}; // Check imports

use backend::CpuBackend;
use dtype::float::Float32;
use storage::DenseStorage;
use crate::tensor::PyTensor;

use nn::modules::convolution::Conv2D;
use nn::core::module::Module;

#[pyclass(name = "Conv2D", module = "nn", unsendable)]
#[derive(Clone)]
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
        kernel_size: Bound<PyAny>,
        stride: Option<Bound<PyAny>>,
        padding: Option<Bound<PyAny>>,
        bias: Option<bool>,
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
        
        let p = if let Some(pad) = padding {
            if let Ok(p_val) = pad.extract::<usize>() {
                Some((p_val, p_val))
            } else {
                Some(pad.extract::<(usize, usize)>()?)
            }
        } else {
            None
        };

        let conv = Conv2D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
            in_channels,
            out_channels,
            k,
            s,
            p,
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
    m.add_class::<PyConv2D>()?;
    Ok(())
}
