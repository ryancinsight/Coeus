use super::{to_py_err, Conv1DWrapper};
use crate::tensor::{PyTensor, TensorWrapper};
use backend::CpuBackend;
use coeus_nn::core::module::Module;
use coeus_nn::modules::convolution::Conv1D;
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

// ============================================================================
// Conv1d
// ============================================================================

#[pyclass(name = "Conv1d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyConv1d {
    pub inner: Conv1DWrapper,
}

#[pymethods]
impl PyConv1d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=None, padding=None, bias=None, dtype="float32", device="cpu"))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        bias: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let m = Conv1D::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias,
                )
                .map_err(to_py_err)?;
                Conv1DWrapper::CpuF32(m)
            }
            ("cpu", "float64") => {
                let m = Conv1D::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias,
                )
                .map_err(to_py_err)?;
                Conv1DWrapper::CpuF64(m)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let m = Conv1D::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias,
                )
                .map_err(to_py_err)?;
                Conv1DWrapper::GpuF32(m)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported device/dtype for Conv1d",
                ))
            }
        };

        Ok(PyConv1d { inner: wrapper })
    }

    #[getter]
    fn weight(&self) -> PyTensor {
        match &self.inner {
            Conv1DWrapper::CpuF32(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF32(m.weight().data().clone()),
            },
            Conv1DWrapper::CpuF64(m) => PyTensor {
                inner: TensorWrapper::CpuDenseF64(m.weight().data().clone()),
            },
            #[cfg(feature = "gpu")]
            Conv1DWrapper::GpuF32(m) => PyTensor {
                inner: TensorWrapper::GpuDenseF32(m.weight().data().clone()),
            },
        }
    }

    #[getter]
    fn bias(&self) -> Option<PyTensor> {
        match &self.inner {
            Conv1DWrapper::CpuF32(m) => m.bias().map(|b| PyTensor {
                inner: TensorWrapper::CpuDenseF32(b.data().clone()),
            }),
            Conv1DWrapper::CpuF64(m) => m.bias().map(|b| PyTensor {
                inner: TensorWrapper::CpuDenseF64(b.data().clone()),
            }),
            #[cfg(feature = "gpu")]
            Conv1DWrapper::GpuF32(m) => m.bias().map(|b| PyTensor {
                inner: TensorWrapper::GpuDenseF32(b.data().clone()),
            }),
        }
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (Conv1DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (Conv1DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (Conv1DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Input/Module backend/dtype mismatch",
            )),
        }
    }

    fn train(&mut self, mode: bool) {
        match &mut self.inner {
            Conv1DWrapper::CpuF32(m) => m.train(mode),
            Conv1DWrapper::CpuF64(m) => m.train(mode),
            #[cfg(feature = "gpu")]
            Conv1DWrapper::GpuF32(m) => m.train(mode),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        let mut params = Vec::new();
        params.push(self.weight());
        if let Some(b) = self.bias() {
            params.push(b);
        }
        params
    }
}
