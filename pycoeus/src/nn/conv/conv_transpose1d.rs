use super::{to_py_err, ConvTranspose1DWrapper};
use crate::tensor::{PyTensor, TensorWrapper};
use backend::CpuBackend;
use coeus_nn::core::module::Module;
use coeus_nn::modules::convolution::ConvTranspose1d;
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

// ============================================================================
// ConvTranspose1d
// ============================================================================

#[pyclass(name = "ConvTranspose1d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyConvTranspose1d {
    pub inner: ConvTranspose1DWrapper,
}

#[pymethods]
impl PyConvTranspose1d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=None, padding=None, output_padding=None, groups=None, dilation=None, bias=None, dtype="float32", device="cpu"))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: Option<usize>,
        padding: Option<usize>,
        output_padding: Option<usize>,
        groups: Option<usize>,
        dilation: Option<usize>,
        bias: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let m =
                    ConvTranspose1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        output_padding,
                        groups,
                        dilation,
                        bias,
                    )
                    .map_err(to_py_err)?;
                ConvTranspose1DWrapper::CpuF32(m)
            }
            ("cpu", "float64") => {
                let m =
                    ConvTranspose1d::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        output_padding,
                        groups,
                        dilation,
                        bias,
                    )
                    .map_err(to_py_err)?;
                ConvTranspose1DWrapper::CpuF64(m)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let m =
                    ConvTranspose1d::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding,
                        output_padding,
                        groups,
                        dilation,
                        bias,
                    )
                    .map_err(to_py_err)?;
                ConvTranspose1DWrapper::GpuF32(m)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported device/dtype for ConvTranspose1d",
                ))
            }
        };

        Ok(PyConvTranspose1d { inner: wrapper })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (ConvTranspose1DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (ConvTranspose1DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (ConvTranspose1DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
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
            ConvTranspose1DWrapper::CpuF32(m) => m.train(mode),
            ConvTranspose1DWrapper::CpuF64(m) => m.train(mode),
            #[cfg(feature = "gpu")]
            ConvTranspose1DWrapper::GpuF32(m) => m.train(mode),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        let mut params = Vec::new();
        match &self.inner {
            ConvTranspose1DWrapper::CpuF32(m) => {
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(m.weight().data().clone()),
                });
                if let Some(b) = m.bias() {
                    params.push(PyTensor {
                        inner: TensorWrapper::CpuDenseF32(b.data().clone()),
                    });
                }
            }
            ConvTranspose1DWrapper::CpuF64(m) => {
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(m.weight().data().clone()),
                });
                if let Some(b) = m.bias() {
                    params.push(PyTensor {
                        inner: TensorWrapper::CpuDenseF64(b.data().clone()),
                    });
                }
            }
            #[cfg(feature = "gpu")]
            ConvTranspose1DWrapper::GpuF32(m) => {
                params.push(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(m.weight().data().clone()),
                });
                if let Some(b) = m.bias() {
                    params.push(PyTensor {
                        inner: TensorWrapper::GpuDenseF32(b.data().clone()),
                    });
                }
            }
        }
        params
    }
}
