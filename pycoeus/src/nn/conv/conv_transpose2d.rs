use super::{to_py_err, ConvTranspose2DWrapper};
use crate::tensor::{PyTensor, TensorWrapper};
use backend::CpuBackend;
use coeus_nn::core::module::Module;
use coeus_nn::modules::convolution::ConvTranspose2d;
use dtype::float::{Float32, Float64};
use pyo3::prelude::*;
use storage::DenseStorage;

#[cfg(feature = "gpu")]
use backend::GpuBackend;

// ============================================================================
// ConvTranspose2d
// ============================================================================

#[pyclass(name = "ConvTranspose2d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyConvTranspose2d {
    pub inner: ConvTranspose2DWrapper,
}

#[pymethods]
impl PyConvTranspose2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=None, padding=None, output_padding=None, bias=None, dtype="float32", device="cpu"))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: Bound<PyAny>,
        stride: Option<Bound<PyAny>>,
        padding: Option<Bound<PyAny>>,
        output_padding: Option<Bound<PyAny>>,
        bias: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

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

        let op = if let Some(opad) = output_padding {
            if let Ok(op_val) = opad.extract::<usize>() {
                Some((op_val, op_val))
            } else {
                Some(opad.extract::<(usize, usize)>()?)
            }
        } else {
            None
        };

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let m =
                    ConvTranspose2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                        in_channels,
                        out_channels,
                        k,
                        s,
                        p,
                        op,
                        bias,
                    )
                    .map_err(to_py_err)?;
                ConvTranspose2DWrapper::CpuF32(m)
            }
            ("cpu", "float64") => {
                let m =
                    ConvTranspose2d::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                        in_channels,
                        out_channels,
                        k,
                        s,
                        p,
                        op,
                        bias,
                    )
                    .map_err(to_py_err)?;
                ConvTranspose2DWrapper::CpuF64(m)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let m =
                    ConvTranspose2d::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                        in_channels,
                        out_channels,
                        k,
                        s,
                        p,
                        op,
                        bias,
                    )
                    .map_err(to_py_err)?;
                ConvTranspose2DWrapper::GpuF32(m)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported device/dtype for ConvTranspose2d",
                ))
            }
        };

        Ok(PyConvTranspose2d { inner: wrapper })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (ConvTranspose2DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (ConvTranspose2DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (ConvTranspose2DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
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
            ConvTranspose2DWrapper::CpuF32(m) => m.train(mode),
            ConvTranspose2DWrapper::CpuF64(m) => m.train(mode),
            #[cfg(feature = "gpu")]
            ConvTranspose2DWrapper::GpuF32(m) => m.train(mode),
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        let mut params = Vec::new();
        match &self.inner {
            ConvTranspose2DWrapper::CpuF32(m) => {
                params.push(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(m.weight().data().clone()),
                });
                if let Some(b) = m.bias() {
                    params.push(PyTensor {
                        inner: TensorWrapper::CpuDenseF32(b.data().clone()),
                    });
                }
            }
            ConvTranspose2DWrapper::CpuF64(m) => {
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
            ConvTranspose2DWrapper::GpuF32(m) => {
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
