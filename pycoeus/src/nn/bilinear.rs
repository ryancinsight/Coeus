use crate::tensor::{PyTensor, TensorWrapper};
use pyo3::prelude::*;

use backend::CpuBackend;
#[cfg(feature = "gpu")]
use backend::GpuBackend;
use coeus_nn::modules::linear::Bilinear;
use dtype::float::{Float32, Float64};
use storage::DenseStorage;

#[derive(Clone)]
pub enum BilinearWrapper {
    CpuF32(Bilinear<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(Bilinear<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(Bilinear<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    crate::error::convert_error(format!("layer: Bilinear error: {}", e))
}

#[pyclass(name = "Bilinear", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyBilinear {
    pub inner: BilinearWrapper,
    pub use_bias: bool,
}

#[pymethods]
impl PyBilinear {
    #[new]
    #[pyo3(signature = (in1_features, in2_features, out_features, bias=true, dtype="float32", device="cpu"))]
    fn new(
        in1_features: usize,
        in2_features: usize,
        out_features: usize,
        bias: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let use_bias = bias.unwrap_or(true);
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let layer = Bilinear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                    in1_features,
                    in2_features,
                    out_features,
                    use_bias,
                ).map_err(to_py_err)?;
                BilinearWrapper::CpuF32(layer)
            }
            ("cpu", "float64") => {
                let layer = Bilinear::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                    in1_features,
                    in2_features,
                    out_features,
                    use_bias,
                ).map_err(to_py_err)?;
                BilinearWrapper::CpuF64(layer)
            }
            // Add GPU support if needed
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Unsupported device/dtype: {}/{}", device, dtype))),
        };

        Ok(PyBilinear { inner: wrapper, use_bias })
    }

    fn forward(&self, input1: &PyTensor, input2: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input1.inner, &input2.inner) {
            (BilinearWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i1), TensorWrapper::CpuDenseF32(i2)) => {
                let res = m.forward_bilinear(i1, i2).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (BilinearWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i1), TensorWrapper::CpuDenseF64(i2)) => {
                let res = m.forward_bilinear(i1, i2).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Backend mismatch"))
        }
    }

    fn __call__(&self, input1: &PyTensor, input2: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input1, input2)
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBilinear>()?;
    Ok(())
}
