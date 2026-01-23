use crate::tensor::{PyTensor, TensorWrapper};
use pyo3::prelude::*;

use backend::CpuBackend;
#[cfg(feature = "gpu")]
use backend::GpuBackend;
use coeus_nn::core::module::Module;
use coeus_nn::modules::normalization::{LazyBatchNorm1d, LazyBatchNorm2d, LazyBatchNorm3d};
use dtype::float::{Float32, Float64};
use storage::DenseStorage;

use super::to_py_err;

#[derive(Clone)]
pub enum LazyBatchNorm1DWrapper {
    CpuF32(LazyBatchNorm1d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(LazyBatchNorm1d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(LazyBatchNorm1d<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum LazyBatchNorm2DWrapper {
    CpuF32(LazyBatchNorm2d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(LazyBatchNorm2d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(LazyBatchNorm2d<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[derive(Clone)]
pub enum LazyBatchNorm3DWrapper {
    CpuF32(LazyBatchNorm3d<CpuBackend<Float32>, DenseStorage<Float32>, Float32>),
    CpuF64(LazyBatchNorm3d<CpuBackend<Float64>, DenseStorage<Float64>, Float64>),
    #[cfg(feature = "gpu")]
    GpuF32(LazyBatchNorm3d<GpuBackend<Float32>, DenseStorage<Float32>, Float32>),
}

#[pyclass(name = "LazyBatchNorm1d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyLazyBatchNorm1d {
    inner: LazyBatchNorm1DWrapper,
}

#[pymethods]
impl PyLazyBatchNorm1d {
    #[new]
    #[pyo3(signature = (eps=1e-5, momentum=0.1, affine=true, track_running_stats=true, dtype="float32", device="cpu"))]
    fn new(
        eps: Option<f64>,
        momentum: Option<f64>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        // momentum in pytorch can be None
        let affine = affine.unwrap_or(true);
        let track_running_stats = track_running_stats.unwrap_or(true);
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let layer =
                    LazyBatchNorm1d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                        eps,
                        momentum,
                        affine,
                        track_running_stats,
                    );
                LazyBatchNorm1DWrapper::CpuF32(layer)
            }
            ("cpu", "float64") => {
                let layer =
                    LazyBatchNorm1d::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                        eps,
                        momentum,
                        affine,
                        track_running_stats,
                    );
                LazyBatchNorm1DWrapper::CpuF64(layer)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let layer =
                    LazyBatchNorm1d::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                        eps,
                        momentum,
                        affine,
                        track_running_stats,
                    );
                LazyBatchNorm1DWrapper::GpuF32(layer)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype: {}/{}",
                    device, dtype
                )))
            }
        };
        Ok(PyLazyBatchNorm1d { inner: wrapper })
    }

    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }

    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (LazyBatchNorm1DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (LazyBatchNorm1DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (LazyBatchNorm1DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Backend mismatch",
            )),
        }
    }
}

#[pyclass(name = "LazyBatchNorm2d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyLazyBatchNorm2d {
    inner: LazyBatchNorm2DWrapper,
}

#[pymethods]
impl PyLazyBatchNorm2d {
    #[new]
    #[pyo3(signature = (eps=1e-5, momentum=0.1, affine=true, track_running_stats=true, dtype="float32", device="cpu"))]
    fn new(
        eps: Option<f64>,
        momentum: Option<f64>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        // momentum in pytorch can be None
        let affine = affine.unwrap_or(true);
        let track_running_stats = track_running_stats.unwrap_or(true);
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let layer =
                    LazyBatchNorm2d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                        eps,
                        momentum,
                        affine,
                        track_running_stats,
                    );
                LazyBatchNorm2DWrapper::CpuF32(layer)
            }
            ("cpu", "float64") => {
                let layer =
                    LazyBatchNorm2d::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                        eps,
                        momentum,
                        affine,
                        track_running_stats,
                    );
                LazyBatchNorm2DWrapper::CpuF64(layer)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let layer =
                    LazyBatchNorm2d::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                        eps,
                        momentum,
                        affine,
                        track_running_stats,
                    );
                LazyBatchNorm2DWrapper::GpuF32(layer)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype: {}/{}",
                    device, dtype
                )))
            }
        };
        Ok(PyLazyBatchNorm2d { inner: wrapper })
    }
    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (LazyBatchNorm2DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (LazyBatchNorm2DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (LazyBatchNorm2DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Backend mismatch",
            )),
        }
    }
}

#[pyclass(name = "LazyBatchNorm3d", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyLazyBatchNorm3d {
    inner: LazyBatchNorm3DWrapper,
}

#[pymethods]
impl PyLazyBatchNorm3d {
    #[new]
    #[pyo3(signature = (eps=1e-5, momentum=0.1, affine=true, track_running_stats=true, dtype="float32", device="cpu"))]
    fn new(
        eps: Option<f64>,
        momentum: Option<f64>,
        affine: Option<bool>,
        track_running_stats: Option<bool>,
        dtype: Option<&str>,
        device: Option<&str>,
    ) -> PyResult<Self> {
        let eps = eps.unwrap_or(1e-5);
        // momentum in pytorch can be None
        let affine = affine.unwrap_or(true);
        let track_running_stats = track_running_stats.unwrap_or(true);
        let dtype = dtype.unwrap_or("float32");
        let device = device.unwrap_or("cpu");

        let wrapper = match (device, dtype) {
            ("cpu", "float32") => {
                let layer =
                    LazyBatchNorm3d::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                        eps,
                        momentum,
                        affine,
                        track_running_stats,
                    );
                LazyBatchNorm3DWrapper::CpuF32(layer)
            }
            ("cpu", "float64") => {
                let layer =
                    LazyBatchNorm3d::<CpuBackend<Float64>, DenseStorage<Float64>, Float64>::new(
                        eps,
                        momentum,
                        affine,
                        track_running_stats,
                    );
                LazyBatchNorm3DWrapper::CpuF64(layer)
            }
            #[cfg(feature = "gpu")]
            ("cuda", "float32") | ("gpu", "float32") => {
                let layer =
                    LazyBatchNorm3d::<GpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(
                        eps,
                        momentum,
                        affine,
                        track_running_stats,
                    );
                LazyBatchNorm3DWrapper::GpuF32(layer)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported device/dtype: {}/{}",
                    device, dtype
                )))
            }
        };
        Ok(PyLazyBatchNorm3d { inner: wrapper })
    }
    fn __call__(&self, input: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input)
    }
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        match (&self.inner, &input.inner) {
            (LazyBatchNorm3DWrapper::CpuF32(m), TensorWrapper::CpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF32(res),
                })
            }
            (LazyBatchNorm3DWrapper::CpuF64(m), TensorWrapper::CpuDenseF64(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::CpuDenseF64(res),
                })
            }
            #[cfg(feature = "gpu")]
            (LazyBatchNorm3DWrapper::GpuF32(m), TensorWrapper::GpuDenseF32(i)) => {
                let res = m.forward(i).map_err(to_py_err)?;
                Ok(PyTensor {
                    inner: TensorWrapper::GpuDenseF32(res),
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Backend mismatch",
            )),
        }
    }
}
