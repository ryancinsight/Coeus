use crate::tensor::{PyTensor, TensorWrapper};
use coeus_nn::modules::distance::{PairwiseDistance, CosineSimilarity};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyErr, PyResult};
use coeus_nn::Module;

fn to_py_err(e: impl std::fmt::Display) -> PyErr {
    crate::error::convert_error(format!("layer: Distance error: {}", e))
}

#[pyclass(name = "PairwiseDistance", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyPairwiseDistance {
    pub p: f64,
}

#[pymethods]
impl PyPairwiseDistance {
    #[new]
    #[pyo3(signature = (p=2.0))]
    fn new(p: f64) -> Self {
        PyPairwiseDistance { p }
    }

    fn __call__(&self, input1: &PyTensor, input2: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input1, input2)
    }

    fn forward(&self, input1: &PyTensor, input2: &PyTensor) -> PyResult<PyTensor> {
        match (&input1.inner, &input2.inner) {
            (TensorWrapper::CpuDenseF32(p1), TensorWrapper::CpuDenseF32(p2)) => {
                let res = PairwiseDistance::new(self.p, 1e-6, false).forward(&(p1.clone(), p2.clone())).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (TensorWrapper::CpuDenseF64(p1), TensorWrapper::CpuDenseF64(p2)) => {
                let res = PairwiseDistance::new(self.p, 1e-6, false).forward(&(p1.clone(), p2.clone())).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Input tensors mismatch")),
        }
    }
}

#[pyclass(name = "CosineSimilarity", module = "coeus.nn", subclass, unsendable)]
#[derive(Clone)]
pub struct PyCosineSimilarity {
    pub dim: usize,
    pub eps: f64,
}

#[pymethods]
impl PyCosineSimilarity {
    #[new]
    #[pyo3(signature = (dim=1, eps=1e-8))]
    fn new(dim: usize, eps: f64) -> Self {
        PyCosineSimilarity { dim, eps }
    }

    fn __call__(&self, input1: &PyTensor, input2: &PyTensor) -> PyResult<PyTensor> {
        self.forward(input1, input2)
    }

    fn forward(&self, input1: &PyTensor, input2: &PyTensor) -> PyResult<PyTensor> {
        match (&input1.inner, &input2.inner) {
            (TensorWrapper::CpuDenseF32(p1), TensorWrapper::CpuDenseF32(p2)) => {
                let res = CosineSimilarity::new(self.dim, self.eps).forward(&(p1.clone(), p2.clone())).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF32(res) })
            }
            (TensorWrapper::CpuDenseF64(p1), TensorWrapper::CpuDenseF64(p2)) => {
                let res = CosineSimilarity::new(self.dim, self.eps).forward(&(p1.clone(), p2.clone())).map_err(to_py_err)?;
                Ok(PyTensor { inner: TensorWrapper::CpuDenseF64(res) })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Input tensors mismatch")),
        }
    }
}

pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPairwiseDistance>()?;
    m.add_class::<PyCosineSimilarity>()?;
    let dict = m.dict();
    dict.set_item("PairwiseDistance", m.getattr("PairwiseDistance")?)?;
    dict.set_item("CosineSimilarity", m.getattr("CosineSimilarity")?)?;
    Ok(())
}
