// ── Python wrappers: LSTMCell and GRUCell ──

use crate::tensor::PyTensor;
use pyo3::prelude::*;

/// Python-exposed LSTM cell.
///
/// ```python
/// cell = pycoeus.LSTMCell(input_size=8, hidden_size=16)
/// h_new, c_new = cell.step(x, h, c)
/// ```
///
/// All tensors: `[batch, size]`.
#[pyclass(name = "LSTMCell")]
pub struct PyLSTMCell {
    pub input_size: usize,
    pub hidden_size: usize,
    #[pyo3(get)]
    pub w_ih: Py<PyTensor>,
    #[pyo3(get)]
    pub w_hh: Py<PyTensor>,
}

#[pymethods]
impl PyLSTMCell {
    #[new]
    pub fn new(py: Python<'_>, input_size: usize, hidden_size: usize) -> PyResult<Self> {
        let cell =
            coeus_nn::rnn::LSTMCell::<f64, coeus_core::MoiraiBackend>::new(input_size, hidden_size);
        let w_ih = Py::new(
            py,
            PyTensor {
                inner: cell.w_ih.weight,
            },
        )?;
        let w_hh = Py::new(
            py,
            PyTensor {
                inner: cell.w_hh.weight,
            },
        )?;
        Ok(Self {
            input_size,
            hidden_size,
            w_ih,
            w_hh,
        })
    }

    /// Single-step forward: `(x, h, c) → (h_new, c_new)`.
    pub fn step(
        &self,
        x: &PyTensor,
        h: &PyTensor,
        c: &PyTensor,
        py: Python<'_>,
    ) -> PyResult<(PyTensor, PyTensor)> {
        let w_ih_var = self.w_ih.bind(py).borrow().inner.clone();
        let w_hh_var = self.w_hh.bind(py).borrow().inner.clone();
        let x_v = x.inner.clone();
        let h_v = h.inner.clone();
        let c_v = c.inner.clone();
        let hs = self.hidden_size;

        let (h_new, c_new) = py.allow_threads(move || {
            let mut cell = coeus_nn::rnn::LSTMCell::<f64, coeus_core::MoiraiBackend>::new(
                w_ih_var.tensor.shape()[1],
                hs,
            );
            cell.w_ih.weight = w_ih_var;
            cell.w_hh.weight = w_hh_var;
            cell.step(&x_v, &h_v, &c_v)
        });
        Ok((PyTensor::from_var(h_new), PyTensor::from_var(c_new)))
    }

    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![self.w_ih.clone_ref(py), self.w_hh.clone_ref(py)]
    }

    pub fn zero_grad(&self, py: Python<'_>) {
        self.w_ih.bind(py).borrow().zero_grad();
        self.w_hh.bind(py).borrow().zero_grad();
    }
}

/// Python-exposed GRU cell.
///
/// ```python
/// cell = pycoeus.GRUCell(input_size=8, hidden_size=16)
/// h_new = cell.step(x, h)
/// ```
#[pyclass(name = "GRUCell")]
pub struct PyGRUCell {
    pub input_size: usize,
    pub hidden_size: usize,
    #[pyo3(get)]
    pub w_ih: Py<PyTensor>,
    #[pyo3(get)]
    pub w_hh: Py<PyTensor>,
}

#[pymethods]
impl PyGRUCell {
    #[new]
    pub fn new(py: Python<'_>, input_size: usize, hidden_size: usize) -> PyResult<Self> {
        let cell =
            coeus_nn::rnn::GRUCell::<f64, coeus_core::MoiraiBackend>::new(input_size, hidden_size);
        let w_ih = Py::new(
            py,
            PyTensor {
                inner: cell.w_ih.weight,
            },
        )?;
        let w_hh = Py::new(
            py,
            PyTensor {
                inner: cell.w_hh.weight,
            },
        )?;
        Ok(Self {
            input_size,
            hidden_size,
            w_ih,
            w_hh,
        })
    }

    /// Single-step forward: `(x, h) → h_new`.
    pub fn step(&self, x: &PyTensor, h: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let w_ih_var = self.w_ih.bind(py).borrow().inner.clone();
        let w_hh_var = self.w_hh.bind(py).borrow().inner.clone();
        let x_v = x.inner.clone();
        let h_v = h.inner.clone();
        let hs = self.hidden_size;

        let h_new = py.allow_threads(move || {
            let mut cell = coeus_nn::rnn::GRUCell::<f64, coeus_core::MoiraiBackend>::new(
                w_ih_var.tensor.shape()[1],
                hs,
            );
            cell.w_ih.weight = w_ih_var;
            cell.w_hh.weight = w_hh_var;
            cell.step(&x_v, &h_v)
        });
        Ok(PyTensor::from_var(h_new))
    }

    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![self.w_ih.clone_ref(py), self.w_hh.clone_ref(py)]
    }

    pub fn zero_grad(&self, py: Python<'_>) {
        self.w_ih.bind(py).borrow().zero_grad();
        self.w_hh.bind(py).borrow().zero_grad();
    }
}
