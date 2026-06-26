// ── Python wrappers: LSTMCell and GRUCell ──

use crate::tensor::PyTensor;
use pyo3::prelude::*;

/// Python-exposed LSTM cell.
///
/// ```python
/// cell = pycoeus.LSTMCell(input_size=8, hidden_size=16)
/// h_new, c_new = cell.step(x, h, c)
/// ```
#[pyclass(name = "LSTMCell")]
pub struct PyLSTMCell {
    /// Dimensionality of the input vector.
    pub input_size: usize,
    /// Dimensionality of the hidden state.
    pub hidden_size: usize,
    /// Input-hidden weight matrix, shape `[4*hidden_size, input_size]`.
    #[pyo3(get)]
    pub w_ih: Py<PyTensor>,
    /// Optional input-hidden bias, shape `[4*hidden_size]`.
    #[pyo3(get)]
    pub b_ih: Option<Py<PyTensor>>,
    /// Hidden-hidden weight matrix, shape `[4*hidden_size, hidden_size]`.
    #[pyo3(get)]
    pub w_hh: Py<PyTensor>,
    /// Optional hidden-hidden bias, shape `[4*hidden_size]`.
    #[pyo3(get)]
    pub b_hh: Option<Py<PyTensor>>,
}

#[pymethods]
impl PyLSTMCell {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias = true))]
    /// Create an LSTMCell with given input and hidden sizes.
    pub fn new(
        py: Python<'_>,
        input_size: usize,
        hidden_size: usize,
        bias: bool,
    ) -> PyResult<Self> {
        let cell =
            coeus_nn::rnn::LSTMCell::<f64, coeus_core::MoiraiBackend>::new(input_size, hidden_size);
        let w_ih = Py::new(
            py,
            PyTensor {
                inner: cell.w_ih.weight,
            },
        )?;
        let b_ih = if bias {
            cell.w_ih
                .bias
                .map(|b| Py::new(py, PyTensor { inner: b }))
                .transpose()?
        } else {
            None
        };
        let w_hh = Py::new(
            py,
            PyTensor {
                inner: cell.w_hh.weight,
            },
        )?;
        let b_hh = if bias {
            cell.w_hh
                .bias
                .map(|b| Py::new(py, PyTensor { inner: b }))
                .transpose()?
        } else {
            None
        };
        Ok(Self {
            input_size,
            hidden_size,
            w_ih,
            b_ih,
            w_hh,
            b_hh,
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
        let w_ih = self.w_ih.bind(py).borrow().inner.clone();
        let b_ih = self
            .b_ih
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let w_hh = self.w_hh.bind(py).borrow().inner.clone();
        let b_hh = self
            .b_hh
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let x_v = x.inner.clone();
        let h_v = h.inner.clone();
        let c_v = c.inner.clone();
        let hs = self.hidden_size;

        let (h_new, c_new) = py.allow_threads(move || {
            let mut cell = coeus_nn::rnn::LSTMCell::<f64, coeus_core::MoiraiBackend>::new(
                w_ih.tensor.shape()[1],
                hs,
            );
            cell.w_ih.weight = w_ih;
            cell.w_ih.bias = b_ih;
            cell.w_hh.weight = w_hh;
            cell.w_hh.bias = b_hh;
            cell.step(&x_v, &h_v, &c_v)
        });
        Ok((PyTensor::from_var(h_new), PyTensor::from_var(c_new)))
    }

    /// Return the list of learnable parameters (w_ih, w_hh, b_ih, b_hh).
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        let mut p = vec![self.w_ih.clone_ref(py), self.w_hh.clone_ref(py)];
        if let Some(ref b) = self.b_ih {
            p.push(b.clone_ref(py));
        }
        if let Some(ref b) = self.b_hh {
            p.push(b.clone_ref(py));
        }
        p
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.w_ih.bind(py).borrow().zero_grad();
        self.w_hh.bind(py).borrow().zero_grad();
        if let Some(ref b) = self.b_ih {
            b.bind(py).borrow().zero_grad();
        }
        if let Some(ref b) = self.b_hh {
            b.bind(py).borrow().zero_grad();
        }
    }
}

/// Python-exposed GRU cell.
#[pyclass(name = "GRUCell")]
pub struct PyGRUCell {
    /// Dimensionality of the input vector.
    pub input_size: usize,
    /// Dimensionality of the hidden state.
    pub hidden_size: usize,
    /// Input-hidden weight matrix, shape `[3*hidden_size, input_size]`.
    #[pyo3(get)]
    pub w_ih: Py<PyTensor>,
    /// Optional input-hidden bias, shape `[3*hidden_size]`.
    #[pyo3(get)]
    pub b_ih: Option<Py<PyTensor>>,
    /// Hidden-hidden weight matrix, shape `[3*hidden_size, hidden_size]`.
    #[pyo3(get)]
    pub w_hh: Py<PyTensor>,
    /// Optional hidden-hidden bias, shape `[3*hidden_size]`.
    #[pyo3(get)]
    pub b_hh: Option<Py<PyTensor>>,
}

#[pymethods]
impl PyGRUCell {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias = true))]
    /// Create a GRUCell with given input and hidden sizes.
    pub fn new(
        py: Python<'_>,
        input_size: usize,
        hidden_size: usize,
        bias: bool,
    ) -> PyResult<Self> {
        let cell =
            coeus_nn::rnn::GRUCell::<f64, coeus_core::MoiraiBackend>::new(input_size, hidden_size);
        let w_ih = Py::new(
            py,
            PyTensor {
                inner: cell.w_ih.weight,
            },
        )?;
        let b_ih = if bias {
            cell.w_ih
                .bias
                .map(|b| Py::new(py, PyTensor { inner: b }))
                .transpose()?
        } else {
            None
        };
        let w_hh = Py::new(
            py,
            PyTensor {
                inner: cell.w_hh.weight,
            },
        )?;
        let b_hh = if bias {
            cell.w_hh
                .bias
                .map(|b| Py::new(py, PyTensor { inner: b }))
                .transpose()?
        } else {
            None
        };
        Ok(Self {
            input_size,
            hidden_size,
            w_ih,
            b_ih,
            w_hh,
            b_hh,
        })
    }

    /// Single-step forward: `(x, h) → h_new`.
    pub fn step(&self, x: &PyTensor, h: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        let w_ih = self.w_ih.bind(py).borrow().inner.clone();
        let b_ih = self
            .b_ih
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let w_hh = self.w_hh.bind(py).borrow().inner.clone();
        let b_hh = self
            .b_hh
            .as_ref()
            .map(|b| b.bind(py).borrow().inner.clone());
        let x_v = x.inner.clone();
        let h_v = h.inner.clone();
        let hs = self.hidden_size;

        let h_new = py.allow_threads(move || {
            let mut cell = coeus_nn::rnn::GRUCell::<f64, coeus_core::MoiraiBackend>::new(
                w_ih.tensor.shape()[1],
                hs,
            );
            cell.w_ih.weight = w_ih;
            cell.w_ih.bias = b_ih;
            cell.w_hh.weight = w_hh;
            cell.w_hh.bias = b_hh;
            cell.step(&x_v, &h_v)
        });
        Ok(PyTensor::from_var(h_new))
    }

    /// Return the list of learnable parameters (w_ih, w_hh, b_ih, b_hh).
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        let mut p = vec![self.w_ih.clone_ref(py), self.w_hh.clone_ref(py)];
        if let Some(ref b) = self.b_ih {
            p.push(b.clone_ref(py));
        }
        if let Some(ref b) = self.b_hh {
            p.push(b.clone_ref(py));
        }
        p
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.w_ih.bind(py).borrow().zero_grad();
        self.w_hh.bind(py).borrow().zero_grad();
        if let Some(ref b) = self.b_ih {
            b.bind(py).borrow().zero_grad();
        }
        if let Some(ref b) = self.b_hh {
            b.bind(py).borrow().zero_grad();
        }
    }
}
