// ── Python wrappers: LSTMCell and GRUCell ──

use crate::{error::map_module_error, tensor::PyTensor};
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
            coeus_nn::rnn::LSTMCell::<f64, coeus_core::MoiraiBackend>::new(input_size, hidden_size)
                .map_err(crate::init::map_initialization_error)?;
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

        // Built before the GIL is released: constructing a cell can fail, and
        // a `PyErr` cannot be raised without the GIL. Every weight below is
        // overwritten by the tensors this wrapper already holds, so the draw
        // the constructor performs is discarded -- keeping it outside the
        // compute region at least keeps it off the timed path.
        let mut cell = coeus_nn::rnn::LSTMCell::<f64, coeus_core::MoiraiBackend>::new(
            w_ih.tensor.shape()[1],
            hs,
        )
        .map_err(crate::init::map_initialization_error)?;
        cell.w_ih.weight = w_ih;
        cell.w_ih.bias = b_ih;
        cell.w_hh.weight = w_hh;
        cell.w_hh.bias = b_hh;

        let (h_new, c_new) = py
            .allow_threads(move || cell.step(&x_v, &h_v, &c_v))
            .map_err(map_module_error)?;
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
    pub fn zero_grad(&self, py: Python<'_>) -> PyResult<()> {
        self.w_ih.bind(py).borrow().zero_grad()?;
        self.w_hh.bind(py).borrow().zero_grad()?;
        if let Some(ref b) = self.b_ih {
            b.bind(py).borrow().zero_grad()?;
        }
        if let Some(ref b) = self.b_hh {
            b.bind(py).borrow().zero_grad()?;
        }
        Ok(())
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
            coeus_nn::rnn::GRUCell::<f64, coeus_core::MoiraiBackend>::new(input_size, hidden_size)
                .map_err(crate::init::map_initialization_error)?;
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

        // Built before the GIL is released: constructing a cell can fail, and
        // a `PyErr` cannot be raised without the GIL. Every weight below is
        // overwritten by the tensors this wrapper already holds, so the draw
        // the constructor performs is discarded -- keeping it outside the
        // compute region at least keeps it off the timed path.
        let mut cell = coeus_nn::rnn::GRUCell::<f64, coeus_core::MoiraiBackend>::new(
            w_ih.tensor.shape()[1],
            hs,
        )
        .map_err(crate::init::map_initialization_error)?;
        cell.w_ih.weight = w_ih;
        cell.w_ih.bias = b_ih;
        cell.w_hh.weight = w_hh;
        cell.w_hh.bias = b_hh;

        let h_new = py.allow_threads(move || cell.step(&x_v, &h_v));
        h_new.map(PyTensor::from_var).map_err(map_module_error)
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
    pub fn zero_grad(&self, py: Python<'_>) -> PyResult<()> {
        self.w_ih.bind(py).borrow().zero_grad()?;
        self.w_hh.bind(py).borrow().zero_grad()?;
        if let Some(ref b) = self.b_ih {
            b.bind(py).borrow().zero_grad()?;
        }
        if let Some(ref b) = self.b_hh {
            b.bind(py).borrow().zero_grad()?;
        }
        Ok(())
    }
}

// ── Python wrapper: vanilla (Elman) RNNCell ──

/// Python-exposed vanilla RNN cell.
///
/// ```python
/// cell = pycoeus.RNNCell(input_size=8, hidden_size=16, nonlinearity="tanh")
/// h_new = cell.step(x, h)
/// ```
#[pyclass(name = "RNNCell")]
pub struct PyRNNCell {
    /// Dimensionality of the input vector.
    pub input_size: usize,
    /// Dimensionality of the hidden state.
    pub hidden_size: usize,
    /// Input-hidden weight matrix, shape `[hidden_size, input_size]`.
    #[pyo3(get)]
    pub w_ih: Py<PyTensor>,
    /// Optional input-hidden bias, shape `[hidden_size]`.
    #[pyo3(get)]
    pub b_ih: Option<Py<PyTensor>>,
    /// Hidden-hidden weight matrix, shape `[hidden_size, hidden_size]`.
    #[pyo3(get)]
    pub w_hh: Py<PyTensor>,
    /// Optional hidden-hidden bias, shape `[hidden_size]`.
    #[pyo3(get)]
    pub b_hh: Option<Py<PyTensor>>,
    nonlinearity: coeus_nn::rnn::RnnNonlinearity,
}

fn parse_nonlinearity(s: &str) -> PyResult<coeus_nn::rnn::RnnNonlinearity> {
    match s {
        "tanh" => Ok(coeus_nn::rnn::RnnNonlinearity::Tanh),
        "relu" => Ok(coeus_nn::rnn::RnnNonlinearity::Relu),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "nonlinearity must be 'tanh' or 'relu', got '{other}'"
        ))),
    }
}

#[pymethods]
impl PyRNNCell {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, nonlinearity = "tanh", bias = true))]
    /// Create an RNNCell with the given sizes and `tanh`/`relu` nonlinearity.
    pub fn new(
        py: Python<'_>,
        input_size: usize,
        hidden_size: usize,
        nonlinearity: &str,
        bias: bool,
    ) -> PyResult<Self> {
        let nl = parse_nonlinearity(nonlinearity)?;
        let cell = coeus_nn::rnn::RNNCell::<f64, coeus_core::MoiraiBackend>::new(
            input_size,
            hidden_size,
            nl,
        )
        .map_err(crate::init::map_initialization_error)?;
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
            nonlinearity: nl,
        })
    }

    /// Single-step forward: `(x, h) -> h_new`.
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
        let nl = self.nonlinearity;
        // Built before the GIL is released: constructing a cell can fail, and
        // a `PyErr` cannot be raised without the GIL. Every weight below is
        // overwritten by the tensors this wrapper already holds, so the draw
        // the constructor performs is discarded -- keeping it outside the
        // compute region at least keeps it off the timed path.
        let mut cell = coeus_nn::rnn::RNNCell::<f64, coeus_core::MoiraiBackend>::new(
            w_ih.tensor.shape()[1],
            hs,
            nl,
        )
        .map_err(crate::init::map_initialization_error)?;
        cell.w_ih.weight = w_ih;
        cell.w_ih.bias = b_ih;
        cell.w_hh.weight = w_hh;
        cell.w_hh.bias = b_hh;

        let h_new = py.allow_threads(move || cell.step(&x_v, &h_v));
        h_new.map(PyTensor::from_var).map_err(map_module_error)
    }

    /// Learnable parameters (w_ih, w_hh, b_ih, b_hh).
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
    pub fn zero_grad(&self, py: Python<'_>) -> PyResult<()> {
        self.w_ih.bind(py).borrow().zero_grad()?;
        self.w_hh.bind(py).borrow().zero_grad()?;
        if let Some(ref b) = self.b_ih {
            b.bind(py).borrow().zero_grad()?;
        }
        if let Some(ref b) = self.b_hh {
            b.bind(py).borrow().zero_grad()?;
        }
        Ok(())
    }
}

// ── PyBidirectional ──

use coeus_nn::Module as _;

/// Python-exposed Bidirectional RNN wrapper.
///
/// Wraps a sequence module and runs it forward + backward (time-reversed),
/// concatenating outputs to `[batch, seq, 2*hidden]`.
#[pyclass(name = "Bidirectional")]
pub struct PyBidirectional {
    /// Forward-direction LSTM.
    inner_fwd: coeus_nn::rnn::Lstm<f64, coeus_core::MoiraiBackend>,
    /// Backward-direction LSTM.
    inner_bwd: coeus_nn::rnn::Lstm<f64, coeus_core::MoiraiBackend>,
}

#[pymethods]
impl PyBidirectional {
    /// Create a Bidirectional LSTM with independent forward/backward weights.
    #[new]
    pub fn new(input_size: usize, hidden_size: usize) -> PyResult<Self> {
        Ok(Self {
            inner_fwd: coeus_nn::rnn::Lstm::<f64, coeus_core::MoiraiBackend>::new(
                input_size,
                hidden_size,
            )
            .map_err(crate::init::map_initialization_error)?,
            inner_bwd: coeus_nn::rnn::Lstm::<f64, coeus_core::MoiraiBackend>::new(
                input_size,
                hidden_size,
            )
            .map_err(crate::init::map_initialization_error)?,
        })
    }

    /// Forward: `x [N, T, D_in]` → `[N, T, 2*D_hidden]`.
    pub fn forward(
        &self,
        input: &crate::tensor::PyTensor,
        py: Python<'_>,
    ) -> PyResult<crate::tensor::PyTensor> {
        let input_var = input.inner.clone();
        let bi = coeus_nn::rnn::Bidirectional::new(self.inner_fwd.clone(), self.inner_bwd.clone());
        let out = py.allow_threads(move || bi.forward(&input_var));
        out.map(crate::tensor::PyTensor::from_var)
            .map_err(map_module_error)
    }
}

// ── Python wrapper: multi-step GRU ──

/// Python-exposed multi-step GRU (one layer, single direction).
///
/// The cell weights live here as Python tensors, so optimizers and `.data`
/// assignment drive them; every `forward` rebuilds the cell around them and
/// unrolls over the sequence dimension.
///
/// ```python
/// gru = pycoeus.Gru(input_size=4, hidden_size=8)
/// y = gru.forward(x)          # [batch, seq, input_size] -> [batch, seq, 8]
/// y, h_n = gru.forward_seq(x)
/// ```
#[pyclass(name = "Gru")]
pub struct PyGru {
    /// Dimensionality of each input vector.
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

impl PyGru {
    /// Rebuild the cell over the Python-owned weights and run `op` on it.
    ///
    /// The cell is constructed before the GIL is released (construction can
    /// fail, and a `PyErr` needs the GIL) and every weight is immediately
    /// overwritten with the stored tensors, so the draw is never used.
    fn with_cell<R>(
        &self,
        py: Python<'_>,
        op: impl for<'a> FnOnce(&coeus_nn::rnn::Gru<f64, coeus_core::MoiraiBackend>) -> R + Send,
    ) -> PyResult<R>
    where
        R: Send,
    {
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

        let mut cell = coeus_nn::rnn::GRUCell::<f64, coeus_core::MoiraiBackend>::new(
            self.input_size,
            self.hidden_size,
        )
        .map_err(crate::init::map_initialization_error)?;
        cell.w_ih.weight = w_ih;
        cell.w_ih.bias = b_ih;
        cell.w_hh.weight = w_hh;
        cell.w_hh.bias = b_hh;
        let gru = coeus_nn::rnn::Gru::from_cell(cell);
        Ok(py.allow_threads(move || op(&gru)))
    }
}

#[pymethods]
impl PyGru {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias = true))]
    /// Create a single-layer GRU.
    pub fn new(
        py: Python<'_>,
        input_size: usize,
        hidden_size: usize,
        bias: bool,
    ) -> PyResult<Self> {
        // A single-layer `Gru` is its cell, so the weights come straight off a
        // freshly initialized cell — the same path `GRUCell` uses.
        let cell =
            coeus_nn::rnn::GRUCell::<f64, coeus_core::MoiraiBackend>::new(input_size, hidden_size)
                .map_err(crate::init::map_initialization_error)?;
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

    /// Unroll over the sequence: `[batch, seq, input_size]` → `[batch, seq, hidden_size]`.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let x = input.inner.clone();
        self.with_cell(py, move |gru| gru.forward(&x))?
            .map(PyTensor::from_var)
            .map_err(map_module_error)
    }

    /// Unroll over the sequence, returning `(output, h_n)`.
    pub fn forward_seq(&self, input: &PyTensor, py: Python<'_>) -> PyResult<(PyTensor, PyTensor)> {
        let x = input.inner.clone();
        let states = self.with_cell(py, move |gru| gru.forward_seq(&x))?;
        let (output, h_n) = states.map_err(map_module_error)?;
        Ok((PyTensor::from_var(output), PyTensor::from_var(h_n)))
    }

    /// Return the learnable parameters (w_ih, w_hh, b_ih, b_hh).
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
    pub fn zero_grad(&self, py: Python<'_>) -> PyResult<()> {
        self.w_ih.bind(py).borrow().zero_grad()?;
        self.w_hh.bind(py).borrow().zero_grad()?;
        if let Some(ref b) = self.b_ih {
            b.bind(py).borrow().zero_grad()?;
        }
        if let Some(ref b) = self.b_hh {
            b.bind(py).borrow().zero_grad()?;
        }
        Ok(())
    }

    /// Weights keyed as in torch's `nn.GRU` (`weight_ih_l0`, ...).
    fn state_dict(&self, py: Python<'_>) -> PyResult<crate::tensor::PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert(
            "weight_ih_l0",
            self.w_ih.bind(py).borrow().inner.tensor.clone(),
        );
        sd.insert(
            "weight_hh_l0",
            self.w_hh.bind(py).borrow().inner.tensor.clone(),
        );
        if let Some(ref b) = self.b_ih {
            sd.insert("bias_ih_l0", b.bind(py).borrow().inner.tensor.clone());
        }
        if let Some(ref b) = self.b_hh {
            sd.insert("bias_hh_l0", b.bind(py).borrow().inner.tensor.clone());
        }
        Ok(crate::tensor::PyStateDict { inner: sd })
    }
}
