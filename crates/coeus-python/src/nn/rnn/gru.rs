//! Python-exposed multi-step GRU (one layer, single direction).

use crate::error::map_module_error;
use crate::tensor::PyTensor;
use pyo3::prelude::*;

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
