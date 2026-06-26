use crate::tensor::{PyStateDict, PyTensor};
use pyo3::prelude::*;

/// Python-exposed Embedding layer.
#[pyclass(name = "Embedding")]
pub struct PyEmbedding {
    /// Learnable embedding weight table, shape `[num_embeddings, embedding_dim]`.
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Vocabulary size (number of distinct token indices).
    #[pyo3(get)]
    pub num_embeddings: usize,
    /// Dimensionality of each embedding vector.
    #[pyo3(get)]
    pub embedding_dim: usize,
    /// Token index whose embedding row is forced to all-zeros.
    #[pyo3(get)]
    pub padding_idx: Option<usize>,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    #[pyo3(signature = (num_embeddings, embedding_dim, padding_idx = None))]
    /// Create an Embedding layer with `num_embeddings` tokens of dimension `embedding_dim`.
    pub fn new(
        py: Python<'_>,
        num_embeddings: usize,
        embedding_dim: usize,
        padding_idx: Option<usize>,
    ) -> PyResult<Self> {
        let rust_emb = match padding_idx {
            Some(idx) => coeus_nn::Embedding::<f64, coeus_core::MoiraiBackend>::with_padding_idx(
                num_embeddings,
                embedding_dim,
                idx,
            ),
            None => coeus_nn::Embedding::<f64, coeus_core::MoiraiBackend>::new(
                num_embeddings,
                embedding_dim,
            ),
        };
        let weight = Py::new(
            py,
            PyTensor {
                inner: rust_emb.weight,
            },
        )?;
        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx,
        })
    }

    /// Forward pass through the Embedding layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let num_emb = self.num_embeddings;
        let emb_dim = self.embedding_dim;
        let padding_idx = self.padding_idx;

        let inner = py.allow_threads(move || {
            let emb = coeus_nn::Embedding {
                weight: w_var,
                num_embeddings: num_emb,
                embedding_dim: emb_dim,
                padding_idx,
            };
            emb.forward(&input_var)
        });
        Ok(PyTensor::from_var(inner))
    }

    fn state_dict(&self, py: Python<'_>) -> PyResult<PyStateDict> {
        let mut sd = coeus_tensor::checkpoint::StateDict::new();
        sd.insert("weight", self.weight.bind(py).borrow().inner.tensor.clone());
        Ok(PyStateDict { inner: sd })
    }

    fn load_state_dict(&self, state_dict: &PyStateDict, py: Python<'_>) -> PyResult<()> {
        if let Some(w) = state_dict.inner.get("weight") {
            self.weight.bind(py).borrow_mut().inner.tensor = w.clone();
        }
        Ok(())
    }

    /// Return the list of learnable parameters.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![self.weight.clone_ref(py)]
    }

    /// Zero the gradients of all parameters.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.weight.bind(py).borrow().zero_grad();
    }
}
