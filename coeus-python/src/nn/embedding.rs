use pyo3::prelude::*;
use crate::tensor::{PyTensor, PyStateDict};

/// Python-exposed Embedding layer.
#[pyclass(name = "Embedding")]
pub struct PyEmbedding {
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    #[pyo3(get)]
    pub num_embeddings: usize,
    #[pyo3(get)]
    pub embedding_dim: usize,
}

#[pymethods]
impl PyEmbedding {
    #[new]
    pub fn new(py: Python<'_>, num_embeddings: usize, embedding_dim: usize) -> PyResult<Self> {
        let rust_emb = coeus_nn::Embedding::<f64, coeus_core::MoiraiBackend>::new(num_embeddings, embedding_dim);
        let weight = Py::new(py, PyTensor { inner: rust_emb.weight })?;
        Ok(Self {
            weight,
            num_embeddings,
            embedding_dim,
        })
    }

    /// Forward pass through the Embedding layer.
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        use coeus_nn::Module;
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let input_var = input.inner.clone();
        let num_emb = self.num_embeddings;
        let emb_dim = self.embedding_dim;

        let inner = py.allow_threads(move || {
            let emb = coeus_nn::Embedding {
                weight: w_var,
                num_embeddings: num_emb,
                embedding_dim: emb_dim,
            };
            emb.forward(&input_var)
        });
        Ok(PyTensor { inner })
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
}
