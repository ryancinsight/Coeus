use crate::tensor::{PyStateDict, PyTensor};
use pyo3::exceptions::PyValueError;
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

/// Python-exposed EmbeddingBag layer.
#[pyclass(name = "EmbeddingBag")]
pub struct PyEmbeddingBag {
    /// Learnable embedding table, shape `[num_embeddings, embedding_dim]`.
    #[pyo3(get)]
    pub weight: Py<PyTensor>,
    /// Vocabulary size.
    #[pyo3(get)]
    pub num_embeddings: usize,
    /// Embedding dimension.
    #[pyo3(get)]
    pub embedding_dim: usize,
    /// Aggregation mode: `"sum"`, `"mean"`, or `"max"`.
    #[pyo3(get)]
    pub mode: String,
}

fn tensor_to_indices(name: &str, tensor: &PyTensor) -> PyResult<Vec<usize>> {
    let backend = coeus_core::MoiraiBackend::new();
    let contiguous = tensor.inner.tensor.to_contiguous_on(&backend);
    contiguous
        .as_slice()
        .iter()
        .enumerate()
        .map(|(position, &value)| {
            if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
                return Err(PyValueError::new_err(format!(
                    "EmbeddingBag.forward: {name}[{position}] must be a non-negative integer, got {value}"
                )));
            }
            Ok(value as usize)
        })
        .collect()
}

fn no_offset_bag_starts(indices: &PyTensor) -> PyResult<Option<Vec<usize>>> {
    let shape = indices.inner.tensor.shape();
    match shape {
        [_] => Ok(None),
        [num_bags, bag_size] => Ok(Some((0..*num_bags).map(|bag| bag * *bag_size).collect())),
        _ => Err(PyValueError::new_err(format!(
            "EmbeddingBag.forward: expected 1-D input with offsets or 2-D input without offsets, got {}-D input",
            shape.len()
        ))),
    }
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

#[pymethods]
impl PyEmbeddingBag {
    #[new]
    #[pyo3(signature = (num_embeddings, embedding_dim, mode = "sum"))]
    /// Create an EmbeddingBag layer with aggregation `mode` in {"sum","mean","max"}.
    pub fn new(
        py: Python<'_>,
        num_embeddings: usize,
        embedding_dim: usize,
        mode: &str,
    ) -> PyResult<Self> {
        let Some(parsed_mode) = coeus_nn::EmbeddingBagMode::parse(mode) else {
            return Err(PyValueError::new_err(
                "EmbeddingBag mode must be one of: sum, mean, max",
            ));
        };
        let rust_emb = coeus_nn::EmbeddingBag::<f64, coeus_core::MoiraiBackend>::new(
            num_embeddings,
            embedding_dim,
            parsed_mode,
        );
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
            mode: mode.to_string(),
        })
    }

    /// Forward pass from index and optional offset tensors (torch
    /// `EmbeddingBag(input, offsets)` API). The integer indices are carried as
    /// the elements of a float tensor; they are read back and truncated to
    /// `usize`, then routed through the same tracked core as
    /// [`Self::forward_with_offsets`].
    #[pyo3(signature = (indices, offsets = None))]
    pub fn forward(
        &self,
        indices: &PyTensor,
        offsets: Option<&PyTensor>,
        py: Python<'_>,
    ) -> PyResult<PyTensor> {
        let idx = tensor_to_indices("indices", indices)?;
        let off = match offsets {
            Some(offset_tensor) => Some(tensor_to_indices("offsets", offset_tensor)?),
            None => no_offset_bag_starts(indices)?,
        };
        self.forward_with_offsets(idx, off, py)
    }

    /// Forward pass from flat indices and optional bag offsets.
    #[pyo3(signature = (indices, offsets = None))]
    pub fn forward_with_offsets(
        &self,
        indices: Vec<usize>,
        offsets: Option<Vec<usize>>,
        py: Python<'_>,
    ) -> PyResult<PyTensor> {
        let Some(mode) = coeus_nn::EmbeddingBagMode::parse(&self.mode) else {
            return Err(PyValueError::new_err(
                "EmbeddingBag mode must be one of: sum, mean, max",
            ));
        };
        let w_var = self.weight.bind(py).borrow().inner.clone();
        let num_emb = self.num_embeddings;
        let emb_dim = self.embedding_dim;
        let inner = py.allow_threads(move || {
            let emb = coeus_nn::EmbeddingBag {
                weight: w_var,
                num_embeddings: num_emb,
                embedding_dim: emb_dim,
                mode,
            };
            emb.forward_with_offsets(&indices, offsets.as_deref())
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

    /// Return learnable parameters.
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![self.weight.clone_ref(py)]
    }

    /// Zero parameter gradients.
    pub fn zero_grad(&self, py: Python<'_>) {
        self.weight.bind(py).borrow().zero_grad();
    }
}
