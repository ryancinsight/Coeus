use pyo3::prelude::*;
use pyo3::pyclass;

/// Encoding result
#[pyclass(name = "Encoding", module = "_coeus")]
pub struct Encoding {
    pub ids: Vec<usize>,
    pub attention_mask: Vec<usize>,
}

#[pymethods]
impl Encoding {
    #[getter]
    fn ids(&self) -> Vec<usize> {
        self.ids.clone()
    }

    #[getter]
    fn attention_mask(&self) -> Vec<usize> {
        self.attention_mask.clone()
    }
}

/// BPE Tokenizer (placeholder)
#[pyclass(name = "BpeTokenizer", module = "_coeus")]
pub struct BpeTokenizer;

#[pymethods]
impl BpeTokenizer {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "BPE tokenizer not yet implemented",
        ))
    }
}

/// GPT-2 Tokenizer (placeholder)
#[pyclass(name = "GPT2Tokenizer", module = "_coeus")]
pub struct GPT2Tokenizer;

#[pymethods]
impl GPT2Tokenizer {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "GPT-2 tokenizer not yet implemented",
        ))
    }
}

/// CLIP Tokenizer (placeholder)
#[pyclass(name = "CLIPTokenizer", module = "_coeus")]
pub struct CLIPTokenizer;

#[pymethods]
impl CLIPTokenizer {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "CLIP tokenizer not yet implemented",
        ))
    }
}

/// BERT Tokenizer (placeholder)
#[pyclass(name = "BERTTokenizer", module = "_coeus")]
pub struct BERTTokenizer;

#[pymethods]
impl BERTTokenizer {
    #[new]
    fn new() -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "BERT tokenizer not yet implemented",
        ))
    }
}
