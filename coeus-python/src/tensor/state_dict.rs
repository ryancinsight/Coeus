// ── Python-exposed StateDict class wrapping weight/bias checkpoints ──

use pyo3::prelude::*;

/// Python-exposed StateDict class wrapping weight/bias checkpoints.
#[pyclass(name = "StateDict")]
pub struct PyStateDict {
    /// Underlying Rust StateDict holding named tensors.
    pub inner: coeus_tensor::checkpoint::StateDict<f64, coeus_core::MoiraiBackend>,
}

#[pymethods]
impl PyStateDict {
    #[new]
    fn new() -> Self {
        Self {
            inner: coeus_tensor::checkpoint::StateDict::new(),
        }
    }

    fn insert(&mut self, name: String, tensor: &super::PyTensor) {
        self.inner.insert(name, tensor.inner.tensor.clone());
    }

    fn get(&self, name: &str) -> Option<super::PyTensor> {
        self.inner.get(name).map(|t| super::PyTensor {
            inner: coeus_autograd::Var::new(t.clone(), false),
        })
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let mut file = std::fs::File::create(path)?;
        self.inner.save(&mut file)?;
        Ok(())
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let mut file = std::fs::File::open(path)?;
        let inner = coeus_tensor::checkpoint::StateDict::load(&mut file)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "StateDict(keys={:?})",
            self.inner.tensors.keys().collect::<Vec<_>>()
        )
    }
}
