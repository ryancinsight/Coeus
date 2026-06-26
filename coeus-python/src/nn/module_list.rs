// -- nn.ModuleList --

use crate::tensor::PyTensor;
use pyo3::prelude::*;

/// Dynamic ordered container of nn modules.
///
/// Unlike `Sequential`, `ModuleList` does NOT chain forwards automatically.
/// Callers index modules and call `.forward()` explicitly.
#[pyclass(name = "ModuleList")]
pub struct PyModuleList {
    /// Ordered list of contained child modules.
    pub modules: Vec<PyObject>,
}

#[pymethods]
impl PyModuleList {
    #[new]
    #[pyo3(signature = (modules = None))]
    /// Create a ModuleList, optionally pre-populated with `modules`.
    pub fn new(modules: Option<Vec<PyObject>>) -> Self {
        Self {
            modules: modules.unwrap_or_default(),
        }
    }

    /// Append a single module to the end of the list.
    pub fn append(&mut self, module: PyObject) {
        self.modules.push(module);
    }

    /// Extend the list with an iterable of modules.
    pub fn extend(&mut self, modules: Vec<PyObject>) {
        self.modules.extend(modules);
    }

    fn __len__(&self) -> usize {
        self.modules.len()
    }

    fn __getitem__(&self, index: isize, py: Python<'_>) -> PyResult<PyObject> {
        let n = self.modules.len() as isize;
        let idx = if index < 0 { n + index } else { index };
        if idx < 0 || idx >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "ModuleList index {index} out of range for length {n}"
            )));
        }
        Ok(self.modules[idx as usize].clone_ref(py))
    }

    fn __setitem__(&mut self, index: isize, module: PyObject) -> PyResult<()> {
        let n = self.modules.len() as isize;
        let idx = if index < 0 { n + index } else { index };
        if idx < 0 || idx >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "ModuleList index {index} out of range for length {n}"
            )));
        }
        self.modules[idx as usize] = module;
        Ok(())
    }

    /// Collect parameters from all child modules (modules with `.parameters()`).
    pub fn parameters(&self, py: Python<'_>) -> Vec<Py<PyTensor>> {
        let mut params: Vec<Py<PyTensor>> = Vec::new();
        for module in &self.modules {
            if let Ok(ps) = module.bind(py).call_method0("parameters") {
                if let Ok(list) = ps.extract::<Vec<Py<PyTensor>>>() {
                    params.extend(list);
                }
            }
        }
        params
    }

    /// Zero gradients in all child modules.
    pub fn zero_grad(&self, py: Python<'_>) {
        for module in &self.modules {
            let _ = module.bind(py).call_method0("zero_grad");
        }
    }
}
