// ── nn.Sequential ──
//
// A generic ordered container of nn.Module callables.  Equivalent to
// `torch.nn.Sequential(*modules)`.

use crate::tensor::PyTensor;
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// Python-exposed sequential container of nn modules.
///
/// Chains an ordered list of Python callables (any object with `.forward(x)`)
/// together, passing the output of each as the input to the next.
///
/// ```python
/// model = pycoeus.Sequential([
///     pycoeus.Linear(784, 256),
///     pycoeus.LayerNorm(256),
/// ])
/// out = model.forward(x)
/// ```
///
/// Each element must be a Python object that has a `forward(tensor) -> tensor`
/// method.
#[pyclass(name = "Sequential")]
pub struct PySequential {
    /// Ordered list of module-like objects.
    pub modules: Vec<PyObject>,
}

#[pymethods]
impl PySequential {
    /// Construct a Sequential from a list of modules.
    #[new]
    pub fn new(modules: Vec<PyObject>) -> Self {
        Self { modules }
    }

    /// Forward pass: chains `module.forward(x)` for each module in order.
    ///
    /// Each module's `forward` method is called with the output of the
    /// previous module (or the initial `input` for the first module).
    pub fn forward(&self, input: &PyTensor, py: Python<'_>) -> PyResult<PyTensor> {
        if self.modules.is_empty() {
            return Ok(input.clone());
        }
        // Convert the initial input to a PyObject for uniform chaining.
        let init_obj: Py<PyAny> = Py::new(py, input.clone())?.into_any();
        let mut current = init_obj;
        for module in &self.modules {
            let result = module.bind(py).call_method1("forward", (&current,))?;
            current = result.into();
        }
        let result = current.bind(py).extract::<PyTensor>()?;
        Ok(result)
    }

    /// Append a module to the end of the sequence.
    pub fn append(&mut self, module: PyObject) {
        self.modules.push(module);
    }

    /// Return the number of modules.
    fn __len__(&self) -> usize {
        self.modules.len()
    }

    /// Retrieve a module by index.
    fn __getitem__(&self, index: isize, py: Python<'_>) -> PyResult<PyObject> {
        let n = self.modules.len() as isize;
        let idx = if index < 0 { n + index } else { index };
        if idx < 0 || idx >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Sequential index {index} out of range for length {n}"
            )));
        }
        Ok(self.modules[idx as usize].clone_ref(py))
    }

    /// Collect all learnable parameters from all child modules.
    ///
    /// Each child must expose a `parameters()` method that returns a list of
    /// `Tensor` objects. Modules without `parameters()` are silently skipped.
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

    /// Zero the accumulated gradients of all parameters in all modules.
    pub fn zero_grad(&self, py: Python<'_>) {
        for module in &self.modules {
            let _ = module.bind(py).call_method0("zero_grad");
        }
    }
}
