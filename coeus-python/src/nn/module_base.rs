// ── pycoeus.Module — abstract base for user-defined modules ──
//
// Provides a base class that user Python code can inherit to build custom
// neural-network modules that are compatible with Sequential, ModuleList,
// and the parameter/gradient utilities.
//
// Usage:
// ```python
// class MyMLP(pycoeus.Module):
//     def __init__(self):
//         self.fc1 = pycoeus.Linear(4, 8)
//         self.fc2 = pycoeus.Linear(8, 2)
//
//     def forward(self, x):
//         x = pycoeus.relu(self.fc1.forward(x))
//         return self.fc2.forward(x)
//
// model = MyMLP()
// out = model.forward(inp)
// ```

use crate::tensor::PyTensor;
use pyo3::prelude::*;

/// Abstract base class for user-defined pycoeus modules.
///
/// Subclass this in Python and override `forward(self, x)`.
///
/// Built-in helper methods:
/// - `parameters()` — returns an empty list by default; subclasses should
///   override to return their learnable tensors.
/// - `zero_grad()` — iterates over `self.parameters()` and zeros each gradient.
/// - `train(mode=True)` / `eval()` — training-mode flag stub (no-op; hook for
///   Dropout / BatchNorm subclasses).
#[pyclass(subclass, name = "Module")]
pub struct PyModule {
    pub training: bool,
}

#[pymethods]
impl PyModule {
    #[new]
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Override in subclasses to compute the module output from `x`.
    ///
    /// The default raises `NotImplementedError`.
    pub fn forward(&self, _x: &PyTensor) -> PyResult<PyTensor> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Module.forward() must be overridden in subclasses",
        ))
    }

    /// Return the learnable parameters of this module.
    ///
    /// Override in subclasses to return `[param1, param2, ...]`.
    pub fn parameters(&self, _py: Python<'_>) -> Vec<Py<PyTensor>> {
        vec![]
    }

    /// Zero the gradients of all parameters returned by `parameters()`.
    pub fn zero_grad(&self, py: Python<'_>) {
        for p in self.parameters(py) {
            p.bind(py).borrow().zero_grad();
        }
    }

    /// Set the module to training mode.
    #[pyo3(signature = (mode = true))]
    pub fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    /// Set the module to evaluation mode.
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Whether the module is in training mode.
    #[getter]
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        "Module()".to_string()
    }
}

impl Default for PyModule {
    fn default() -> Self {
        Self::new()
    }
}
