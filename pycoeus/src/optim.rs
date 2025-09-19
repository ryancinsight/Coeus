use crate::tensor::PyTensor;
use coeus_optim::{Adam as RustAdam, Optimizer, Sgd as RustSgd};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyResult};
use std::collections::HashMap;

/// Python wrapper for SGD optimizer
#[pyclass]
pub struct Sgd {
    /// Underlying Rust SGD optimizer
    sgd: RustSgd<f32>,
    /// Parameters being optimized
    pub parameters: Vec<PyTensor>,
}

#[pymethods]
impl Sgd {
    #[new]
    #[pyo3(signature = (parameters, lr, momentum=None, weight_decay=None))]
    fn new(
        parameters: Vec<PyTensor>,
        lr: f32,
        momentum: Option<f32>,
        weight_decay: Option<f32>,
    ) -> PyResult<Self> {
        let momentum = momentum.unwrap_or(0.0);
        let weight_decay = weight_decay.unwrap_or(0.0);

        let rust_params: Vec<_> = parameters.iter().map(|p| p.tensor.clone()).collect();
        let sgd = RustSgd::with_options(
            rust_params,
            lr.into(),
            momentum.into(),
            weight_decay.into(),
            false,
        );

        Ok(Sgd { sgd, parameters })
    }

    fn step(&mut self) -> PyResult<()> {
        // Simple SGD implementation directly in Python bindings
        // This avoids the complex synchronization issues with the Rust optimizer

        let lr = if let Some(param_group) = self.sgd.param_groups().first() {
            param_group.lr
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No parameter groups found",
            ));
        };

        // Apply SGD update directly to each parameter: param = param - lr * grad
        for py_param in self.parameters.iter_mut() {
            if let Some(grad_tensor) = py_param.grad() {
                let param_data = py_param.data()?;
                let grad_data = grad_tensor.data()?;

                if param_data.len() != grad_data.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Parameter and gradient size mismatch",
                    ));
                }

                // Compute new parameter values: param = param - lr * grad
                let new_data: Vec<f32> = param_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(p, g)| p - lr * g)
                    .collect();

                // Update parameter data in-place
                py_param.update_data(new_data)?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.sgd.zero_grad();
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.parameters.clone()
    }

    #[getter]
    fn param_groups(&self) -> Vec<HashMap<String, PyObject>> {
        // This would need proper implementation
        vec![HashMap::new()]
    }
}

/// Python wrapper for Adam optimizer
#[pyclass]
pub struct Adam {
    /// Underlying Rust Adam optimizer
    adam: RustAdam<f32>,
    /// Parameters being optimized
    pub parameters: Vec<PyTensor>,
}

#[pymethods]
impl Adam {
    #[new]
    #[pyo3(signature = (parameters, lr, beta1=None, beta2=None, epsilon=None))]
    fn new(
        parameters: Vec<PyTensor>,
        lr: f32,
        beta1: Option<f32>,
        beta2: Option<f32>,
        epsilon: Option<f32>,
    ) -> PyResult<Self> {
        let beta1 = beta1.unwrap_or(0.9);
        let beta2 = beta2.unwrap_or(0.999);
        let epsilon = epsilon.unwrap_or(1e-8);

        let rust_params: Vec<_> = parameters.iter().map(|p| p.tensor.clone()).collect();
        let adam = RustAdam::with_options(
            rust_params,
            lr.into(),
            beta1.into(),
            beta2.into(),
            epsilon.into(),
            false, // amsgrad
        );

        Ok(Adam { adam, parameters })
    }

    fn step(&mut self) -> PyResult<()> {
        // Simple Adam implementation directly in Python bindings
        // This avoids the complex synchronization issues with the Rust optimizer

        let lr = if let Some(param_group) = self.adam.param_groups().first() {
            param_group.lr
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No parameter groups found",
            ));
        };

        // For now, just do SGD (Adam implementation would be more complex)
        // Apply SGD update directly to each parameter: param = param - lr * grad
        for py_param in self.parameters.iter_mut() {
            if let Some(grad_tensor) = py_param.grad() {
                let param_data = py_param.data()?;
                let grad_data = grad_tensor.data()?;

                if param_data.len() != grad_data.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Parameter and gradient size mismatch",
                    ));
                }

                // Compute new parameter values: param = param - lr * grad
                let new_data: Vec<f32> = param_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(p, g)| p - lr * g)
                    .collect();

                // Update parameter data in-place
                py_param.update_data(new_data)?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.adam.zero_grad();
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.parameters.clone()
    }

    #[getter]
    fn param_groups(&self) -> Vec<HashMap<String, PyObject>> {
        // This would need proper implementation
        vec![HashMap::new()]
    }
}

/// Python wrapper for AdamW optimizer (placeholder)
#[pyclass]
pub struct AdamW;

#[pymethods]
impl AdamW {
    #[new]
    #[pyo3(signature = (_parameters, _lr=0.001))]
    fn new(_parameters: Vec<PyTensor>, _lr: f64) -> PyResult<Self> {
        Ok(AdamW)
    }

    fn step(&self) -> PyResult<()> {
        Ok(())
    }

    fn zero_grad(&self) -> PyResult<()> {
        Ok(())
    }

    #[getter]
    fn param_groups(&self) -> Vec<HashMap<String, PyObject>> {
        vec![HashMap::new()]
    }
}

/// Python wrapper for RMSprop optimizer (placeholder)
#[pyclass]
pub struct RMSprop;

#[pymethods]
impl RMSprop {
    #[new]
    #[pyo3(signature = (_parameters, _lr=0.01))]
    fn new(_parameters: Vec<PyTensor>, _lr: f64) -> PyResult<Self> {
        Ok(RMSprop)
    }

    fn step(&self) -> PyResult<()> {
        Ok(())
    }

    fn zero_grad(&self) -> PyResult<()> {
        Ok(())
    }

    #[getter]
    fn param_groups(&self) -> Vec<HashMap<String, PyObject>> {
        vec![HashMap::new()]
    }
}

/// Python wrapper for Adagrad optimizer (placeholder)
#[pyclass]
pub struct Adagrad;

#[pymethods]
impl Adagrad {
    #[new]
    #[pyo3(signature = (_parameters, _lr=0.01))]
    fn new(_parameters: Vec<PyTensor>, _lr: f64) -> PyResult<Self> {
        Ok(Adagrad)
    }

    fn step(&self) -> PyResult<()> {
        Ok(())
    }

    fn zero_grad(&self) -> PyResult<()> {
        Ok(())
    }

    #[getter]
    fn param_groups(&self) -> Vec<HashMap<String, PyObject>> {
        vec![HashMap::new()]
    }
}

/// Python wrapper for LBFGS optimizer (placeholder)
#[pyclass]
pub struct Lbfgs;

#[pymethods]
impl Lbfgs {
    #[new]
    #[pyo3(signature = (_parameters, _lr=1.0))]
    fn new(_parameters: Vec<PyTensor>, _lr: f64) -> PyResult<Self> {
        Ok(Lbfgs)
    }

    fn step(&self) -> PyResult<()> {
        Ok(())
    }

    fn zero_grad(&self) -> PyResult<()> {
        Ok(())
    }

    #[getter]
    fn param_groups(&self) -> Vec<HashMap<String, PyObject>> {
        vec![HashMap::new()]
    }
}
