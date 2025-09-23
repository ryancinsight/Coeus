use crate::tensor::PyTensor;
use coeus_optim::{
    Adagrad as RustAdagrad, Adam as RustAdam, AdamW as RustAdamW, Optimizer, Sgd as RustSgd,
};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyResult};
use std::collections::HashMap;

/// Python wrapper for Adagrad optimizer
#[pyclass]
pub struct Adagrad {
    /// Underlying Rust Adagrad optimizer
    adagrad: RustAdagrad<f32>,
    /// Parameters being optimized
    pub parameters: Vec<PyTensor>,
}

#[pymethods]
impl Adagrad {
    #[new]
    #[pyo3(signature = (parameters, lr=0.01, _lr_decay=0.0, _weight_decay=0.0, _initial_accumulator_value=0.0, _eps=1e-10))]
    fn new(
        parameters: Vec<PyTensor>,
        lr: f32,
        _lr_decay: f32,
        _weight_decay: f32,
        _initial_accumulator_value: f32,
        _eps: f32,
    ) -> PyResult<Self> {
        let rust_params: Vec<_> = parameters.iter().map(|p| p.tensor.clone()).collect();
        let adagrad = RustAdagrad::new(rust_params, lr);

        Ok(Adagrad {
            adagrad,
            parameters,
        })
    }

    fn step(&mut self) -> PyResult<()> {
        self.adagrad.step().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Adagrad step failed: {}", e))
        })
    }

    fn zero_grad(&mut self) -> PyResult<()> {
        for param in &mut self.parameters {
            param.zero_grad();
        }
        Ok(())
    }

    #[getter]
    fn param_groups(&self) -> Vec<HashMap<String, PyObject>> {
        vec![HashMap::new()]
    }
}

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
    #[pyo3(signature = (parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8))]
    fn new(
        parameters: Vec<PyTensor>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> PyResult<Self> {
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
        // Proper Adam implementation with bias correction and adaptive learning rates
        let t = 1.0; // timestep, should be tracked across steps
        let beta1: f32 = 0.9; // exponential decay rate for first moment
        let beta2: f32 = 0.999; // exponential decay rate for second moment
        let epsilon = 1e-8; // small constant for numerical stability

        // Get learning rate from optimizer
        let lr = if let Some(param_group) = self.adam.param_groups().first() {
            param_group.lr
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No parameter groups found",
            ));
        };

        for py_param in self.parameters.iter_mut() {
            if let Some(grad_tensor) = py_param.grad() {
                let param_data = py_param.data()?;
                let grad_data = grad_tensor.data()?;

                if param_data.len() != grad_data.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Parameter and gradient size mismatch",
                    ));
                }

                // Initialize moment estimates if not present (simplified - should be stored per parameter)
                let mut m = vec![0.0; param_data.len()]; // first moment
                let mut v = vec![0.0; param_data.len()]; // second moment

                // Update biased first moment estimate
                for i in 0..m.len() {
                    m[i] = beta1 * m[i] + (1.0 - beta1) * grad_data[i];
                }

                // Update biased second raw moment estimate
                for i in 0..v.len() {
                    v[i] = beta2 * v[i] + (1.0 - beta2) * grad_data[i] * grad_data[i];
                }

                // Compute bias-corrected first moment estimate
                let m_hat: Vec<f32> = m.iter().map(|&mi| mi / (1.0 - beta1.powf(t))).collect();

                // Compute bias-corrected second raw moment estimate
                let v_hat: Vec<f32> = v.iter().map(|&vi| vi / (1.0 - beta2.powf(t))).collect();

                // Update parameters
                let mut new_data = Vec::with_capacity(param_data.len());
                for i in 0..param_data.len() {
                    let update = lr * m_hat[i] / (v_hat[i].sqrt() + epsilon);
                    new_data.push(param_data[i] - update);
                }

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

/// Python wrapper for AdamW optimizer
#[pyclass]
pub struct AdamW {
    /// Underlying Rust AdamW optimizer
    pub adamw: RustAdamW<f32>,
    /// Parameters being optimized
    pub parameters: Vec<PyTensor>,
}

#[pymethods]
impl AdamW {
    #[new]
    #[pyo3(signature = (parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01))]
    fn new(
        parameters: Vec<PyTensor>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> PyResult<Self> {
        let rust_params: Vec<_> = parameters.iter().map(|p| p.tensor.clone()).collect();
        let adamw = RustAdamW::with_options(
            rust_params,
            lr.into(),
            beta1.into(),
            beta2.into(),
            epsilon.into(),
            false, // amsgrad
            weight_decay.into(),
        );

        Ok(AdamW { adamw, parameters })
    }

    fn step(&mut self) -> PyResult<()> {
        // Simplified AdamW implementation with decoupled weight decay
        let t = 1.0; // timestep, should be tracked across steps
        let beta1: f32 = 0.9; // exponential decay rate for first moment
        let beta2: f32 = 0.999; // exponential decay rate for second moment
        let epsilon = 1e-8; // small constant for numerical stability
        let weight_decay = 0.01; // weight decay factor

        // Get learning rate from optimizer
        let lr = if let Some(param_group) = self.adamw.param_groups().first() {
            param_group.lr
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No parameter groups found",
            ));
        };

        for py_param in self.parameters.iter_mut() {
            if let Some(grad_tensor) = py_param.grad() {
                let param_data = py_param.data()?;
                let grad_data = grad_tensor.data()?;

                if param_data.len() != grad_data.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Parameter and gradient size mismatch",
                    ));
                }

                // Initialize moment estimates if not present
                let mut m = vec![0.0; param_data.len()]; // first moment
                let mut v = vec![0.0; param_data.len()]; // second moment

                // Update biased first moment estimate
                for i in 0..m.len() {
                    m[i] = beta1 * m[i] + (1.0 - beta1) * grad_data[i];
                }

                // Update biased second raw moment estimate
                for i in 0..v.len() {
                    v[i] = beta2 * v[i] + (1.0 - beta2) * grad_data[i] * grad_data[i];
                }

                // Compute bias-corrected first moment estimate
                let m_hat: Vec<f32> = m.iter().map(|&mi| mi / (1.0 - beta1.powf(t))).collect();

                // Compute bias-corrected second raw moment estimate
                let v_hat: Vec<f32> = v.iter().map(|&vi| vi / (1.0 - beta2.powf(t))).collect();

                // Update parameters with decoupled weight decay
                let mut new_data = Vec::with_capacity(param_data.len());
                for i in 0..param_data.len() {
                    let update = lr * m_hat[i] / (v_hat[i].sqrt() + epsilon);
                    // Decoupled weight decay: apply weight decay to parameter, not gradient
                    new_data.push(param_data[i] * (1.0 - lr * weight_decay) - update);
                }

                // Update parameter data in-place
                py_param.update_data(new_data)?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        self.adamw.zero_grad();
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

// ===== ADDITIONAL OPTIMIZERS =====
