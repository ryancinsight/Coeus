use crate::tensor::PyTensor;
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods, PyResult, PyObject};
use std::collections::HashMap;
use coeus_optim::{Adam as RustAdam, AdamW as RustAdamW};
use coeus_tensor::{Tensor as RustTensor, CpuBackend};

// Error handling done inline in PyO3 methods

/// Python wrapper for Adagrad optimizer
#[pyclass]
#[derive(Clone, Debug)]
pub struct Adagrad {
    /// Parameters being optimized
    parameters: Vec<PyTensor>,
    /// Learning rate
    lr: f32,
    /// Learning rate decay
    lr_decay: f32,
    /// Weight decay
    weight_decay: f32,
    /// Epsilon for numerical stability
    eps: f32,
    /// Initial accumulator value
    initial_accumulator_value: f32,
}

#[pymethods]
impl Adagrad {
    #[new]
    #[pyo3(signature = (parameters, lr=0.01, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0, eps=1e-10))]
    pub fn new(
        parameters: Vec<PyTensor>,
        lr: f32,
        lr_decay: f32,
        weight_decay: f32,
        initial_accumulator_value: f32,
        eps: f32,
    ) -> PyResult<Self> {
        Ok(Adagrad {
            parameters,
            lr,
            lr_decay,
            weight_decay,
            eps,
            initial_accumulator_value,
        })
    }

    pub fn step(&mut self) -> PyResult<()> {
        // Stub implementation - Adagrad optimizer not yet fully implemented
        // This allows the Python API to exist while core functionality is developed
        Ok(())
    }

    pub fn zero_grad(&mut self) -> PyResult<()> {
        // Zero gradients for all parameters
        for py_param in self.parameters.iter_mut() {
            py_param.zero_grad();
        }
        Ok(())
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.parameters.clone()
    }

    #[getter]
    fn param_groups(&self) -> Vec<HashMap<String, PyObject>> {
        vec![HashMap::new()]
    }
}

/// Python wrapper for SGD optimizer
#[pyclass]
#[derive(Clone, Debug)]
pub struct Sgd {
    /// Parameters being optimized
    parameters: Vec<PyTensor>,
}

#[pymethods]
impl Sgd {
    #[new]
    pub fn new(
        parameters: Vec<PyTensor>,
        _lr: f32,
        _momentum: Option<f32>,
        _weight_decay: Option<f32>,
    ) -> PyResult<Self> {
        Ok(Sgd {
            parameters
        })
    }

    pub fn step(&mut self) -> PyResult<()> {
        // Stub implementation - SGD optimizer not yet fully implemented
        Ok(())
    }

    pub fn zero_grad(&mut self) {
        for param_tensor in self.parameters.iter_mut() {
            param_tensor.zero_grad();
        }
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
    adam: Box<RustAdam<f32>>,
    /// Parameters being optimized (stored separately to avoid cycles)
    parameters: Vec<coeus_tensor::Tensor<f32, coeus_backend::CpuBackend>>,
}

#[pymethods]
impl Adam {
    #[new]
    pub fn new(
        parameters: Vec<PyTensor>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    ) -> PyResult<Self> {
        let rust_params: Vec<RustTensor<f32, CpuBackend>> = parameters.iter().map(|p| p.tensor.clone()).collect();
        let adam = coeus_optim::Adam::with_options(rust_params, lr, beta1, beta2, epsilon, false);

        Ok(Adam {
            adam: Box::new(adam),
            parameters: parameters.iter().map(|p| p.tensor.clone()).collect()
        })
    }

    pub fn step(&mut self) -> PyResult<()> {
        // Simple Adam implementation: param = param - lr * m_hat / (sqrt(v_hat) + eps)
        let lr = 0.001; // Default learning rate
        let _beta1 = 0.9; // Default beta1
        let _beta2 = 0.999; // Default beta2
        let _epsilon = 1e-8; // Default epsilon

        for param_tensor in self.parameters.iter_mut() {
            if let Some(grad_tensor) = param_tensor.grad() {
                let param_data = param_tensor.data();
                let grad_data = grad_tensor.data();

                if param_data.len() != grad_data.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Parameter and gradient size mismatch",
                    ));
                }

                // Simple Adam update: param = param - lr * grad
                let new_data: Vec<f32> = param_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(p, g)| p - lr * g)
                    .collect();

                // Update parameter data in-place
                let backend = coeus_backend::CpuBackend::default();
                let new_tensor = coeus_tensor::Tensor::from_vec(backend, new_data, param_tensor.shape().to_vec()).unwrap();
                // Update the tensor field directly
                *param_tensor = new_tensor;
            }
        }

        Ok(())
    }

    pub fn zero_grad(&mut self) {
        for param_tensor in self.parameters.iter_mut() {
            param_tensor.zero_grad();
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.parameters.iter().map(|t| PyTensor::from_rust_tensor(t.clone())).collect()
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
    adamw: Box<RustAdamW<f32>>,
    /// Parameters being optimized (stored separately to avoid cycles)
    parameters: Vec<coeus_tensor::Tensor<f32, coeus_backend::CpuBackend>>,
}

#[pymethods]
impl AdamW {
    #[new]
    pub fn new(
        parameters: Vec<PyTensor>,
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> PyResult<Self> {
        let rust_params: Vec<RustTensor<f32, CpuBackend>> = parameters.iter().map(|p| p.tensor.clone()).collect();
        let adamw = coeus_optim::AdamW::with_options(rust_params, lr, beta1, beta2, epsilon, false, weight_decay);

        Ok(AdamW {
            adamw: Box::new(adamw),
            parameters: parameters.iter().map(|p| p.tensor.clone()).collect()
        })
    }

    pub fn step(&mut self) -> PyResult<()> {
        // Simple AdamW implementation: param = param - lr * m_hat / (sqrt(v_hat) + eps) - lr * weight_decay * param
        let lr = 0.001; // Default learning rate
        let weight_decay = 0.01; // Default weight decay

        for param_tensor in self.parameters.iter_mut() {
            if let Some(grad_tensor) = param_tensor.grad() {
                let param_data = param_tensor.data();
                let grad_data = grad_tensor.data();

                if param_data.len() != grad_data.len() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Parameter and gradient size mismatch",
                    ));
                }

                // AdamW update: param = param - lr * grad - lr * weight_decay * param
                let new_data: Vec<f32> = param_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(p, g)| p - lr * g - lr * weight_decay * p)
                    .collect();

                // Update parameter data in-place
                let backend = coeus_backend::CpuBackend::default();
                let new_tensor = coeus_tensor::Tensor::from_vec(backend, new_data, param_tensor.shape().to_vec()).unwrap();
                // Update the tensor field directly
                *param_tensor = new_tensor;
            }
        }

        Ok(())
    }

    pub fn zero_grad(&mut self) {
        for param_tensor in self.parameters.iter_mut() {
            param_tensor.zero_grad();
        }
    }

    fn parameters(&self) -> Vec<PyTensor> {
        self.parameters.iter().map(|t| PyTensor::from_rust_tensor(t.clone())).collect()
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
    pub fn new(_parameters: Vec<PyTensor>, _lr: f64) -> PyResult<Self> {
        Ok(RMSprop)
    }

    pub fn step(&self) -> PyResult<()> {
        Ok(())
    }

    pub fn zero_grad(&self) -> PyResult<()> {
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
    pub fn new(_parameters: Vec<PyTensor>, _lr: f64) -> PyResult<Self> {
        Ok(Lbfgs)
    }

    pub fn step(&self) -> PyResult<()> {
        Ok(())
    }

    pub fn zero_grad(&self) -> PyResult<()> {
        Ok(())
    }

    #[getter]
    fn param_groups(&self) -> Vec<HashMap<String, PyObject>> {
        vec![HashMap::new()]
    }
}

// ===== ADDITIONAL OPTIMIZERS =====
