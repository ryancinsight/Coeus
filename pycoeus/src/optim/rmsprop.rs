use backend::CpuBackend;
use dtype::float::Float32;
use optim::RMSprop as RustRMSprop;
use pyo3::prelude::*;
use storage::DenseStorage;

use crate::optim::base::extract_f32_params;
use crate::tensor::PyTensor;

/// RMSprop optimizer
#[pyclass(name = "RMSprop", module = "coeus", subclass, unsendable)]
pub struct PyRMSprop {
    pub inner: RustRMSprop<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyRMSprop {
    #[new]
    #[pyo3(signature = (params, lr=0.01, alpha=0.99, epsilon=1e-8, weight_decay=0.0, momentum=0.0, centered=false))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        alpha: f64,
        epsilon: f64,
        weight_decay: f64,
        momentum: f64,
        centered: bool,
    ) -> PyResult<Self> {
        if lr <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "lr must be > 0",
            ));
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "alpha must be in [0, 1]",
            ));
        }
        if epsilon < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "epsilon must be >= 0",
            ));
        }
        if weight_decay < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "weight_decay must be >= 0",
            ));
        }
        if momentum < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "momentum must be >= 0",
            ));
        }
        let rust_params = extract_f32_params(params)?;
        let mut rmsprop = RustRMSprop::new(lr, alpha, epsilon, weight_decay, momentum, centered);
        // Add parameters using the BaseOptimizer trait
        optim::BaseOptimizer::add_param_group(&mut rmsprop, rust_params);
        Ok(PyRMSprop { inner: rmsprop })
    }

    fn step(&mut self) -> PyResult<()> {
        self.step_impl()
    }

    fn zero_grad(&mut self) {
        self.zero_grad_impl();
    }

    fn add_param_group(&mut self, params: Vec<PyTensor>) -> PyResult<()> {
        self.add_param_group_impl(params)
    }

    fn state_dict(&self) -> PyResult<std::collections::HashMap<String, PyTensor>> {
        self.state_dict_impl()
    }

    fn load_state_dict(
        &mut self,
        state_dict: std::collections::HashMap<String, PyTensor>,
    ) -> PyResult<()> {
        self.load_state_dict_impl(state_dict)
    }

    fn get_lr(&self) -> f64 {
        self.get_lr_impl()
    }

    fn set_lr(&mut self, lr: f64) {
        self.set_lr_impl(lr);
    }
}

crate::impl_optimizer_methods!(PyRMSprop);
