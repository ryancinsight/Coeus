use backend::CpuBackend;
use dtype::float::Float32;
use optim::SGD as RustSGD;
use optim::BaseOptimizer; // Need BaseOptimizer import for add_param_group on SGD?
// Actually in mod.rs it says `optim::BaseOptimizer::add_param_group(&mut sgd, rust_params);`
// So I just need `use optim;` or imports.
use pyo3::prelude::*;

use crate::optim::base::extract_f32_params;
use crate::tensor::PyTensor;

/// SGD optimizer
#[pyclass(name = "SGD", module = "coeus", subclass, unsendable)]
pub struct PySGD {
    pub inner: RustSGD<CpuBackend<Float32>, Float32>,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (params, lr=0.01, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=false))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        momentum: f64,
        dampening: f64,
        weight_decay: f64,
        nesterov: bool,
    ) -> PyResult<Self> {
        if lr <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "lr must be > 0",
            ));
        }
        if momentum < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "momentum must be >= 0",
            ));
        }
        if dampening < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "dampening must be >= 0",
            ));
        }
        if weight_decay < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "weight_decay must be >= 0",
            ));
        }
        let rust_params = extract_f32_params(params)?;
        let mut sgd = RustSGD::new(lr, momentum, weight_decay, dampening, nesterov);
        // Add parameters using the BaseOptimizer trait
        optim::BaseOptimizer::add_param_group(&mut sgd, rust_params);
        Ok(PySGD { inner: sgd })
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

crate::impl_optimizer_methods!(PySGD);
