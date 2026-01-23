use backend::CpuBackend;
use dtype::float::Float32;
use optim::adadelta::Adadelta as RustAdadelta;
use pyo3::prelude::*;
use storage::DenseStorage;

use crate::optim::base::extract_f32_params;
use crate::tensor::PyTensor;

#[pyclass(name = "Adadelta", module = "coeus", subclass, unsendable)]
pub struct PyAdadelta {
    pub inner: RustAdadelta<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyAdadelta {
    #[new]
    #[pyo3(signature = (params, lr=1.0, rho=0.9, epsilon=1e-6, weight_decay=0.0))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        rho: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> PyResult<Self> {
        if lr <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "lr must be > 0",
            ));
        }
        if !(0.0..=1.0).contains(&rho) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "rho must be in [0, 1]",
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

        let rust_params = extract_f32_params(params)?;
        let adadelta = RustAdadelta::with_hyperparams(rust_params, lr, rho, epsilon, weight_decay);
        Ok(PyAdadelta { inner: adadelta })
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

crate::impl_optimizer_methods!(PyAdadelta);
