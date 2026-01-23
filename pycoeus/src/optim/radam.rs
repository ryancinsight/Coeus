use backend::CpuBackend;
use dtype::float::Float32;
use optim::radam::RAdam as RustRAdam;
use pyo3::prelude::*;
use storage::DenseStorage;

use crate::optim::base::extract_f32_params;
use crate::tensor::PyTensor;

#[pyclass(name = "RAdam", module = "coeus", subclass, unsendable)]
pub struct PyRAdam {
    pub inner: RustRAdam<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyRAdam {
    #[new]
    #[pyo3(signature = (params, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> PyResult<Self> {
        if lr <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "lr must be > 0",
            ));
        }
        if !(0.0..1.0).contains(&beta1) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "beta1 must be in [0, 1)",
            ));
        }
        if !(0.0..1.0).contains(&beta2) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "beta2 must be in [0, 1)",
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
        let radam =
            RustRAdam::with_hyperparams(rust_params, lr, beta1, beta2, epsilon, weight_decay);
        Ok(PyRAdam { inner: radam })
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

crate::impl_optimizer_methods!(PyRAdam);
