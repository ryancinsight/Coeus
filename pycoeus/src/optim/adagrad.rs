use backend::CpuBackend;
use dtype::float::Float32;
use optim::Adagrad as RustAdagrad;
use pyo3::prelude::*;
use storage::DenseStorage;

use crate::optim::base::extract_f32_params;
use crate::tensor::PyTensor;

/// Adagrad optimizer
#[pyclass(name = "Adagrad", module = "coeus", subclass, unsendable)]
pub struct PyAdagrad {
    pub inner: RustAdagrad<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl PyAdagrad {
    #[new]
    #[pyo3(signature = (params, lr=0.01, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0, epsilon=1e-10))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        lr_decay: f64,
        weight_decay: f64,
        initial_accumulator_value: f64,
        epsilon: f64,
    ) -> PyResult<Self> {
        if lr <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "lr must be > 0",
            ));
        }
        if lr_decay < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "lr_decay must be >= 0",
            ));
        }
        if weight_decay < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "weight_decay must be >= 0",
            ));
        }
        if initial_accumulator_value < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "initial_accumulator_value must be >= 0",
            ));
        }
        if epsilon < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "epsilon must be >= 0",
            ));
        }
        let rust_params = extract_f32_params(params)?;
        let adagrad = RustAdagrad::with_hyperparams(
            rust_params,
            lr,
            lr_decay,
            weight_decay,
            initial_accumulator_value,
            epsilon,
        );
        Ok(PyAdagrad { inner: adagrad })
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

crate::impl_optimizer_methods!(PyAdagrad);
