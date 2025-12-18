use backend::CpuBackend;
use dtype::float::Float32;
use optim::{
    Adagrad as RustAdagrad, Adam as RustAdam, BaseOptimizer, Optimizer, SGD as RustSGD,
    RMSprop as RustRMSprop,
};
use storage::DenseStorage;
use pyo3::prelude::*;
use pyo3::pyclass;

// Forward declaration for PyTensor
use super::tensor::PyTensor;

/// Adam optimizer
#[pyclass(name = "Adam", module = "_coeus")]
pub struct Adam {
    pub inner: RustAdam<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl Adam {
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
        let rust_params: Vec<_> = params.into_iter().map(|p| p.inner).collect();
        let adam = RustAdam::with_hyperparams(rust_params, lr, beta1, beta2, epsilon, weight_decay);
        Ok(Adam { inner: adam })
    }

    fn step(&mut self) -> PyResult<()> {
        BaseOptimizer::step(&mut self.inner).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Step failed: {:?}", e))
        })?;
        Ok(())
    }

    fn zero_grad(&mut self) {
        BaseOptimizer::zero_grad(&mut self.inner);
    }

    fn add_param_group(&mut self, params: Vec<PyTensor>) {
        let rust_params: Vec<_> = params.into_iter().map(|p| p.inner).collect();
        BaseOptimizer::add_param_group(&mut self.inner, rust_params);
    }
}

/// SGD optimizer
#[pyclass(name = "SGD", module = "_coeus")]
pub struct Sgd {
    pub inner: RustSGD<CpuBackend<Float32>, Float32>,
}

#[pymethods]
impl Sgd {
    #[new]
    #[pyo3(signature = (params, lr=0.01, momentum=0.0, weight_decay=0.0, dampening=0.0, nesterov=false))]
    fn new(
        mut params: Vec<PyTensor>,
        lr: f64,
        momentum: f64,
        weight_decay: f64,
        dampening: f64,
        nesterov: bool,
    ) -> PyResult<Self> {
        let sgd = RustSGD::new(lr, momentum, weight_decay, dampening, nesterov);
        let mut sgd_instance = Sgd { inner: sgd };
        for (i, param) in params.iter_mut().enumerate() {
            Optimizer::add_param(
                &mut sgd_instance.inner,
                &mut param.inner,
                format!("param_{}", i),
            )
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Parameter addition failed: {:?}",
                    e
                ))
            })?;
        }
        Ok(sgd_instance)
    }

    fn step(&mut self) -> PyResult<()> {
        self.inner.step().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Step failed: {:?}", e))
        })?;
        Ok(())
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    fn add_param_group(&mut self, _params: Vec<PyTensor>) {
        // SGD doesn't support parameter groups in the same way as Adam/Adagrad
        // This is a placeholder implementation
    }
}

/// AdamW optimizer (placeholder - not implemented in core)
#[pyclass(name = "AdamW", module = "_coeus")]
pub struct AdamW;

#[pymethods]
impl AdamW {
    #[new]
    fn new(_lr: f64) -> PyResult<Self> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
            "AdamW optimizer not yet implemented in Coeus core",
        ))
    }
}

/// Adagrad optimizer
#[pyclass(name = "Adagrad", module = "_coeus")]
pub struct Adagrad {
    pub inner: RustAdagrad<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl Adagrad {
    #[new]
    #[pyo3(signature = (params, lr=0.01, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0, eps=1e-10))]
    fn new(
        params: Vec<PyTensor>,
        lr: f64,
        lr_decay: f64,
        weight_decay: f64,
        initial_accumulator_value: f64,
        eps: f64,
    ) -> PyResult<Self> {
        let rust_params: Vec<_> = params.into_iter().map(|p| p.inner).collect();
        let adagrad = RustAdagrad::with_hyperparams(
            rust_params,
            lr,
            lr_decay,
            weight_decay,
            initial_accumulator_value,
            eps,
        );
        Ok(Adagrad { inner: adagrad })
    }

    fn step(&mut self) -> PyResult<()> {
        BaseOptimizer::step(&mut self.inner).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Adagrad step failed: {:?}",
                e
            ))
        })?;
        Ok(())
    }

    fn zero_grad(&mut self) {
        BaseOptimizer::zero_grad(&mut self.inner);
    }

    fn add_param_group(&mut self, params: Vec<PyTensor>) {
        let rust_params: Vec<_> = params.into_iter().map(|p| p.inner).collect();
        BaseOptimizer::add_param_group(&mut self.inner, rust_params);
    }
}

/// RMSprop optimizer
#[pyclass(name = "RMSprop", module = "_coeus")]
pub struct RMSprop {
    pub inner: RustRMSprop<CpuBackend<Float32>, DenseStorage<Float32>, Float32>,
}

#[pymethods]
impl RMSprop {
    #[new]
    #[pyo3(signature = (params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0.0, momentum=0.0, centered=false))]
    fn new(
        mut params: Vec<PyTensor>,
        lr: f64,
        alpha: f64,
        eps: f64,
        weight_decay: f64,
        momentum: f64,
        centered: bool,
    ) -> PyResult<Self> {
        let rmsprop = RustRMSprop::new(lr, alpha, eps, weight_decay, momentum, centered);
        let mut rmsprop_instance = RMSprop { inner: rmsprop };
        for (i, param) in params.iter_mut().enumerate() {
            Optimizer::add_param(
                &mut rmsprop_instance.inner,
                &mut param.inner,
                format!("param_{}", i),
            )
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Parameter addition failed: {:?}",
                    e
                ))
            })?;
        }
        Ok(rmsprop_instance)
    }

    fn step(&mut self) -> PyResult<()> {
        Optimizer::step(&mut self.inner).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("RMSprop step failed: {:?}", e))
        })?;
        Ok(())
    }

    fn zero_grad(&mut self) {
        Optimizer::zero_grad(&mut self.inner);
    }
}
