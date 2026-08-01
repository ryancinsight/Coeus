use crate::{nn::error::map_backend_error, tensor::PyTensor};
use coeus_autograd::Parameter;
use pyo3::prelude::*;

type NamedPyParameter = (String, Py<PyTensor>);

fn split_parameters(
    py: Python<'_>,
    parameters: Vec<NamedPyParameter>,
) -> (
    Vec<Py<PyTensor>>,
    Vec<Parameter<f64, coeus_core::MoiraiBackend>>,
) {
    let mut python = Vec::with_capacity(parameters.len());
    let mut rust = Vec::with_capacity(parameters.len());
    for (name, parameter) in parameters {
        rust.push(Parameter::new(parameter.borrow(py).inner.clone(), name));
        python.push(parameter);
    }
    (python, rust)
}

fn sync_parameters(
    py: Python<'_>,
    python: &[Py<PyTensor>],
    rust: &[Parameter<f64, coeus_core::MoiraiBackend>],
) {
    for (parameter, source) in python.iter().zip(rust) {
        parameter.borrow_mut(py).inner.tensor = source.var.tensor.clone();
    }
}

/// Python-exposed SGD optimizer.
#[pyclass(name = "SGD")]
pub struct PySGD {
    /// Python-owned parameter tensors being optimized.
    pub params: Vec<Py<PyTensor>>,
    /// Underlying Rust SGD optimizer state.
    pub inner: coeus_optim::SGD<f64, coeus_core::MoiraiBackend>,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (params, lr, momentum = 0.0))]
    /// Create an SGD optimizer over `params` with learning rate `lr` and optional `momentum`.
    pub fn new(py: Python<'_>, params: Vec<NamedPyParameter>, lr: f64, momentum: f64) -> Self {
        let (params, named) = split_parameters(py, params);
        Self {
            params,
            inner: coeus_optim::SGD::new(named, lr, momentum),
        }
    }

    /// Perform a single optimization step.
    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        use coeus_optim::traits::Optimizer;
        let result = py.allow_threads(|| self.inner.step());
        sync_parameters(py, &self.params, &self.inner.params);
        result.map_err(map_backend_error)
    }

    /// Zero all parameter gradients.
    pub fn zero_grad(&mut self) {
        use coeus_optim::traits::Optimizer;
        self.inner.zero_grad();
    }

    /// Clip gradient norms across all parameters to `max_norm`.
    pub fn clip_grad_norm(&mut self, max_norm: f64) -> f64 {
        use coeus_optim::traits::Optimizer;
        self.inner.clip_grad_norm(max_norm)
    }
}

/// Python-exposed Adam optimizer.
#[pyclass(name = "Adam")]
pub struct PyAdam {
    /// Python-owned parameter tensors being optimized.
    pub params: Vec<Py<PyTensor>>,
    /// Underlying Rust Adam optimizer state.
    pub inner: coeus_optim::Adam<f64, coeus_core::MoiraiBackend>,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (params, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8))]
    /// Create an Adam optimizer over `params`.
    pub fn new(
        py: Python<'_>,
        params: Vec<NamedPyParameter>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
    ) -> Self {
        let (params, named) = split_parameters(py, params);
        Self {
            params,
            inner: coeus_optim::Adam::new(named, lr, beta1, beta2, eps),
        }
    }

    /// Perform a single optimization step.
    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        use coeus_optim::traits::Optimizer;
        let result = py.allow_threads(|| self.inner.step());
        sync_parameters(py, &self.params, &self.inner.params);
        result.map_err(map_backend_error)
    }

    /// Zero all parameter gradients.
    pub fn zero_grad(&mut self) {
        use coeus_optim::traits::Optimizer;
        self.inner.zero_grad();
    }

    /// Clip gradient norms across all parameters to `max_norm`.
    pub fn clip_grad_norm(&mut self, max_norm: f64) -> f64 {
        use coeus_optim::traits::Optimizer;
        self.inner.clip_grad_norm(max_norm)
    }
}

/// Python-exposed RMSProp optimizer.
#[pyclass(name = "RMSProp")]
pub struct PyRMSProp {
    /// Python-owned parameter tensors being optimized.
    pub params: Vec<Py<PyTensor>>,
    /// Underlying Rust RMSProp optimizer state.
    pub inner: coeus_optim::RMSProp<f64, coeus_core::MoiraiBackend>,
}

#[pymethods]
impl PyRMSProp {
    #[new]
    #[pyo3(signature = (params, lr = 1e-2, alpha = 0.99, eps = 1e-8))]
    /// Create an RMSProp optimizer over `params`.
    pub fn new(
        py: Python<'_>,
        params: Vec<NamedPyParameter>,
        lr: f64,
        alpha: f64,
        eps: f64,
    ) -> Self {
        let (params, named) = split_parameters(py, params);
        Self {
            params,
            inner: coeus_optim::RMSProp::new(named, lr, alpha, eps),
        }
    }

    /// Perform a single optimization step.
    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        use coeus_optim::traits::Optimizer;
        let result = py.allow_threads(|| self.inner.step());
        sync_parameters(py, &self.params, &self.inner.params);
        result.map_err(map_backend_error)
    }

    /// Zero all parameter gradients.
    pub fn zero_grad(&mut self) {
        use coeus_optim::traits::Optimizer;
        self.inner.zero_grad();
    }

    /// Clip gradient norms across all parameters to `max_norm`.
    pub fn clip_grad_norm(&mut self, max_norm: f64) -> f64 {
        use coeus_optim::traits::Optimizer;
        self.inner.clip_grad_norm(max_norm)
    }
}

/// Python-exposed AdaGrad optimizer.
#[pyclass(name = "AdaGrad")]
pub struct PyAdaGrad {
    /// Python-owned parameter tensors being optimized.
    pub params: Vec<Py<PyTensor>>,
    /// Underlying Rust AdaGrad optimizer state.
    pub inner: coeus_optim::AdaGrad<f64, coeus_core::MoiraiBackend>,
}

#[pymethods]
impl PyAdaGrad {
    #[new]
    #[pyo3(signature = (params, lr = 1e-2, eps = 1e-10))]
    /// Create an AdaGrad optimizer over `params`.
    pub fn new(py: Python<'_>, params: Vec<NamedPyParameter>, lr: f64, eps: f64) -> Self {
        let (params, named) = split_parameters(py, params);
        Self {
            params,
            inner: coeus_optim::AdaGrad::new(named, lr, eps),
        }
    }

    /// Perform a single optimization step.
    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        use coeus_optim::traits::Optimizer;
        let result = py.allow_threads(|| self.inner.step());
        sync_parameters(py, &self.params, &self.inner.params);
        result.map_err(map_backend_error)
    }

    /// Zero all parameter gradients.
    pub fn zero_grad(&mut self) {
        use coeus_optim::traits::Optimizer;
        self.inner.zero_grad();
    }

    /// Clip gradient norms across all parameters to `max_norm`.
    pub fn clip_grad_norm(&mut self, max_norm: f64) -> f64 {
        use coeus_optim::traits::Optimizer;
        self.inner.clip_grad_norm(max_norm)
    }
}

/// Python-exposed AdamW optimizer.
#[pyclass(name = "AdamW")]
pub struct PyAdamW {
    /// Python-owned parameter tensors being optimized.
    pub params: Vec<Py<PyTensor>>,
    /// Underlying Rust AdamW optimizer state.
    pub inner: coeus_optim::AdamW<f64, coeus_core::MoiraiBackend>,
}

#[pymethods]
impl PyAdamW {
    #[new]
    #[pyo3(signature = (params, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 1e-2))]
    /// Create an AdamW optimizer over `params`.
    pub fn new(
        py: Python<'_>,
        params: Vec<NamedPyParameter>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        let (params, named) = split_parameters(py, params);
        Self {
            params,
            inner: coeus_optim::AdamW::new(named, lr, beta1, beta2, eps, weight_decay),
        }
    }

    /// Perform a single optimization step.
    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        use coeus_optim::traits::Optimizer;
        let result = py.allow_threads(|| self.inner.step());
        sync_parameters(py, &self.params, &self.inner.params);
        result.map_err(map_backend_error)
    }

    /// Zero all parameter gradients.
    pub fn zero_grad(&mut self) {
        use coeus_optim::traits::Optimizer;
        self.inner.zero_grad();
    }

    /// Clip gradient norms across all parameters to `max_norm`.
    pub fn clip_grad_norm(&mut self, max_norm: f64) -> f64 {
        use coeus_optim::traits::Optimizer;
        self.inner.clip_grad_norm(max_norm)
    }
}

/// Helper enum for Python-exposed scheduler strategies.
pub enum PySchedulerStrategy {
    /// Step decay: multiply LR by `gamma` every `step_size` epochs.
    StepDecay(coeus_optim::StepDecay),
    /// Cosine annealing: LR follows a cosine curve down to `eta_min`.
    CosineAnneal(coeus_optim::CosineAnneal),
    /// Linear warmup: LR increases linearly from 0 to `base_lr` over `warmup_steps`.
    LinearWarmup(coeus_optim::LinearWarmup),
    /// Warmup then cosine decay: linear warmup followed by cosine annealing.
    WarmupCosine(coeus_optim::WarmupCosine),
}

/// Python-exposed compile-time learning rate scheduler wrapper.
#[pyclass(name = "LrScheduler")]
pub struct PyLrScheduler {
    /// Active scheduling strategy.
    pub strategy: PySchedulerStrategy,
    /// The wrapped optimizer whose LR will be updated each step.
    pub optimizer: PyObject,
    /// Initial (peak) learning rate.
    pub base_lr: f64,
    /// Current training step count.
    pub step: usize,
}

#[pymethods]
impl PyLrScheduler {
    #[staticmethod]
    /// Create a scheduler using StepDecay: multiply LR by `gamma` every `step_size` steps.
    pub fn step_decay(optimizer: PyObject, base_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            strategy: PySchedulerStrategy::StepDecay(coeus_optim::StepDecay { step_size, gamma }),
            optimizer,
            base_lr,
            step: 0,
        }
    }

    #[staticmethod]
    /// Create a scheduler using CosineAnneal over `t_max` steps down to `eta_min`.
    pub fn cosine_anneal(optimizer: PyObject, base_lr: f64, t_max: usize, eta_min: f64) -> Self {
        Self {
            strategy: PySchedulerStrategy::CosineAnneal(coeus_optim::CosineAnneal {
                t_max,
                eta_min,
            }),
            optimizer,
            base_lr,
            step: 0,
        }
    }

    #[staticmethod]
    /// Create a scheduler using LinearWarmup over `warmup_steps` steps.
    pub fn linear_warmup(optimizer: PyObject, base_lr: f64, warmup_steps: usize) -> Self {
        Self {
            strategy: PySchedulerStrategy::LinearWarmup(coeus_optim::LinearWarmup { warmup_steps }),
            optimizer,
            base_lr,
            step: 0,
        }
    }

    #[staticmethod]
    /// Create a scheduler using linear warmup followed by cosine decay.
    pub fn warmup_cosine(
        optimizer: PyObject,
        base_lr: f64,
        warmup_steps: usize,
        t_max: usize,
        eta_min: f64,
    ) -> Self {
        Self {
            strategy: PySchedulerStrategy::WarmupCosine(coeus_optim::WarmupCosine {
                warmup_steps,
                t_max,
                eta_min,
            }),
            optimizer,
            base_lr,
            step: 0,
        }
    }

    /// Advance one training step, updating the wrapped optimizer's learning rate and stepping.
    pub fn step(&mut self, py: Python<'_>) -> PyResult<()> {
        let new_lr = match &self.strategy {
            PySchedulerStrategy::StepDecay(s) => {
                use coeus_optim::SchedulerStrategy;
                s.lr(self.base_lr, self.step)
            }
            PySchedulerStrategy::CosineAnneal(s) => {
                use coeus_optim::SchedulerStrategy;
                s.lr(self.base_lr, self.step)
            }
            PySchedulerStrategy::LinearWarmup(s) => {
                use coeus_optim::SchedulerStrategy;
                s.lr(self.base_lr, self.step)
            }
            PySchedulerStrategy::WarmupCosine(s) => {
                use coeus_optim::SchedulerStrategy;
                s.lr(self.base_lr, self.step)
            }
        };

        if let Ok(bound) = self.optimizer.bind(py).downcast::<PySGD>() {
            let mut sgd = bound.borrow_mut();
            use coeus_optim::traits::Optimizer;
            sgd.inner.set_lr(new_lr);
            sgd.step(py)?;
        } else if let Ok(bound) = self.optimizer.bind(py).downcast::<PyAdam>() {
            let mut adam = bound.borrow_mut();
            use coeus_optim::traits::Optimizer;
            adam.inner.set_lr(new_lr);
            adam.step(py)?;
        } else if let Ok(bound) = self.optimizer.bind(py).downcast::<PyAdamW>() {
            let mut adamw = bound.borrow_mut();
            use coeus_optim::traits::Optimizer;
            adamw.inner.set_lr(new_lr);
            adamw.step(py)?;
        } else if let Ok(bound) = self.optimizer.bind(py).downcast::<PyRMSProp>() {
            let mut rmsprop = bound.borrow_mut();
            use coeus_optim::traits::Optimizer;
            rmsprop.inner.set_lr(new_lr);
            rmsprop.step(py)?;
        } else if let Ok(bound) = self.optimizer.bind(py).downcast::<PyAdaGrad>() {
            let mut adagrad = bound.borrow_mut();
            use coeus_optim::traits::Optimizer;
            adagrad.inner.set_lr(new_lr);
            adagrad.step(py)?;
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Unsupported optimizer type",
            ));
        }

        self.step += 1;
        Ok(())
    }

    /// Get the learning rate for the current step.
    pub fn current_lr(&self) -> f64 {
        match &self.strategy {
            PySchedulerStrategy::StepDecay(s) => {
                use coeus_optim::SchedulerStrategy;
                s.lr(self.base_lr, self.step)
            }
            PySchedulerStrategy::CosineAnneal(s) => {
                use coeus_optim::SchedulerStrategy;
                s.lr(self.base_lr, self.step)
            }
            PySchedulerStrategy::LinearWarmup(s) => {
                use coeus_optim::SchedulerStrategy;
                s.lr(self.base_lr, self.step)
            }
            PySchedulerStrategy::WarmupCosine(s) => {
                use coeus_optim::SchedulerStrategy;
                s.lr(self.base_lr, self.step)
            }
        }
    }
}
