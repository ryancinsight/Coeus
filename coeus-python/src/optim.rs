use pyo3::prelude::*;
use crate::tensor::PyTensor;
use coeus_autograd::Var;

/// Python-exposed SGD optimizer.
#[pyclass(name = "SGD")]
pub struct PySGD {
    pub params: Vec<Py<PyTensor>>,
    pub inner: coeus_optim::SGD<f64, coeus_core::MoiraiBackend>,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (params, lr, momentum = 0.0))]
    pub fn new(py: Python<'_>, params: Vec<Py<PyTensor>>, lr: f64, momentum: f64) -> Self {
        let vars: Vec<Var<f64, coeus_core::MoiraiBackend>> = params
            .iter()
            .map(|p| p.borrow(py).inner.clone())
            .collect();
        Self {
            params,
            inner: coeus_optim::SGD::new(vars, lr, momentum),
        }
    }

    /// Perform a single optimization step.
    pub fn step(&mut self, py: Python<'_>) {
        use coeus_optim::traits::Optimizer;
        py.allow_threads(|| self.inner.step());
        for (i, p) in self.params.iter().enumerate() {
            let mut p_borrow = p.borrow_mut(py);
            p_borrow.inner.tensor = self.inner.params[i].tensor.clone();
        }
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
    pub params: Vec<Py<PyTensor>>,
    pub inner: coeus_optim::Adam<f64, coeus_core::MoiraiBackend>,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (params, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8))]
    pub fn new(
        py: Python<'_>,
        params: Vec<Py<PyTensor>>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
    ) -> Self {
        let vars: Vec<Var<f64, coeus_core::MoiraiBackend>> = params
            .iter()
            .map(|p| p.borrow(py).inner.clone())
            .collect();
        Self {
            params,
            inner: coeus_optim::Adam::new(vars, lr, beta1, beta2, eps),
        }
    }

    /// Perform a single optimization step.
    pub fn step(&mut self, py: Python<'_>) {
        use coeus_optim::traits::Optimizer;
        py.allow_threads(|| self.inner.step());
        for (i, p) in self.params.iter().enumerate() {
            let mut p_borrow = p.borrow_mut(py);
            p_borrow.inner.tensor = self.inner.params[i].tensor.clone();
        }
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
    pub params: Vec<Py<PyTensor>>,
    pub inner: coeus_optim::RMSProp<f64, coeus_core::MoiraiBackend>,
}

#[pymethods]
impl PyRMSProp {
    #[new]
    #[pyo3(signature = (params, lr = 1e-2, alpha = 0.99, eps = 1e-8))]
    pub fn new(
        py: Python<'_>,
        params: Vec<Py<PyTensor>>,
        lr: f64,
        alpha: f64,
        eps: f64,
    ) -> Self {
        let vars: Vec<Var<f64, coeus_core::MoiraiBackend>> = params
            .iter()
            .map(|p| p.borrow(py).inner.clone())
            .collect();
        Self {
            params,
            inner: coeus_optim::RMSProp::new(vars, lr, alpha, eps),
        }
    }

    /// Perform a single optimization step.
    pub fn step(&mut self, py: Python<'_>) {
        use coeus_optim::traits::Optimizer;
        py.allow_threads(|| self.inner.step());
        for (i, p) in self.params.iter().enumerate() {
            let mut p_borrow = p.borrow_mut(py);
            p_borrow.inner.tensor = self.inner.params[i].tensor.clone();
        }
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
    pub params: Vec<Py<PyTensor>>,
    pub inner: coeus_optim::AdaGrad<f64, coeus_core::MoiraiBackend>,
}

#[pymethods]
impl PyAdaGrad {
    #[new]
    #[pyo3(signature = (params, lr = 1e-2, eps = 1e-10))]
    pub fn new(
        py: Python<'_>,
        params: Vec<Py<PyTensor>>,
        lr: f64,
        eps: f64,
    ) -> Self {
        let vars: Vec<Var<f64, coeus_core::MoiraiBackend>> = params
            .iter()
            .map(|p| p.borrow(py).inner.clone())
            .collect();
        Self {
            params,
            inner: coeus_optim::AdaGrad::new(vars, lr, eps),
        }
    }

    /// Perform a single optimization step.
    pub fn step(&mut self, py: Python<'_>) {
        use coeus_optim::traits::Optimizer;
        py.allow_threads(|| self.inner.step());
        for (i, p) in self.params.iter().enumerate() {
            let mut p_borrow = p.borrow_mut(py);
            p_borrow.inner.tensor = self.inner.params[i].tensor.clone();
        }
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
    pub params: Vec<Py<PyTensor>>,
    pub inner: coeus_optim::AdamW<f64, coeus_core::MoiraiBackend>,
}

#[pymethods]
impl PyAdamW {
    #[new]
    #[pyo3(signature = (params, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 1e-2))]
    pub fn new(
        py: Python<'_>,
        params: Vec<Py<PyTensor>>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        let vars: Vec<Var<f64, coeus_core::MoiraiBackend>> = params
            .iter()
            .map(|p| p.borrow(py).inner.clone())
            .collect();
        Self {
            params,
            inner: coeus_optim::AdamW::new(vars, lr, beta1, beta2, eps, weight_decay),
        }
    }

    /// Perform a single optimization step.
    pub fn step(&mut self, py: Python<'_>) {
        use coeus_optim::traits::Optimizer;
        py.allow_threads(|| self.inner.step());
        for (i, p) in self.params.iter().enumerate() {
            let mut p_borrow = p.borrow_mut(py);
            p_borrow.inner.tensor = self.inner.params[i].tensor.clone();
        }
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
    StepDecay(coeus_optim::StepDecay),
    CosineAnneal(coeus_optim::CosineAnneal),
    LinearWarmup(coeus_optim::LinearWarmup),
    WarmupCosine(coeus_optim::WarmupCosine),
}

/// Python-exposed compile-time learning rate scheduler wrapper.
#[pyclass(name = "LrScheduler")]
pub struct PyLrScheduler {
    pub strategy: PySchedulerStrategy,
    pub optimizer: PyObject,
    pub base_lr: f64,
    pub step: usize,
}

#[pymethods]
impl PyLrScheduler {
    #[staticmethod]
    pub fn step_decay(optimizer: PyObject, base_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            strategy: PySchedulerStrategy::StepDecay(coeus_optim::StepDecay { step_size, gamma }),
            optimizer,
            base_lr,
            step: 0,
        }
    }

    #[staticmethod]
    pub fn cosine_anneal(optimizer: PyObject, base_lr: f64, t_max: usize, eta_min: f64) -> Self {
        Self {
            strategy: PySchedulerStrategy::CosineAnneal(coeus_optim::CosineAnneal { t_max, eta_min }),
            optimizer,
            base_lr,
            step: 0,
        }
    }

    #[staticmethod]
    pub fn linear_warmup(optimizer: PyObject, base_lr: f64, warmup_steps: usize) -> Self {
        Self {
            strategy: PySchedulerStrategy::LinearWarmup(coeus_optim::LinearWarmup { warmup_steps }),
            optimizer,
            base_lr,
            step: 0,
        }
    }

    #[staticmethod]
    pub fn warmup_cosine(optimizer: PyObject, base_lr: f64, warmup_steps: usize, t_max: usize, eta_min: f64) -> Self {
        Self {
            strategy: PySchedulerStrategy::WarmupCosine(coeus_optim::WarmupCosine { warmup_steps, t_max, eta_min }),
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
            sgd.step(py);
        } else if let Ok(bound) = self.optimizer.bind(py).downcast::<PyAdam>() {
            let mut adam = bound.borrow_mut();
            use coeus_optim::traits::Optimizer;
            adam.inner.set_lr(new_lr);
            adam.step(py);
        } else if let Ok(bound) = self.optimizer.bind(py).downcast::<PyAdamW>() {
            let mut adamw = bound.borrow_mut();
            use coeus_optim::traits::Optimizer;
            adamw.inner.set_lr(new_lr);
            adamw.step(py);
        } else if let Ok(bound) = self.optimizer.bind(py).downcast::<PyRMSProp>() {
            let mut rmsprop = bound.borrow_mut();
            use coeus_optim::traits::Optimizer;
            rmsprop.inner.set_lr(new_lr);
            rmsprop.step(py);
        } else if let Ok(bound) = self.optimizer.bind(py).downcast::<PyAdaGrad>() {
            let mut adagrad = bound.borrow_mut();
            use coeus_optim::traits::Optimizer;
            adagrad.inner.set_lr(new_lr);
            adagrad.step(py);
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err("Unsupported optimizer type"));
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
