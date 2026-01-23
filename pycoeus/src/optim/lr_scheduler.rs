//! Python bindings for learning rate schedulers

use optim::schedulers::{
    CosineAnnealingLR, ExponentialLR, LRScheduler, MultiStepLR, OneCycleLR, ReduceLRMode,
    ReduceLROnPlateau, StepLR,
};
use pyo3::prelude::*;
use std::cmp::Ordering;

/// Step learning rate scheduler.
#[pyclass(name = "StepLR", module = "coeus.optim", subclass, unsendable)]
pub struct PyStepLR {
    inner: StepLR,
    optimizer: Py<PyAny>,
}

#[pymethods]
impl PyStepLR {
    #[new]
    #[pyo3(signature = (optimizer, step_size, gamma=0.1, last_epoch=-1))]
    fn new(py: Python, optimizer: Py<PyAny>, step_size: usize, gamma: f64, last_epoch: i64) -> PyResult<Self> {
        if step_size == 0 {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("step_size must be > 0"));
        }
        if gamma.partial_cmp(&0.0) != Some(Ordering::Greater) {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("gamma must be > 0"));
        }

        let base_lr: f64 = optimizer
            .bind(py)
            .call_method0("get_lr")
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Optimizer must have 'get_lr' method"))?
            .extract()?;

        let mut inner = StepLR::new(base_lr, step_size, gamma);
        
        if last_epoch != -1 {
            inner.set_last_epoch(last_epoch as usize);
        }

        Ok(PyStepLR { inner, optimizer })
    }

    #[pyo3(signature = (epoch=None))]
    fn step(&mut self, py: Python, epoch: Option<i64>) -> PyResult<()> {
        if let Some(e) = epoch {
            self.inner.set_last_epoch(e as usize);
        } else {
            self.inner.step();
        }
        let lr = self.inner.learning_rate();
        self.optimizer.bind(py).call_method1("set_lr", (lr,))?;
        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.inner.learning_rate()]
    }
}

/// Exponential learning rate scheduler.
#[pyclass(name = "ExponentialLR", module = "coeus.optim", subclass, unsendable)]
pub struct PyExponentialLR {
    inner: ExponentialLR,
    optimizer: Py<PyAny>,
}

#[pymethods]
impl PyExponentialLR {
    #[new]
    #[pyo3(signature = (optimizer, gamma, last_epoch=-1))]
    fn new(py: Python, optimizer: Py<PyAny>, gamma: f64, last_epoch: i64) -> PyResult<Self> {
        if gamma.partial_cmp(&0.0) != Some(Ordering::Greater) {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("gamma must be > 0"));
        }

        let base_lr: f64 = optimizer
            .bind(py)
            .call_method0("get_lr")
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Optimizer must have 'get_lr' method"))?
            .extract()?;

        let mut inner = ExponentialLR::new(base_lr, gamma);
        
        if last_epoch != -1 {
            inner.set_last_epoch(last_epoch as usize);
        }

        Ok(PyExponentialLR { inner, optimizer })
    }

    #[pyo3(signature = (epoch=None))]
    fn step(&mut self, py: Python, epoch: Option<i64>) -> PyResult<()> {
        if let Some(e) = epoch {
            self.inner.set_last_epoch(e as usize);
        } else {
            self.inner.step();
        }
        let lr = self.inner.learning_rate();
        self.optimizer.bind(py).call_method1("set_lr", (lr,))?;
        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.inner.learning_rate()]
    }
}

/// Cosine annealing learning rate scheduler.
#[pyclass(name = "CosineAnnealingLR", module = "coeus.optim", subclass, unsendable)]
pub struct PyCosineAnnealingLR {
    inner: CosineAnnealingLR,
    optimizer: Py<PyAny>,
}

#[pymethods]
impl PyCosineAnnealingLR {
    #[new]
    #[pyo3(signature = (optimizer, T_max, eta_min=0.0, last_epoch=-1))]
    #[allow(non_snake_case)]
    fn new(py: Python, optimizer: Py<PyAny>, T_max: usize, eta_min: f64, last_epoch: i64) -> PyResult<Self> {
        if T_max == 0 {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("T_max must be > 0"));
        }
        if eta_min < 0.0 {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("eta_min must be >= 0"));
        }

        let base_lr: f64 = optimizer
            .bind(py)
            .call_method0("get_lr")
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Optimizer must have 'get_lr' method"))?
            .extract()?;

        if base_lr <= eta_min {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "base_lr (from optimizer) must be greater than eta_min",
            ));
        }

        let mut inner = CosineAnnealingLR::new(base_lr, eta_min, T_max);
        
        if last_epoch != -1 {
            inner.set_last_epoch(last_epoch as usize);
        }

        Ok(PyCosineAnnealingLR { inner, optimizer })
    }

    #[pyo3(signature = (epoch=None))]
    fn step(&mut self, py: Python, epoch: Option<i64>) -> PyResult<()> {
        if let Some(e) = epoch {
            self.inner.set_last_epoch(e as usize);
        } else {
            self.inner.step();
        }
        let lr = self.inner.learning_rate();
        self.optimizer.bind(py).call_method1("set_lr", (lr,))?;
        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.inner.learning_rate()]
    }
}

/// Multi-step learning rate scheduler.
#[pyclass(name = "MultiStepLR", module = "coeus.optim", subclass, unsendable)]
pub struct PyMultiStepLR {
    inner: MultiStepLR,
    optimizer: Py<PyAny>,
}

#[pymethods]
impl PyMultiStepLR {
    #[new]
    #[pyo3(signature = (optimizer, milestones, gamma=0.1, last_epoch=-1))]
    fn new(py: Python, optimizer: Py<PyAny>, milestones: Vec<usize>, gamma: f64, last_epoch: i64) -> PyResult<Self> {
        if milestones.is_empty() {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("milestones must be non-empty"));
        }
        if gamma.partial_cmp(&0.0) != Some(Ordering::Greater) {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("gamma must be > 0"));
        }

        let base_lr: f64 = optimizer
            .bind(py)
            .call_method0("get_lr")
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Optimizer must have 'get_lr' method"))?
            .extract()?;

        let mut sorted_milestones = milestones.clone();
        sorted_milestones.sort();

        let mut inner = MultiStepLR::new(base_lr, sorted_milestones, gamma);
        
        if last_epoch != -1 {
            inner.set_last_epoch(last_epoch as usize);
        }

        Ok(PyMultiStepLR { inner, optimizer })
    }

    #[pyo3(signature = (epoch=None))]
    fn step(&mut self, py: Python, epoch: Option<i64>) -> PyResult<()> {
        if let Some(e) = epoch {
            self.inner.set_last_epoch(e as usize);
        } else {
            self.inner.step();
        }
        let lr = self.inner.learning_rate();
        self.optimizer.bind(py).call_method1("set_lr", (lr,))?;
        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.inner.learning_rate()]
    }
}

/// Reduce learning rate on plateau scheduler.
#[pyclass(name = "ReduceLROnPlateau", module = "coeus.optim", subclass, unsendable)]
pub struct PyReduceLROnPlateau {
    inner: ReduceLROnPlateau,
    optimizer: Py<PyAny>,
}

#[pymethods]
impl PyReduceLROnPlateau {
    #[new]
    #[pyo3(signature = (optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4, threshold_mode="rel", cooldown=0, min_lr=0.0, eps=1e-8))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python,
        optimizer: Py<PyAny>,
        mode: &str,
        factor: f64,
        patience: usize,
        threshold: f64,
        threshold_mode: &str,
        cooldown: usize,
        min_lr: f64,
        eps: Option<f64>,
    ) -> PyResult<Self> {
        let _ = eps;
        let _ = threshold_mode;
        
        if !(factor > 0.0 && factor < 1.0) {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("factor must be in (0, 1)"));
        }
        if min_lr < 0.0 {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("min_lr must be >= 0"));
        }
        if threshold < 0.0 {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("threshold must be >= 0"));
        }

        let initial_lr: f64 = optimizer
            .bind(py)
            .call_method0("get_lr")
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyValueError, _>("Optimizer must have 'get_lr' method"))?
            .extract()?;

        if initial_lr.partial_cmp(&0.0) != Some(Ordering::Greater) {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("optimizer learning rate must be > 0"));
        }
        if min_lr.partial_cmp(&initial_lr) != Some(Ordering::Less) {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("min_lr must be < optimizer learning rate"));
        }

        let reduce_mode = match mode.to_lowercase().as_str() {
            "min" => ReduceLRMode::Min,
            "max" => ReduceLRMode::Max,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("mode must be 'min' or 'max'")),
        };

        let inner = ReduceLROnPlateau::new(
            initial_lr,
            reduce_mode,
            factor,
            patience,
            threshold,
            cooldown,
            min_lr,
        );

        Ok(PyReduceLROnPlateau { inner, optimizer })
    }

    #[pyo3(signature = (metrics, epoch=None))]
    fn step(&mut self, py: Python, metrics: f64, epoch: Option<i64>) -> PyResult<()> {
        let _ = epoch; 
        self.inner.step(metrics);
        let lr = self.inner.learning_rate();
        self.optimizer.bind(py).call_method1("set_lr", (lr,))?;
        Ok(())
    }
    
    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.inner.learning_rate()]
    }
}

/// OneCycleLR scheduler.
#[pyclass(name = "OneCycleLR", module = "coeus.optim", subclass, unsendable)]
pub struct PyOneCycleLR {
    inner: OneCycleLR,
    optimizer: Py<PyAny>,
}

#[pymethods]
impl PyOneCycleLR {
    #[new]
    #[pyo3(signature = (optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy="cos", cycle_momentum=true, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=1e4, three_phase=false, last_epoch=-1, verbose=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        optimizer: Py<PyAny>,
        max_lr: f64,
        total_steps: Option<usize>,
        epochs: Option<usize>,
        steps_per_epoch: Option<usize>,
        pct_start: f64,
        anneal_strategy: &str,
        cycle_momentum: bool,
        base_momentum: f64,
        max_momentum: f64,
        div_factor: f64,
        final_div_factor: f64,
        three_phase: bool,
        last_epoch: i64,
        verbose: bool,
    ) -> PyResult<Self> {
        let _ = anneal_strategy;
        let _ = cycle_momentum;
        let _ = base_momentum;
        let _ = max_momentum;
        let _ = final_div_factor;
        let _ = three_phase;
        let _ = verbose;

        // Calculate total steps
        let steps = match total_steps {
            Some(s) => s,
            None => match (epochs, steps_per_epoch) {
                (Some(e), Some(s)) => e * s,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Either total_steps or (epochs, steps_per_epoch) must be provided",
                    ))
                }
            },
        };

        if max_lr.partial_cmp(&0.0) != Some(Ordering::Greater) {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("max_lr must be > 0"));
        }
        if div_factor.partial_cmp(&0.0) != Some(Ordering::Greater) {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("div_factor must be > 0"));
        }
        
        let initial_lr = max_lr / div_factor;
        
        let mut inner = OneCycleLR::new(max_lr, steps, pct_start, initial_lr);

        if last_epoch != -1 {
            inner.set_last_epoch(last_epoch as usize);
        }

        Ok(PyOneCycleLR { inner, optimizer })
    }

    #[pyo3(signature = (epoch=None))]
    fn step(&mut self, py: Python, epoch: Option<i64>) -> PyResult<()> {
        if let Some(e) = epoch {
            self.inner.set_last_epoch(e as usize);
        } else {
            self.inner.step();
        }
        let lr = self.inner.learning_rate();
        self.optimizer.bind(py).call_method1("set_lr", (lr,))?;
        Ok(())
    }

    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.inner.learning_rate()]
    }
}
