use optim::schedulers::{
    CosineAnnealingLR as RustCosineAnnealingLR, ExponentialLR as RustExponentialLR, LRScheduler,
    MultiStepLR as RustMultiStepLR, OneCycleLR as RustOneCycleLR, ReduceLRMode,
    ReduceLROnPlateau as RustReduceLROnPlateau, StepLR as RustStepLR,
};
use pyo3::prelude::*;
use pyo3::PyErr;
use std::cmp::Ordering;

use optim::Optimizer as OptimizerTrait;

use crate::optim::{Adagrad, Adam, AdamW, RMSprop, Sgd};

/// StepLR scheduler
#[pyclass(name = "StepLR", module = "_coeus")]
pub struct StepLR {
    inner: RustStepLR,
    optimizer: Py<PyAny>,
}

#[pymethods]
impl StepLR {
    #[new]
    #[pyo3(signature = (optimizer, step_size, gamma=0.1, last_epoch=-1))]
    fn new(optimizer: Py<PyAny>, step_size: usize, gamma: f64, last_epoch: i64) -> PyResult<Self> {
        if step_size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "step_size must be > 0",
            ));
        }
        if gamma.partial_cmp(&0.0) != Some(Ordering::Greater) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "gamma must be > 0",
            ));
        }

        let base_lr = Python::attach(|py| optimizer_get_lr(py, &optimizer))?;
        let mut scheduler = RustStepLR::new(base_lr, step_size, gamma);
        if last_epoch != -1 {
            scheduler.set_last_epoch(last_epoch as usize);
        }
        Ok(StepLR {
            inner: scheduler,
            optimizer,
        })
    }

    fn step(&mut self) -> PyResult<()> {
        self.inner.step();
        let lr = self.inner.learning_rate();
        Python::attach(|py| optimizer_set_lr(py, &self.optimizer, lr))?;
        Ok(())
    }

    fn get_lr(&self) -> f64 {
        self.inner.learning_rate()
    }

    // Helper to sync with optimizer (called from Python)
    fn get_last_lr(&self) -> Vec<f64> {
        vec![self.inner.learning_rate()]
    }
}

/// ExponentialLR scheduler
#[pyclass(name = "ExponentialLR", module = "_coeus")]
pub struct ExponentialLR {
    inner: RustExponentialLR,
    optimizer: Py<PyAny>,
}

#[pymethods]
impl ExponentialLR {
    #[new]
    #[pyo3(signature = (optimizer, gamma, last_epoch=-1))]
    fn new(optimizer: Py<PyAny>, gamma: f64, last_epoch: i64) -> PyResult<Self> {
        if gamma.partial_cmp(&0.0) != Some(Ordering::Greater) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "gamma must be > 0",
            ));
        }

        let base_lr = Python::attach(|py| optimizer_get_lr(py, &optimizer))?;
        let mut scheduler = RustExponentialLR::new(base_lr, gamma);
        if last_epoch != -1 {
            scheduler.set_last_epoch(last_epoch as usize);
        }
        Ok(ExponentialLR {
            inner: scheduler,
            optimizer,
        })
    }

    fn step(&mut self) -> PyResult<()> {
        self.inner.step();
        let lr = self.inner.learning_rate();
        Python::attach(|py| optimizer_set_lr(py, &self.optimizer, lr))?;
        Ok(())
    }

    fn get_lr(&self) -> f64 {
        self.inner.learning_rate()
    }
}

/// CosineAnnealingLR scheduler
#[pyclass(name = "CosineAnnealingLR", module = "_coeus")]
pub struct CosineAnnealingLR {
    inner: RustCosineAnnealingLR,
    optimizer: Py<PyAny>,
}

#[pymethods]
impl CosineAnnealingLR {
    #[new]
    #[pyo3(signature = (optimizer, T_max, eta_min=0.0, last_epoch=-1))]
    #[allow(non_snake_case)]
    fn new(optimizer: Py<PyAny>, T_max: usize, eta_min: f64, last_epoch: i64) -> PyResult<Self> {
        if T_max == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "T_max must be > 0",
            ));
        }
        if eta_min < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "eta_min must be >= 0",
            ));
        }

        let base_lr = Python::attach(|py| optimizer_get_lr(py, &optimizer))?;
        if base_lr.partial_cmp(&eta_min) != Some(Ordering::Greater) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "eta_min must be < optimizer learning rate",
            ));
        }

        let mut scheduler = RustCosineAnnealingLR::new(base_lr, eta_min, T_max);
        if last_epoch != -1 {
            scheduler.set_last_epoch(last_epoch as usize);
        }
        Ok(CosineAnnealingLR {
            inner: scheduler,
            optimizer,
        })
    }

    fn step(&mut self) -> PyResult<()> {
        self.inner.step();
        let lr = self.inner.learning_rate();
        Python::attach(|py| optimizer_set_lr(py, &self.optimizer, lr))?;
        Ok(())
    }

    fn get_lr(&self) -> f64 {
        self.inner.learning_rate()
    }
}

/// MultiStepLR scheduler
#[pyclass(name = "MultiStepLR", module = "_coeus")]
pub struct MultiStepLR {
    inner: RustMultiStepLR,
    optimizer: Py<PyAny>,
}

#[pymethods]
impl MultiStepLR {
    #[new]
    #[pyo3(signature = (optimizer, milestones, gamma=0.1, last_epoch=-1))]
    fn new(
        optimizer: Py<PyAny>,
        milestones: Vec<usize>,
        gamma: f64,
        last_epoch: i64,
    ) -> PyResult<Self> {
        if milestones.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "milestones must be non-empty",
            ));
        }
        if gamma.partial_cmp(&0.0) != Some(Ordering::Greater) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "gamma must be > 0",
            ));
        }

        let base_lr = Python::attach(|py| optimizer_get_lr(py, &optimizer))?;
        let mut scheduler = RustMultiStepLR::new(base_lr, milestones, gamma);
        if last_epoch != -1 {
            scheduler.set_last_epoch(last_epoch as usize);
        }
        Ok(Self {
            inner: scheduler,
            optimizer,
        })
    }

    fn step(&mut self) -> PyResult<()> {
        self.inner.step();
        let lr = self.inner.learning_rate();
        Python::attach(|py| optimizer_set_lr(py, &self.optimizer, lr))?;
        Ok(())
    }

    fn get_lr(&self) -> f64 {
        self.inner.learning_rate()
    }
}

/// ReduceLROnPlateau scheduler
#[pyclass(name = "ReduceLROnPlateau", module = "_coeus")]
pub struct ReduceLROnPlateau {
    inner: RustReduceLROnPlateau,
    optimizer: Py<PyAny>,
}

#[pymethods]
impl ReduceLROnPlateau {
    #[new]
    #[pyo3(signature = (optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4, threshold_mode="rel", cooldown=0, min_lr=0.0, eps=1e-8))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        optimizer: Py<PyAny>,
        mode: &str,
        factor: f64,
        patience: usize,
        threshold: f64,
        threshold_mode: &str,
        cooldown: usize,
        min_lr: f64,
        eps: f64,
    ) -> PyResult<Self> {
        let _ = threshold_mode;
        let _ = eps;
        let reduce_mode = match mode {
            "min" => ReduceLRMode::Min,
            "max" => ReduceLRMode::Max,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "mode must be 'min' or 'max'",
                ))
            }
        };

        if !(factor > 0.0 && factor < 1.0) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "factor must be in (0, 1)",
            ));
        }
        if min_lr < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "min_lr must be >= 0",
            ));
        }
        if threshold < 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "threshold must be >= 0",
            ));
        }

        let initial_lr = Python::attach(|py| optimizer_get_lr(py, &optimizer))?;
        if initial_lr.partial_cmp(&0.0) != Some(Ordering::Greater) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "optimizer learning rate must be > 0",
            ));
        }
        if min_lr.partial_cmp(&initial_lr) != Some(Ordering::Less) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "min_lr must be < optimizer learning rate",
            ));
        }

        let scheduler = RustReduceLROnPlateau::new(
            initial_lr,
            reduce_mode,
            factor,
            patience,
            threshold,
            cooldown,
            min_lr,
        );
        Ok(ReduceLROnPlateau {
            inner: scheduler,
            optimizer,
        })
    }

    fn step(&mut self, metrics: f64) -> PyResult<()> {
        self.inner.step(metrics);
        let lr = self.inner.learning_rate();
        Python::attach(|py| optimizer_set_lr(py, &self.optimizer, lr))?;
        Ok(())
    }

    fn get_lr(&self) -> f64 {
        self.inner.learning_rate()
    }
}

/// OneCycleLR scheduler
#[pyclass(name = "OneCycleLR", module = "_coeus")]
pub struct OneCycleLR {
    inner: RustOneCycleLR,
    optimizer: Py<PyAny>,
}

#[pymethods]
impl OneCycleLR {
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
        let _ = (anneal_strategy, cycle_momentum, base_momentum, max_momentum);
        let _ = (final_div_factor, three_phase, verbose);
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
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "max_lr must be > 0",
            ));
        }
        if div_factor.partial_cmp(&0.0) != Some(Ordering::Greater) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "div_factor must be > 0",
            ));
        }

        let initial_lr = max_lr / div_factor;
        let mut scheduler = RustOneCycleLR::new(max_lr, steps, pct_start, initial_lr);

        if last_epoch != -1 {
            scheduler.set_last_epoch(last_epoch as usize);
        }

        let scheduler_lr = scheduler.learning_rate();
        Python::attach(|py| optimizer_set_lr(py, &optimizer, scheduler_lr))?;

        Ok(OneCycleLR {
            inner: scheduler,
            optimizer,
        })
    }

    fn step(&mut self) -> PyResult<()> {
        self.inner.step();
        let lr = self.inner.learning_rate();
        Python::attach(|py| optimizer_set_lr(py, &self.optimizer, lr))?;
        Ok(())
    }

    fn get_lr(&self) -> f64 {
        self.inner.learning_rate()
    }
}

fn optimizer_get_lr(py: Python<'_>, optimizer: &Py<PyAny>) -> PyResult<f64> {
    let opt = optimizer.bind(py);

    if let Ok(opt) = opt.extract::<PyRef<'_, Adam>>() {
        return Ok(OptimizerTrait::lr(&opt.inner));
    }
    if let Ok(opt) = opt.extract::<PyRef<'_, AdamW>>() {
        return Ok(OptimizerTrait::lr(&opt.inner));
    }
    if let Ok(opt) = opt.extract::<PyRef<'_, Adagrad>>() {
        return Ok(OptimizerTrait::lr(&opt.inner));
    }
    if let Ok(opt) = opt.extract::<PyRef<'_, RMSprop>>() {
        return Ok(OptimizerTrait::lr(&opt.inner));
    }
    if let Ok(opt) = opt.extract::<PyRef<'_, Sgd>>() {
        return Ok(OptimizerTrait::lr(&opt.inner));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Unsupported optimizer type for scheduler",
    ))
}

fn optimizer_set_lr(py: Python<'_>, optimizer: &Py<PyAny>, lr: f64) -> PyResult<()> {
    if lr.partial_cmp(&0.0) != Some(Ordering::Greater) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "learning rate must be > 0",
        ));
    }

    let opt = optimizer.bind(py);

    if let Ok(mut opt) = opt.extract::<PyRefMut<'_, Adam>>() {
        return OptimizerTrait::set_lr(&mut opt.inner, lr).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "failed to set optimizer learning rate: {e:?}"
            ))
        });
    }
    if let Ok(mut opt) = opt.extract::<PyRefMut<'_, AdamW>>() {
        return OptimizerTrait::set_lr(&mut opt.inner, lr).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "failed to set optimizer learning rate: {e:?}"
            ))
        });
    }
    if let Ok(mut opt) = opt.extract::<PyRefMut<'_, Adagrad>>() {
        return OptimizerTrait::set_lr(&mut opt.inner, lr).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "failed to set optimizer learning rate: {e:?}"
            ))
        });
    }
    if let Ok(mut opt) = opt.extract::<PyRefMut<'_, RMSprop>>() {
        return OptimizerTrait::set_lr(&mut opt.inner, lr).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "failed to set optimizer learning rate: {e:?}"
            ))
        });
    }
    if let Ok(mut opt) = opt.extract::<PyRefMut<'_, Sgd>>() {
        return OptimizerTrait::set_lr(&mut opt.inner, lr).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "failed to set optimizer learning rate: {e:?}"
            ))
        });
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Unsupported optimizer type for scheduler",
    ))
}
