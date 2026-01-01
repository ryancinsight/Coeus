use optim::schedulers::{
    CosineAnnealingLR as RustCosineAnnealingLR, ExponentialLR as RustExponentialLR, LRScheduler,
    MultiStepLR as RustMultiStepLR, OneCycleLR as RustOneCycleLR, ReduceLRMode,
    ReduceLROnPlateau as RustReduceLROnPlateau, StepLR as RustStepLR,
};
use pyo3::prelude::*;

/// StepLR scheduler
#[pyclass(name = "StepLR", module = "_coeus")]
pub struct StepLR {
    inner: RustStepLR,
}

#[pymethods]
impl StepLR {
    #[new]
    #[pyo3(signature = (optimizer, step_size, gamma=0.1, last_epoch=-1))]
    fn new(optimizer: Py<PyAny>, step_size: usize, gamma: f64, last_epoch: i64) -> PyResult<Self> {
        let _ = optimizer;
        // Note: optimizer arg is ignored in this low-level binding,
        // the Python wrapper class will handle attaching to optimizer.
        // Or we just return the scheduler and Python side calls step().
        // PyTorch schedulers modify optimizer LR.
        // Rust schedulers are standalone logic engines currently.
        // We will expose them as logic engines.

        let mut scheduler = RustStepLR::new(0.1, step_size, gamma); // Base LR is placeholder
        if last_epoch != -1 {
            scheduler.set_last_epoch(last_epoch as usize);
        }
        Ok(StepLR { inner: scheduler })
    }

    fn step(&mut self) {
        self.inner.step();
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
}

#[pymethods]
impl ExponentialLR {
    #[new]
    #[pyo3(signature = (optimizer, gamma, last_epoch=-1))]
    fn new(optimizer: Py<PyAny>, gamma: f64, last_epoch: i64) -> PyResult<Self> {
        let _ = optimizer;
        let mut scheduler = RustExponentialLR::new(0.1, gamma); // Base LR placeholder
        if last_epoch != -1 {
            scheduler.set_last_epoch(last_epoch as usize);
        }
        Ok(ExponentialLR { inner: scheduler })
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn get_lr(&self) -> f64 {
        self.inner.learning_rate()
    }
}

/// CosineAnnealingLR scheduler
#[pyclass(name = "CosineAnnealingLR", module = "_coeus")]
pub struct CosineAnnealingLR {
    inner: RustCosineAnnealingLR,
}

#[pymethods]
impl CosineAnnealingLR {
    #[new]
    #[pyo3(signature = (optimizer, T_max, eta_min=0.0, last_epoch=-1))]
    #[allow(non_snake_case)]
    fn new(optimizer: Py<PyAny>, T_max: usize, eta_min: f64, last_epoch: i64) -> PyResult<Self> {
        let _ = optimizer;
        let mut scheduler = RustCosineAnnealingLR::new(0.1, eta_min, T_max);
        if last_epoch != -1 {
            scheduler.set_last_epoch(last_epoch as usize);
        }
        Ok(CosineAnnealingLR { inner: scheduler })
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn get_lr(&self) -> f64 {
        self.inner.learning_rate()
    }
}

/// MultiStepLR scheduler
#[pyclass(name = "MultiStepLR", module = "_coeus")]
pub struct MultiStepLR {
    inner: RustMultiStepLR,
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
        let _ = optimizer;
        let mut scheduler = RustMultiStepLR::new(0.1, milestones, gamma);
        if last_epoch != -1 {
            scheduler.set_last_epoch(last_epoch as usize);
        }
        Ok(Self { inner: scheduler })
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn get_lr(&self) -> f64 {
        self.inner.learning_rate()
    }
}

/// ReduceLROnPlateau scheduler
#[pyclass(name = "ReduceLROnPlateau", module = "_coeus")]
pub struct ReduceLROnPlateau {
    inner: RustReduceLROnPlateau,
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
        let _ = optimizer;
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

        let scheduler = RustReduceLROnPlateau::new(
            0.1, // Initial LR placeholder
            reduce_mode,
            factor,
            patience,
            threshold,
            cooldown,
            min_lr,
        );
        Ok(ReduceLROnPlateau { inner: scheduler })
    }

    fn step(&mut self, metrics: f64) {
        self.inner.step(metrics);
    }

    fn get_lr(&self) -> f64 {
        self.inner.learning_rate()
    }
}

/// OneCycleLR scheduler
#[pyclass(name = "OneCycleLR", module = "_coeus")]
pub struct OneCycleLR {
    inner: RustOneCycleLR,
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
        let _ = optimizer;
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

        let initial_lr = max_lr / div_factor;
        let mut scheduler = RustOneCycleLR::new(max_lr, steps, pct_start, initial_lr);

        if last_epoch != -1 {
            scheduler.set_last_epoch(last_epoch as usize);
        }

        Ok(OneCycleLR { inner: scheduler })
    }

    fn step(&mut self) {
        self.inner.step();
    }

    fn get_lr(&self) -> f64 {
        self.inner.learning_rate()
    }
}
