// All schedulers are temporarily disabled due to lifetime issues

// Temporarily disabled due to lifetime and trait bound complexity
// /// Python wrapper for StepLR scheduler
// #[pyclass]
// #[derive(Clone)]
// pub struct StepLR {
//     /// Underlying Rust StepLR scheduler
//     inner: RustStepLR<'static, AdamW, f32>,
// }

// #[pymethods]
// impl StepLR {
//     #[new]
//     #[pyo3(signature = (optimizer, step_size, gamma=0.1, last_epoch=-1))]
//     fn new(optimizer: &mut AdamW, step_size: usize, gamma: f32, last_epoch: i32) -> PyResult<Self> {
//         let inner = RustStepLR::new(&mut optimizer.adamw, step_size, gamma);
//         Ok(StepLR { inner })
//     }
//
//     fn step(&mut self) -> PyResult<()> {
//         self.inner.step().map_err(|e| {
//             PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("StepLR step failed: {}", e))
//         })
//     }
//
//     #[getter]
//     fn get_last_lr(&self) -> Vec<f32> {
//         self.inner.get_last_lr()
//     }
//
//     #[getter]
//     fn get_lr(&self) -> Vec<f32> {
//         self.inner.get_lr()
//     }
// }

// Temporarily disabled due to trait bound complexity
// /// Python wrapper for ExponentialLR scheduler
// #[pyclass]
// #[derive(Clone)]
// pub struct ExponentialLR {
//     /// Underlying Rust ExponentialLR scheduler
//     inner: RustExponentialLR<'static, AdamW, f32>,
// }

// #[pymethods]
// impl ExponentialLR {
//     #[new]
//     #[pyo3(signature = (optimizer, gamma, last_epoch=-1))]
//     fn new(optimizer: &mut AdamW, gamma: f32, last_epoch: i32) -> PyResult<Self> {
//         let inner = RustExponentialLR::new(&mut optimizer.adamw, gamma);
//         Ok(ExponentialLR { inner })
//     }
//
//     fn step(&mut self) -> PyResult<()> {
//         self.inner.step().map_err(|e| {
//             PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("ExponentialLR step failed: {}", e))
//         })
//     }
//
//     #[getter]
//     fn get_last_lr(&self) -> Vec<f32> {
//         self.inner.get_last_lr()
//     }
//
//     #[getter]
//     fn get_lr(&self) -> Vec<f32> {
//         self.inner.get_lr()
//     }
// }

// Temporarily disabled due to trait bound complexity
// /// Python wrapper for CosineAnnealingLR scheduler
// #[pyclass]
// #[derive(Clone)]
// pub struct CosineAnnealingLR {
//     /// Underlying Rust CosineAnnealingLR scheduler
//     inner: RustCosineAnnealingLR<'static, AdamW, f32>,
// }

// #[pymethods]
// impl CosineAnnealingLR {
//     #[new]
//     #[pyo3(signature = (optimizer, t_max, eta_min=0.0, last_epoch=-1))]
//     fn new(optimizer: &mut AdamW, t_max: usize, eta_min: f32, last_epoch: i32) -> PyResult<Self> {
//         let inner = RustCosineAnnealingLR::new(&mut optimizer.adamw, t_max, eta_min);
//         Ok(CosineAnnealingLR { inner })
//     }
//
//     fn step(&mut self) -> PyResult<()> {
//         self.inner.step().map_err(|e| {
//             PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("CosineAnnealingLR step failed: {}", e))
//         })
//     }
//
//     #[getter]
//     fn get_last_lr(&self) -> Vec<f32> {
//         self.inner.get_last_lr()
//     }
//
//     #[getter]
//     fn get_lr(&self) -> Vec<f32> {
//         self.inner.get_lr()
//     }
// }

// NOTE: ReduceLROnPlateau temporarily disabled due to lifetime issues
// #[pyclass(unsendable)]
// pub struct ReduceLROnPlateau {
//     /// Underlying Rust ReduceLROnPlateau scheduler
//     inner: Box<RustReduceLROnPlateau<'static, coeus_optim::AdamW<f32>, f32>>,
// }

// #[pymethods]
// impl ReduceLROnPlateau {
//     #[new]
//     #[pyo3(signature = (optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4, threshold_mode="rel", cooldown=0, min_lr=0.0, eps=1e-8))]
//     fn new(
//         optimizer: &mut AdamW,
//         mode: &str,
//         factor: f32,
//         patience: usize,
//         threshold: f32,
//         threshold_mode: &str,
//         cooldown: usize,
//         min_lr: f32,
//         eps: f32,
//     ) -> PyResult<Self> {
//         let mode = match mode {
//             "min" => ReduceMode::Min,
//             "max" => ReduceMode::Max,
//             _ => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Invalid mode")),
//         };
//         let threshold_mode = match threshold_mode {
//             "rel" => ThresholdMode::Rel,
//             "abs" => ThresholdMode::Abs,
//             _ => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Invalid threshold_mode")),
//         };
//         let inner = RustReduceLROnPlateau::new(
//             &mut optimizer.adamw,
//             mode,
//             factor,
//             patience,
//         );
//         Ok(ReduceLROnPlateau { inner })
//     }

//     #[pyo3(signature = (metrics, epoch=None))]
//     fn step(&mut self, metrics: f32, epoch: Option<usize>) -> PyResult<bool> {
//         self.inner.step(metrics, epoch).map_err(|e| {
//             PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("ReduceLROnPlateau step failed: {}", e))
//         })
//     }

//     #[getter]
//     fn get_lr(&self) -> Vec<f32> {
//         // ReduceLROnPlateau has per-group LR, collect all groups
//         let mut lrs = Vec::new();
//         let mut i = 0;
//         while let Some(lr) = self.inner.get_lr(i) {
//             lrs.push(lr);
//             i += 1;
//         }
//         lrs
//     }
// }

// NOTE: CyclicLR temporarily disabled due to lifetime issues
// #[pyclass(unsendable)]
// pub struct CyclicLR {
//     /// Underlying Rust CyclicLR scheduler
//     inner: RustCyclicLR<'static, coeus_optim::AdamW<f32>, f32>,
// }

//     #[new]
//     #[pyo3(signature = (optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode="triangular", gamma=1.0, scale_fn=None, scale_mode="cycle", cycle_momentum=true, base_momentum=0.8, max_momentum=0.9, last_epoch=-1))]
//     fn new(
//         optimizer: &mut AdamW,
//         base_lr: f32,
//         max_lr: f32,
//         step_size_up: usize,
//         step_size_down: Option<usize>,
//         mode: &str,
//         gamma: f32,
//         scale_fn: Option<PyObject>,
//         scale_mode: &str,
//         cycle_momentum: bool,
//         base_momentum: f32,
//         max_momentum: f32,
//         last_epoch: i32,
//     ) -> PyResult<Self> {
//         let mode = match mode {
//             "triangular" => CyclicMode::Triangular,
//             "triangular2" => CyclicMode::Triangular2,
//             "exp_range" => CyclicMode::ExpRange,
//             _ => return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Invalid mode")),
//         };
//         let inner = RustCyclicLR::new(
//             &mut optimizer.adamw,
//             base_lr,
//             max_lr,
//             step_size_up,
//             step_size_down,
//             mode,
//         );
//         Ok(CyclicLR { inner })
//     }

//     fn step(&mut self) -> PyResult<()> {
//         self.inner.step().map_err(|e| {
//             PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("CyclicLR step failed: {}", e))
//         })
//     }

//     #[getter]
//     fn get_last_lr(&self) -> Vec<f32> {
//         self.inner.get_last_lr()
//     }

//     #[getter]
//     fn get_lr(&self) -> Vec<f32> {
//         self.inner.get_lr()
//     }
// }

// Temporarily disabled due to lifetime complexity - requires architectural redesign
// /// Python wrapper for OneCycleLR scheduler
// #[pyclass]
// pub struct OneCycleLR {
//     /// Underlying Rust OneCycleLR scheduler
//     inner: RustOneCycleLR<'static, coeus_optim::AdamW<f32>, f32>,
// }

// #[pymethods]
// impl OneCycleLR {
//     #[new]
//     #[pyo3(signature = (optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy="cos", cycle_momentum=true, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, three_phase=false, last_epoch=-1))]
//     fn new(
//         optimizer: &mut AdamW,
//         max_lr: f32,
//         total_steps: Option<usize>,
//         epochs: Option<usize>,
//         steps_per_epoch: Option<usize>,
//         pct_start: f32,
//         anneal_strategy: &str,
//         cycle_momentum: bool,
//         base_momentum: f32,
//         max_momentum: f32,
//         div_factor: f32,
//         final_div_factor: f32,
//         three_phase: bool,
//         last_epoch: i32,
//     ) -> PyResult<Self> {
//         let inner = RustOneCycleLR::new(&mut optimizer.adamw, max_lr, total_steps.unwrap_or(1000));
//         Ok(OneCycleLR { inner })
//     }

//     fn step(&mut self) -> PyResult<()> {
//         self.inner.step().map_err(|e| {
//             PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
//                 "OneCycleLR step failed: {}",
//                 e
//             ))
//         })
//     }

//     #[getter]
//     fn get_last_lr(&self) -> Vec<f32> {
//         self.inner.get_last_lr()
//     }

//     #[getter]
//     fn get_lr(&self) -> Vec<f32> {
//         self.inner.get_lr()
//     }
// }

// Temporarily disabled due to lifetime complexity - requires architectural redesign
// /// Python wrapper for CosineAnnealingWarmRestarts scheduler
// #[pyclass]
// pub struct CosineAnnealingWarmRestarts {
//     /// Underlying Rust CosineAnnealingWarmRestarts scheduler
//     inner: RustCosineAnnealingWarmRestarts<'static, coeus_optim::AdamW<f32>, f32>,
// }

// #[pymethods]
// impl CosineAnnealingWarmRestarts {
//     #[new]
//     #[pyo3(signature = (optimizer, t_0, eta_min=0.0, eta_max=None, t_mult=1.0, last_epoch=-1))]
//     fn new(
//         optimizer: &mut AdamW,
//         t_0: usize,
//         eta_min: f32,
//         eta_max: Option<f32>,
//         t_mult: f32,
//         last_epoch: i32,
//     ) -> PyResult<Self> {
//         // Use current learning rate as eta_max if not specified
//         let eta_max_val = eta_max.unwrap_or_else(|| optimizer.adamw.get_lr(0).unwrap_or(0.1) as f32);
//         let inner = RustCosineAnnealingWarmRestarts::with_t_mult(
//             &mut optimizer.adamw,
//             eta_min,
//             eta_max_val,
//             t_0,
//             t_mult,
//         );
//         Ok(CosineAnnealingWarmRestarts { inner })
//     }

//     fn step(&mut self) -> PyResult<()> {
//         self.inner.step().map_err(|e| {
//             PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
//                 "CosineAnnealingWarmRestarts step failed: {}",
//                 e
//             ))
//         })
//     }

//     #[getter]
//     fn get_last_lr(&self) -> Vec<f32> {
//         self.inner.get_last_lr()
//     }

//     #[getter]
//     fn get_lr(&self) -> Vec<f32> {
//         self.inner.get_lr()
//     }
// }

// Temporarily disabled due to lifetime complexity - requires architectural redesign
// /// Python wrapper for PolynomialLR scheduler
// #[pyclass]
// pub struct PolynomialLR {
//     /// Underlying Rust PolynomialLR scheduler
//     inner: RustPolynomialLR<'static, coeus_optim::AdamW<f32>, f32>,
// }

// #[pymethods]
// impl PolynomialLR {
//     #[new]
//     #[pyo3(signature = (optimizer, max_epochs, power=1.0, last_epoch=-1))]
//     fn new(
//         optimizer: &mut AdamW,
//         max_epochs: usize,
//         power: f32,
//         last_epoch: i32,
//     ) -> PyResult<Self> {
//         // Get current learning rate as eta_max, use 0.0 as eta_min
//         let eta_max = optimizer.adamw.get_lr(0).unwrap_or(0.001);
//         let eta_min = 0.0;
//         let inner =
//             RustPolynomialLR::new(&mut optimizer.adamw, eta_max, eta_min, max_epochs, power);
//         Ok(PolynomialLR { inner })
//     }

//     fn step(&mut self) -> PyResult<()> {
//         self.inner.step().map_err(|e| {
//             PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
//                 "PolynomialLR step failed: {}",
//                 e
//             ))
//         })
//     }

//     #[getter]
//     fn get_last_lr(&self) -> Vec<f32> {
//         self.inner.get_last_lr()
//     }

//     #[getter]
//     fn get_lr(&self) -> Vec<f32> {
//         self.inner.get_lr()
//     }
// }

// Temporarily disabled due to lifetime complexity - requires architectural redesign
// /// Python wrapper for LambdaLR scheduler
// #[pyclass]
// pub struct LambdaLR {
//     /// Underlying Rust LambdaLR scheduler
//     inner: RustLambdaLR<'static, coeus_optim::AdamW<f32>, f32>,
// }

// #[pymethods]
// impl LambdaLR {
//     #[new]
//     #[pyo3(signature = (optimizer, lr_lambda, last_epoch=-1))]
//     fn new(optimizer: &mut AdamW, lr_lambda: PyObject, last_epoch: i32) -> PyResult<Self> {
//         // For now, create a simple linear decay lambda: lambda step: 1.0 / (step + 1)
//         let lambda_fn = |step: usize| -> f32 { 1.0 / (step as f32 + 1.0) };
//         let inner = RustLambdaLR::new(&mut optimizer.adamw, lambda_fn);
//         Ok(LambdaLR { inner })
//     }

//     fn step(&mut self) -> PyResult<()> {
//         self.inner.step().map_err(|e| {
//             PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
//                 "LambdaLR step failed: {}",
//                 e
//             ))
//         })
//     }

//     #[getter]
//     fn get_last_lr(&self) -> Vec<f32> {
//         self.inner.get_last_lr()
//     }

//     #[getter]
//     fn get_lr(&self) -> Vec<f32> {
//         self.inner.get_lr()
//     }
// }

// Temporarily disabled due to lifetime complexity - requires architectural redesign
// /// Python wrapper for MultiplicativeLR scheduler
// #[pyclass]
// pub struct MultiplicativeLR {
//     /// Underlying Rust MultiplicativeLR scheduler
//     inner: RustMultiplicativeLR<'static, coeus_optim::AdamW<f32>, f32>,
// }

// #[pymethods]
// impl MultiplicativeLR {
//     #[new]
//     #[pyo3(signature = (optimizer, lr_lambda, last_epoch=-1))]
//     fn new(optimizer: &mut AdamW, lr_lambda: PyObject, last_epoch: i32) -> PyResult<Self> {
//         // For now, use a default multiplier of 0.9 (90% of current LR each step)
//         let multiplier = 0.9f32;
//         let inner = RustMultiplicativeLR::new(&mut optimizer.adamw, multiplier);
//         Ok(MultiplicativeLR { inner })
//     }

//     fn step(&mut self) -> PyResult<()> {
//         self.inner.step().map_err(|e| {
//             PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
//                 "MultiplicativeLR step failed: {}",
//                 e
//             ))
//         })
//     }

//     #[getter]
//     fn get_last_lr(&self) -> Vec<f32> {
//         self.inner.get_last_lr()
//     }

//     #[getter]
//     fn get_lr(&self) -> Vec<f32> {
//         self.inner.get_lr()
//     }
// }
