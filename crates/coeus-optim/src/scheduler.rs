// ── Learning rate schedulers ──
//
// Zero-cost scheduler strategies encoded as ZST-compatible value types.
// `LrScheduler<T, B, O, S>` wraps any `Optimizer<T, B>` and a `SchedulerStrategy`.
// Strategy computation executes in `f64` (schedule math) and converts to `T` via
// `Scalar::from_f64` at the `set_lr` boundary — the only precision crossing.

use crate::traits::Optimizer;
use coeus_core::Float;
use std::marker::PhantomData;

// ── Strategy sealed trait ──

/// Strategy for computing the learning rate at a given training step.
///
/// Implementors are stateless value types — no heap, no vtable.
/// Monomorphized into `LrScheduler` at compile time: zero overhead.
///
/// # Examples
///
/// ```
/// use coeus_optim::scheduler::{SchedulerStrategy, StepDecay};
///
/// let s = StepDecay { step_size: 2, gamma: 0.5 };
/// // lr(t) = base * gamma^floor(t / step_size)
/// assert!((s.lr(1.0, 0) - 1.0).abs() < 1e-12);
/// assert!((s.lr(1.0, 2) - 0.5).abs() < 1e-12);
/// assert!((s.lr(1.0, 4) - 0.25).abs() < 1e-12);
/// ```
pub trait SchedulerStrategy: 'static {
    /// Return the absolute learning rate at `step`, given `base_lr`.
    fn lr(&self, base_lr: f64, step: usize) -> f64;
}

// ── Concrete schedules ──

/// Step decay: multiply LR by `gamma` every `step_size` steps.
///
/// `lr(t) = base_lr × γ^⌊t / step_size⌋`
///
/// # Examples
///
/// ```
/// use coeus_optim::scheduler::{SchedulerStrategy, StepDecay};
///
/// let s = StepDecay { step_size: 10, gamma: 0.1 };
/// assert!((s.lr(1e-3, 0) - 1e-3).abs() < 1e-12);
/// assert!((s.lr(1e-3, 10) - 1e-4).abs() < 1e-12);
/// assert!((s.lr(1e-3, 20) - 1e-5).abs() < 1e-12);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct StepDecay {
    /// Number of steps between each LR decay.
    pub step_size: usize,
    /// Multiplicative decay factor applied every `step_size` steps.
    pub gamma: f64,
}

impl SchedulerStrategy for StepDecay {
    #[inline]
    fn lr(&self, base_lr: f64, step: usize) -> f64 {
        base_lr * self.gamma.powi((step / self.step_size) as i32)
    }
}

/// Cosine annealing from `base_lr` down to `eta_min` over `t_max` steps.
///
/// `lr(t) = η_min + ½(base_lr − η_min)(1 + cos(π · min(t, t_max) / t_max))`
#[derive(Clone, Copy, Debug)]
pub struct CosineAnneal {
    /// Total number of steps over which the LR anneals to `eta_min`.
    pub t_max: usize,
    /// Minimum learning rate reached at `t_max`.
    pub eta_min: f64,
}

impl SchedulerStrategy for CosineAnneal {
    #[inline]
    fn lr(&self, base_lr: f64, step: usize) -> f64 {
        if self.t_max == 0 {
            return self.eta_min;
        }
        let t = step.min(self.t_max) as f64;
        let tm = self.t_max as f64;
        self.eta_min
            + 0.5 * (base_lr - self.eta_min) * (1.0 + (std::f64::consts::PI * t / tm).cos())
    }
}

/// Linear warmup: ramp LR from 0 to `base_lr` over `warmup_steps` steps.
///
/// `lr(t) = base_lr × min(t, warmup_steps) / warmup_steps`
#[derive(Clone, Copy, Debug)]
pub struct LinearWarmup {
    /// Number of steps to linearly ramp the LR from 0 to `base_lr`.
    pub warmup_steps: usize,
}

impl SchedulerStrategy for LinearWarmup {
    #[inline]
    fn lr(&self, base_lr: f64, step: usize) -> f64 {
        if self.warmup_steps == 0 {
            return base_lr;
        }
        base_lr * step.min(self.warmup_steps) as f64 / self.warmup_steps as f64
    }
}

/// Linear warmup then cosine annealing.
///
/// - Steps `[0, warmup_steps)`: linearly ramp from 0 to `base_lr`.
/// - Steps `[warmup_steps, …)`: cosine anneal over `t_max` steps (measured from end of warmup).
#[derive(Clone, Copy, Debug)]
pub struct WarmupCosine {
    /// Number of warmup steps ramping the LR from 0 to `base_lr`.
    pub warmup_steps: usize,
    /// Number of cosine-annealing steps (measured from end of warmup).
    pub t_max: usize,
    /// Minimum learning rate reached at the end of annealing.
    pub eta_min: f64,
}

impl SchedulerStrategy for WarmupCosine {
    #[inline]
    fn lr(&self, base_lr: f64, step: usize) -> f64 {
        if step < self.warmup_steps {
            LinearWarmup {
                warmup_steps: self.warmup_steps,
            }
            .lr(base_lr, step)
        } else {
            let cosine_step = step - self.warmup_steps;
            CosineAnneal {
                t_max: self.t_max,
                eta_min: self.eta_min,
            }
            .lr(base_lr, cosine_step)
        }
    }
}

// ── LrScheduler wrapper ──

/// Wraps an optimizer with a compile-time learning rate schedule.
///
/// Monomorphized over `(T, B, O: Optimizer<T,B>, S: SchedulerStrategy)`.
/// The only overhead beyond `O::step()` is one `f64` multiply (strategy), one
/// `Scalar::from_f64` cast, and one `O::set_lr` call per training step.
///
/// # Examples
///
/// ```
/// use coeus_autograd::Var;
/// use coeus_optim::scheduler::{LrScheduler, StepDecay};
/// use coeus_optim::{Optimizer, SGD};
/// use coeus_tensor::Tensor;
///
/// let p: Var<f32> = Var::new(
///     Tensor::from_slice(vec![1], &[1.0f32]).expect("construct tensor"),
///     true,
/// ).expect("construct variable");
/// let opt = SGD::new(vec![coeus_autograd::Parameter::new(p, "weight")], 1e-3f32, 0.0f32)
///     .expect("construct SGD optimizer");
/// let mut scheduler = LrScheduler::new(opt, StepDecay { step_size: 2, gamma: 0.5 }, 1e-3);
///
/// assert!((scheduler.current_lr() - 1e-3).abs() < 1e-7); // step 0
/// scheduler.step().expect("run scheduler step");
/// assert!((scheduler.current_lr() - 1e-3).abs() < 1e-7); // step 1
/// scheduler.step().expect("run scheduler step");
/// assert!((scheduler.current_lr() - 5e-4).abs() < 1e-7); // step 2: gamma^1
/// ```
pub struct LrScheduler<T, B, O, S>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    O: Optimizer<T, B>,
    S: SchedulerStrategy,
{
    /// The wrapped optimizer, updated each `step()`.
    pub optimizer: O,
    /// The compile-time schedule strategy computing the per-step LR.
    pub strategy: S,
    /// Maximum / reference learning rate (in f64 for schedule arithmetic).
    pub base_lr: f64,
    /// Current training step index (0-based).
    pub step: usize,
    _phantom: PhantomData<(T, B)>,
}

impl<T, B, O, S> LrScheduler<T, B, O, S>
where
    T: Float,
    B: coeus_ops::BackendOps<T> + Default,
    O: Optimizer<T, B>,
    S: SchedulerStrategy,
{
    /// Construct a scheduler wrapping `optimizer` with `strategy`.
    ///
    /// `base_lr` is the reference (maximum) learning rate for the schedule formula.
    pub fn new(optimizer: O, strategy: S, base_lr: f64) -> Self {
        Self {
            optimizer,
            strategy,
            base_lr,
            step: 0,
            _phantom: PhantomData,
        }
    }

    /// Advance one training step:
    /// 1. Compute the new LR via the strategy.
    /// 2. Set LR on the optimizer (`T::from_f64` conversion at boundary).
    /// 3. Call `optimizer.step()`.
    /// 4. Increment `self.step`.
    #[inline]
    pub fn step(&mut self) -> Result<(), B::Error> {
        let new_lr = self.strategy.lr(self.base_lr, self.step);
        self.optimizer.set_lr(T::from_f64(new_lr));
        self.optimizer.step()?;
        self.step += 1;
        Ok(())
    }

    /// Zero gradients on the underlying optimizer.
    #[inline]
    pub fn zero_grad(&mut self) -> Result<(), B::Error> {
        self.optimizer.zero_grad()
    }

    /// Current learning rate as `f64` (before any optimizer step).
    #[inline]
    pub fn current_lr(&self) -> f64 {
        self.strategy.lr(self.base_lr, self.step)
    }
}
