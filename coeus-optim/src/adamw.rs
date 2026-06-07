// ── AdamW Optimizer ──
//
// Decoupled weight-decay regularization (Loshchilov & Hutter, 2019).
// Weight decay is applied directly to the parameter vector rather than
// being folded into the gradient, which prevents the interaction between
// L2 regularization and adaptive learning rates that occurs in Adam+L2.

use coeus_core::{Float, MoiraiBackend};
use coeus_autograd::Var;
use coeus_tensor::Tensor;
use crate::traits::Optimizer;

/// AdamW optimizer.
///
/// # Algorithm
/// ```text
/// m = β₁·m + (1−β₁)·g
/// v = β₂·v + (1−β₂)·g²
/// m̂ = m / (1 − β₁ᵗ),  v̂ = v / (1 − β₂ᵗ)
/// p = p − lr·(m̂ / (√v̂+ε) + λ·p)
/// ```
/// where `λ` is the `weight_decay` coefficient.
pub struct AdamW<T: Float, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Tracked parameter variables.
    pub params: Vec<Var<T, B>>,
    /// Learning rate.
    pub lr: T,
    /// First-moment decay coefficient (β₁).
    pub beta1: T,
    /// Second-moment decay coefficient (β₂).
    pub beta2: T,
    /// Numerical stability constant (ε).
    pub eps: T,
    /// Decoupled weight-decay coefficient (λ).
    pub weight_decay: T,
    /// Current time step (for bias correction).
    pub t: usize,
    /// First moment buffers (m), one per parameter.
    pub m: Vec<Tensor<T, B>>,
    /// Second moment buffers (v), one per parameter.
    pub v: Vec<Tensor<T, B>>,
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> AdamW<T, B> {
    /// Construct an AdamW optimizer.
    ///
    /// # Parameters
    /// - `params`:       trainable variables to optimize.
    /// - `lr`:           learning rate (e.g. `1e-3`).
    /// - `beta1`:        first-moment decay (default `0.9`).
    /// - `beta2`:        second-moment decay (default `0.999`).
    /// - `eps`:          stability constant (default `1e-8`).
    /// - `weight_decay`: decoupled L2 penalty coefficient (e.g. `0.01`).
    pub fn new(
        params: Vec<Var<T, B>>,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
    ) -> Self {
        let backend = B::default();
        let m = params.iter()
            .map(|p| Tensor::zeros_on(p.tensor.shape(), &backend))
            .collect();
        let v = params.iter()
            .map(|p| Tensor::zeros_on(p.tensor.shape(), &backend))
            .collect();
        Self { params, lr, beta1, beta2, eps, weight_decay, t: 0, m, v }
    }

    /// Construct with standard defaults: β₁=0.9, β₂=0.999, ε=1e-8.
    #[inline]
    pub fn with_defaults(params: Vec<Var<T, B>>, lr: T, weight_decay: T) -> Self {
        Self::new(
            params,
            lr,
            T::from_f64(0.9),
            T::from_f64(0.999),
            T::from_f64(1e-8),
            weight_decay,
        )
    }
}

impl<T: Float, B: coeus_ops::BackendOps<T> + Default> Optimizer<T, B> for AdamW<T, B> {
    fn step(&mut self) {
        self.t += 1;
        let backend = B::default();

        for (i, param) in self.params.iter_mut().enumerate() {
            if let Some(ref g) = param.grad {
                let grad_tensor = g.lock().unwrap();
                let m_tensor = &mut self.m[i];
                let v_tensor = &mut self.v[i];

                let (param_storage, param_layout) = param.tensor.storage_mut_and_layout();
                let (m_storage, m_layout) = m_tensor.storage_mut_and_layout();
                let (v_storage, v_layout) = v_tensor.storage_mut_and_layout();

                backend.adamw_step(
                    param_storage,
                    param_layout,
                    grad_tensor.storage(),
                    grad_tensor.layout(),
                    m_storage,
                    m_layout,
                    v_storage,
                    v_layout,
                    self.lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.weight_decay,
                    self.t,
                );
            }
        }
    }

    fn zero_grad(&mut self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    fn set_lr(&mut self, lr: T) {
        self.lr = lr;
    }

    fn clip_grad_norm(&mut self, max_norm: T) -> T
    where
        B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>,
    {
        crate::clip::clip_grad_norm(&self.params, max_norm)
    }
}
