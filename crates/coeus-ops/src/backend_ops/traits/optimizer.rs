//! Fused optimizer step sub-trait.
//!
//! [`OptimizerOps`] is the interface-segregated sub-trait for fused
//! provider-owned stateful updates (SGD, Adam, RMSProp, AdamW, AdaGrad).

use coeus_core::{ComputeBackend, Float, Layout, Scalar};

/// Provider rule and validated scalar parameters for one update preflight.
#[derive(Clone, Copy, Debug)]
pub enum OptimizerStepRule<T> {
    /// SGD learning rate and momentum.
    Sgd {
        /// Update scale.
        learning_rate: T,
        /// Velocity retention coefficient.
        momentum: T,
    },
    /// Adam learning rate, moments, epsilon, and bias-correction step.
    Adam {
        /// Update scale.
        learning_rate: T,
        /// First-moment retention coefficient.
        beta_one: T,
        /// Second-moment retention coefficient.
        beta_two: T,
        /// Positive denominator stabilizer.
        epsilon: T,
        /// One-based bias-correction step.
        step: usize,
    },
    /// RMSProp learning rate, moving-average coefficient, and epsilon.
    RmsProp {
        /// Update scale.
        learning_rate: T,
        /// Squared-gradient retention coefficient.
        alpha: T,
        /// Positive denominator stabilizer.
        epsilon: T,
    },
    /// AdamW parameters, including decoupled weight decay.
    AdamW {
        /// Update scale.
        learning_rate: T,
        /// First-moment retention coefficient.
        beta_one: T,
        /// Second-moment retention coefficient.
        beta_two: T,
        /// Positive denominator stabilizer.
        epsilon: T,
        /// Decoupled parameter decay coefficient.
        weight_decay: T,
        /// One-based bias-correction step.
        step: usize,
    },
    /// AdaGrad learning rate and epsilon.
    AdaGrad {
        /// Update scale.
        learning_rate: T,
        /// Positive denominator stabilizer.
        epsilon: T,
    },
}

/// Persistent state buffers borrowed by one update preflight.
pub enum OptimizerStateRef<'a, T: Scalar, B: ComputeBackend> {
    /// One state buffer and its layout.
    One(&'a B::DeviceBuffer<T>, &'a Layout),
    /// Two state buffers and their layouts.
    Two(
        &'a B::DeviceBuffer<T>,
        &'a Layout,
        &'a B::DeviceBuffer<T>,
        &'a Layout,
    ),
}

/// Mutation-free validation request for one provider-owned update.
pub struct OptimizerStepValidation<'a, T: Scalar, B: ComputeBackend> {
    /// Parameter storage and layout.
    pub parameter: (&'a B::DeviceBuffer<T>, &'a Layout),
    /// Gradient storage and layout.
    pub gradient: (&'a B::DeviceBuffer<T>, &'a Layout),
    /// Persistent optimizer state.
    pub state: OptimizerStateRef<'a, T, B>,
    /// Provider rule and scalar parameters.
    pub rule: OptimizerStepRule<T>,
}

/// Fused optimizer step operations.
///
/// Backends implement this optional capability independently from
/// [`BackendOps`](super::super::BackendOps), so integer and inference-only
/// backends do not inherit an optimizer requirement.
pub trait OptimizerOps<T: Scalar>: ComputeBackend {
    /// Validate one fused update without mutating provider storage.
    fn validate_optimizer_step(
        &self,
        validation: OptimizerStepValidation<'_, T, Self>,
    ) -> Result<(), Self::Error>
    where
        Self: Sized,
        T: Float;

    /// Fused SGD step update.
    fn sgd_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        velocity: &mut Self::DeviceBuffer<T>,
        velocity_layout: &Layout,
        lr: T,
        momentum: T,
    ) -> Result<(), Self::Error>
    where
        T: Float;

    /// Fused Adam step update.
    fn adam_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        m: &mut Self::DeviceBuffer<T>,
        m_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
        v_layout: &Layout,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        t: usize,
    ) -> Result<(), Self::Error>
    where
        T: Float;

    /// Fused RMSProp step update.
    fn rmsprop_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
        v_layout: &Layout,
        lr: T,
        alpha: T,
        eps: T,
    ) -> Result<(), Self::Error>
    where
        T: Float;

    /// Fused AdamW step update (decoupled weight decay).
    fn adamw_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        m: &mut Self::DeviceBuffer<T>,
        m_layout: &Layout,
        v: &mut Self::DeviceBuffer<T>,
        v_layout: &Layout,
        lr: T,
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
        t: usize,
    ) -> Result<(), Self::Error>
    where
        T: Float;

    /// Fused AdaGrad step update.
    fn adagrad_step(
        &self,
        param: &mut Self::DeviceBuffer<T>,
        param_layout: &Layout,
        grad: &Self::DeviceBuffer<T>,
        grad_layout: &Layout,
        history: &mut Self::DeviceBuffer<T>,
        history_layout: &Layout,
        lr: T,
        eps: T,
    ) -> Result<(), Self::Error>
    where
        T: Float;
}
