//! Fused optimizer step sub-trait.
//!
//! [`OptimizerOps`] is the interface-segregated sub-trait for fused
//! provider-owned stateful updates (SGD, Adam, RMSProp, AdamW, AdaGrad).

use coeus_core::{ComputeBackend, Float, Layout, Scalar};

/// Fused optimizer step operations.
///
/// Backends implement this optional capability independently from
/// [`BackendOps`](super::super::BackendOps), so integer and inference-only
/// backends do not inherit an optimizer requirement.
pub trait OptimizerOps<T: Scalar>: ComputeBackend {
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
