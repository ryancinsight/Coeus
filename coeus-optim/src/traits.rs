// ── Optimizer trait ──

use coeus_core::{Scalar, MoiraiBackend};

/// Trait for parameter optimizers.
pub trait Optimizer<T: Scalar, B: coeus_ops::BackendOps<T> + Default = MoiraiBackend> {
    /// Perform one optimization step using accumulated gradients.
    fn step(&mut self);

    /// Zero all parameter gradients.
    fn zero_grad(&mut self);

    /// Update the learning rate for all parameter groups.
    fn set_lr(&mut self, lr: T);

    /// Clip gradient L2 norms across all parameters to `max_norm`.
    ///
    /// Returns the pre-clip total L2 norm.
    fn clip_grad_norm(&mut self, max_norm: T) -> T
    where
        B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T> + coeus_core::CpuAddressableStorageMut<T>;
}
