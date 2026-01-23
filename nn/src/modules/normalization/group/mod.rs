//! Group and Instance Normalization layers.

pub mod instance;

#[path = "group.rs"]
pub mod group_norm;

pub use group_norm::GroupNorm;
pub use instance::InstanceNorm;

/// Type aliases for CPU backend
pub type GroupNormCpu<T> =
    group_norm::GroupNorm<backend::CpuBackend<T>, storage::DenseStorage<T>, T>;
pub type InstanceNormCpu<T> =
    instance::InstanceNorm<backend::CpuBackend<T>, storage::DenseStorage<T>, T>;
