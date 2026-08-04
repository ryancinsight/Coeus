//! Rocm backend implementations partitioned by provider, operation family,
//! and runtime boundary.

mod cross_entropy;
mod elementwise;
mod provider;
mod random_init;
mod reduction;
#[cfg(all(feature = "rocm", target_os = "linux"))]
mod rotate_half;
mod runtime;
mod stateful_update;

pub use provider::RocmProvider;
pub use runtime::RocmBackend;
