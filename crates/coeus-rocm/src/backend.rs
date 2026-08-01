//! Rocm backend implementations partitioned by provider, operation family,
//! and runtime boundary.

mod elementwise;
mod provider;
mod reduction;
mod runtime;
mod stateful_update;

pub use provider::RocmProvider;
pub use runtime::RocmBackend;
