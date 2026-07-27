//! Rocm backend implementations partitioned by provider, operation family,
//! and runtime boundary.

mod elementwise;
mod provider;
mod reduction;
mod runtime;

pub use provider::RocmProvider;
pub use runtime::RocmBackend;
