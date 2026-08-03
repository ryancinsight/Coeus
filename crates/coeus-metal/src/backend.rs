//! Metal backend implementations partitioned by provider, operation family,
//! and runtime boundary.

mod elementwise;
mod provider;
mod random_init;
mod reduction;
mod runtime;
mod stateful_update;

pub use provider::MetalProvider;
pub use runtime::MetalBackend;
