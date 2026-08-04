mod dispatch;
mod implementation;
mod provider;

pub use implementation::prepare_targets;
pub use provider::{prepare_candidate, CrossEntropyBackend, CrossEntropyProvider};
