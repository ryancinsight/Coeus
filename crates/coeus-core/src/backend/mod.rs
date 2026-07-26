// ── Backend module ──
// Execution backend abstraction. Moirai for parallel, Sequential for fallback.

mod error;
mod moirai;
mod sequential;
mod traits;

#[cfg(test)]
mod tests_num_threads;

pub use error::BackendError;
pub use moirai::MoiraiBackend;
pub use sequential::SequentialBackend;
pub use traits::{private, Backend, ComputeBackend};
