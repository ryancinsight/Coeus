// ── Backend module ──
// Execution backend abstraction. Moirai for parallel, Sequential for fallback.

mod moirai;
mod sequential;
mod traits;

pub use moirai::MoiraiBackend;
pub use sequential::SequentialBackend;
pub use traits::{private, Backend, ComputeBackend};
