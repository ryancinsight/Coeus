// ── Backend module ──
// Execution backend abstraction. Moirai for parallel, Sequential for fallback.

mod traits;
mod moirai;
mod sequential;

pub use traits::{Backend, ComputeBackend, private};
pub use moirai::MoiraiBackend;
pub use sequential::SequentialBackend;
