// ── Shape manipulation operations ──
// Concatenation, splitting, and padding of tensors.

mod concat;
mod split;
mod pad;

pub use concat::cat;
pub use split::split;
pub use pad::pad;
