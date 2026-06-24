// ── Shape manipulation operations ──
// Concatenation, splitting, padding, flipping, sorting, and conditional select.

mod concat;
mod flip;
mod index;
mod pad;
mod sort;
mod split;
mod stack;
mod where_cond;

pub use concat::cat;
pub use flip::flip;
pub use pad::pad;
pub use sort::sort;
pub use split::split;
pub use stack::stack;
pub use where_cond::where_cond;
