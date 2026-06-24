// ── Shape manipulation operations ──
// Concatenation, splitting, padding, flipping, sorting, gathering, and conditional select.

mod concat;
mod flip;
mod gather;
mod index;
mod pad;
mod repeat_interleave;
mod scatter;
mod sort;
mod split;
mod stack;
mod where_cond;

pub use concat::cat;
pub use flip::flip;
pub use gather::gather;
pub use pad::pad;
pub use repeat_interleave::repeat_interleave;
pub use scatter::scatter_add;
pub use sort::sort;
pub use split::split;
pub use stack::stack;
pub use where_cond::where_cond;
