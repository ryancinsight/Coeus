// ── Shape manipulation operations ──
// Concatenation, splitting, padding, flipping, sorting, gathering, and conditional select.

mod broadcast;
mod concat;
mod flip;
mod gather;
mod index;
mod masked_fill;
mod nonzero;
mod pad;
mod repeat_interleave;
mod roll;
mod scatter;
mod sort;
mod split;
mod stack;
mod tril;
mod where_cond;

pub use broadcast::broadcast_to;
pub use concat::cat;
pub use flip::flip;
pub use gather::gather;
pub use masked_fill::masked_fill;
pub use nonzero::nonzero;
pub use pad::pad;
pub use repeat_interleave::repeat_interleave;
pub use roll::roll;
pub use scatter::scatter_add;
pub use sort::sort;
pub use split::split;
pub use stack::stack;
pub use tril::{tril, triu};
pub use where_cond::where_cond;
