// ── Shape manipulation operations ──
// Concatenation, splitting, padding, flipping, sorting, gathering, and conditional select.

mod broadcast;
mod chunk;
mod concat;
mod diag;
mod einsum;
mod flip;
mod gather;
mod index;
mod index_select;
mod masked_fill;
mod meshgrid;
mod nonzero;
mod pad;
mod repeat_interleave;
mod roll;
mod scatter;
mod selection;
mod sort;
mod split;
mod stack;
mod tile;
mod tril;
mod where_cond;

pub use broadcast::broadcast_to;
pub use chunk::chunk;
pub use concat::cat;
pub use diag::{diag, diagonal};
pub use einsum::einsum;
pub use flip::flip;
pub use gather::gather;
pub use index_select::index_select;
pub use masked_fill::masked_fill;
pub use meshgrid::meshgrid;
pub use nonzero::nonzero;
pub use pad::pad;
pub use repeat_interleave::repeat_interleave;
pub use roll::roll;
pub use scatter::scatter_add;
pub use selection::{masked_select, one_hot};
pub use sort::sort;
pub use split::split;
pub use stack::stack;
pub use tile::tile;
pub use tril::{tril, triu};
pub use where_cond::where_cond;
