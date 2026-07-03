// ── Shape manipulation operations ──
// Concatenation, splitting, padding, flipping, sorting, gathering, and conditional select.

mod concat_split_stack;
mod einsum;
mod mask;
mod select;
mod transform;
mod util;

pub(crate) use select::flat_to_nd;

pub use concat_split_stack::cat;
pub use concat_split_stack::split;
pub use concat_split_stack::stack;
pub use einsum::{einsum, einsum3};
pub use mask::masked_fill;
pub use select::gather;
pub use select::index_put;
pub use select::index_select;
pub use select::scatter_add;
pub use select::{masked_select, one_hot};
pub use transform::broadcast_to;
pub use transform::chunk;
pub use transform::flip;
pub use transform::pad;
pub use transform::repeat_interleave;
pub use transform::roll;
pub use transform::tile;
pub use transform::where_cond;
pub use transform::{diag, diagonal};
pub use transform::{tril, triu};
pub use util::meshgrid;
pub use util::nonzero;
pub use util::sort;
