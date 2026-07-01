/// Concatenation, splitting, and stacking operations.
pub mod cat_split_stack;
/// Masking operations (tril, triu, masked_fill, where).
pub mod mask;
/// Selection and indexing operations (gather, index_select, slice, flip).
pub mod select;
/// Shape transformation operations (reshape, permute, pad, tile, roll, etc.).
pub mod transform;
/// Utility operations (cumsum, cumprod, contiguous, broadcast_to).
pub mod util;

pub use cat_split_stack::{cat, split, stack};
pub use mask::masked_fill;
pub use select::{gather, index_select};
pub use transform::{
    broadcast_to, diag, diagonal, flatten, flip, movedim, pad, permute, reshape, roll, slice,
    squeeze, swapaxes, tile, transpose, tril, triu, unsqueeze, where_cond,
};
pub use util::{contiguous, cumprod, cumsum, einsum, einsum3};
