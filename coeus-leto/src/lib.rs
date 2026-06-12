#![forbid(unsafe_code)]
#![warn(missing_docs)]
//! # coeus-leto
//!
//! The const-rank dispatch shim that lets coeus delegate CPU array kernels to
//! [`leto`], per leto ADR 0002 (`docs/adr/0002-coeus-rank-boundary.md`).
//!
//! coeus carries a **dynamic-rank** [`coeus_core::Layout`] (rank held at
//! runtime in a `SmallVec`), while leto is **const-rank** (`Layout<const N>`),
//! which is the source of its compile-time shape safety and monomorphized,
//! allocation-free traversal. Rather than fork leto into a dynamic-rank model,
//! this crate resolves coeus's runtime rank to a leto `const N` through a
//! bounded `match` ([`dispatch`]) and calls the monomorphized leto kernel. The
//! shim lives here, in the consumer, so leto stays purely const-rank.
//!
//! This is the consolidation seam: coeus's CPU array operations route through
//! one authoritative leto kernel set instead of a duplicated traversal layer.

/// Zero-copy conversion from coeus dynamic-rank layouts to leto const-rank views.
pub mod convert;
/// Dynamic-rank to const-rank operation dispatch.
pub mod dispatch;

pub use convert::{to_leto_layout, to_leto_view, to_leto_view_mut};
pub use dispatch::{
    argmax_into, argmin_into, concat_values, contiguous_values, cumsum_into, elementwise_add_into,
    elementwise_binary_into, elementwise_unary_into, matmul_into, normal_values, pad_values,
    permute_layout, reduce_into, reshape_layout, split_values, suffix_sum_into, uniform_values,
    MAX_DISPATCH_RANK,
};
pub use leto_ops::RealScalar as RandomScalar;
