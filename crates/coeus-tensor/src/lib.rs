// ── Coeus Tensor ──
// N-dimensional generic tensor over dtype, backend, and storage.
//! # coeus-tensor
//!
//! Generic N-dimensional tensor type with COW semantics, zero-copy views,
//! and backend-abstracted storage.
#![allow(clippy::needless_range_loop)]
#![deny(missing_docs)]

/// Broadcasting shape compatibility.
pub mod broadcast;
/// Factory constructors (zeros, ones, eye, linspace, arange, from_fn).
pub mod constructors;
/// Indexing and assignment operations.
pub mod indexing;
/// Iteration over tensor elements.
pub mod iter;
/// Core tensor type and accessors.
pub mod tensor;
/// Zero-copy view operations (slice, transpose, reshape, permute, broadcast).
pub mod views;

/// StateDict checkpointing for model parameter serialization.
pub mod checkpoint;

pub use checkpoint::{ArchivedTensor, StateArchive, StateDict, StateLimits};
pub use tensor::{Tensor, TensorTransferError};
pub use views::Transpose;
