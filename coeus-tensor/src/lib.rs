// ── Coeus Tensor ──
// N-dimensional generic tensor over dtype, backend, and storage.
#![allow(clippy::needless_range_loop)]

pub mod tensor;
pub mod constructors;
pub mod indexing;
pub mod views;
pub mod iter;
pub mod broadcast;

pub mod checkpoint;

pub use tensor::Tensor;
pub use views::Transpose;
pub use checkpoint::StateDict;
