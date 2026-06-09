// ── Coeus Tensor ──
// N-dimensional generic tensor over dtype, backend, and storage.
#![allow(clippy::needless_range_loop)]

pub mod broadcast;
pub mod constructors;
pub mod indexing;
pub mod iter;
pub mod tensor;
pub mod views;

pub mod checkpoint;

pub use checkpoint::StateDict;
pub use tensor::Tensor;
pub use views::Transpose;
