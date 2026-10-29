//! Python-exposed tensor class.
//!
//! [`class`] holds the struct, its construction path, and the in-place mutation
//! guard; [`dispatch`] holds the closed-set operator tags and the helper
//! methods that carry operands across the GIL boundary. Each `#[pymethods]`
//! section of the class is a further submodule ([`construct`], [`arithmetic`],
//! [`indexing`], ...), so PyO3 sees one class assembled from focused files
//! rather than a single 985-line block.

mod class;
mod dispatch;

pub use class::PyTensor;

mod arithmetic;
mod cast;
mod compare;
mod construct;
mod grad;
mod indexing;
mod inplace;
mod protocol;
mod reduction;
mod shape;
mod unary;
