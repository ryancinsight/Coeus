//! # coeus-sparse
//!
//! Pure sparse tensor data structures for Coeus: [`CooTensor`] (coordinate-list,
//! N-D) and [`CsrTensor`] (compressed-sparse-row, 2-D). Following DIP, these
//! types hold only shape, indices, and values — they carry no arithmetic.
//! Conversions (dense ↔ COO ↔ CSR) and sparse kernels (SpMV/SpMM and their
//! backward passes) live in `coeus-ops`/`coeus-leto`, keeping the data
//! structures backend-agnostic and isolation-testable.
//!
//! Both constructors validate their structural invariants (index rank/count,
//! 2-D CSR shape, `row_offsets` length); see `tests/sparse_tests.rs`.

#![deny(missing_docs)]

mod coo;
mod csr;

pub use coo::CooTensor;
pub use csr::CsrTensor;
