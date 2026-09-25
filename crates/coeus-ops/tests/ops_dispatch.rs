//! Integration harness for `coeus-ops` indexing and elementwise dispatch
//! differential tests.
//!
//! Split out of the former `tests/ops.rs`; see `ops_ownership.rs` for why
//! these two small families are grouped separately from ownership. No test
//! was removed or weakened.

#[path = "ops/elementwise.rs"]
mod elementwise;
#[path = "ops/indexing.rs"]
mod indexing;
