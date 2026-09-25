//! Hierarchical integration harness for the `coeus-ops` numeric operation
//! families: reductions and scans, construction, activations, and
//! tensor-level miscellaneous/normalization operations.
//!
//! Split from the single `tests/ops.rs` harness (see `ops_ownership.rs` for the
//! measured compile-memory rationale). No test was removed or weakened;
//! every leaf module below is unchanged from the prior harness.

#[path = "ops/activations.rs"]
mod activations;
#[path = "ops/construction.rs"]
mod construction;
#[path = "ops/reductions.rs"]
mod reductions;
#[path = "ops/tensor.rs"]
mod tensor;
