//! Hierarchical integration harness for the `coeus-ops` structural operation
//! families: convolution, dense linear algebra, shape transforms, and sparse
//! operations.
//!
//! Split from the single `tests/ops.rs` harness (see `ops_ownership.rs` for the
//! measured compile-memory rationale). No test was removed or weakened;
//! every leaf module below is unchanged from the prior harness.

#[path = "ops/convolution.rs"]
mod convolution;
#[path = "ops/linear_algebra.rs"]
mod linear_algebra;
#[path = "ops/shape.rs"]
mod shape;
#[path = "ops/sparse.rs"]
mod sparse;
