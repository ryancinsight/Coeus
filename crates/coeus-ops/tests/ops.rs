//! Hierarchical integration harness for the `coeus-ops` operation families.
//!
//! The leaf modules retain their original test bodies and assertions. This
//! single Cargo target keeps test-process isolation at the Nextest function
//! level while removing one full-link binary per operation file.

#[path = "ops/activations.rs"]
mod activations;
#[path = "ops/construction.rs"]
mod construction;
#[path = "ops/convolution.rs"]
mod convolution;
#[path = "ops/elementwise.rs"]
mod elementwise;
#[path = "ops/indexing.rs"]
mod indexing;
#[path = "ops/linear_algebra.rs"]
mod linear_algebra;
#[path = "ops/reductions.rs"]
mod reductions;
#[path = "ops/shape.rs"]
mod shape;
#[path = "ops/sparse.rs"]
mod sparse;
#[path = "ops/tensor.rs"]
mod tensor;
