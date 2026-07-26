//! Hierarchical integration harness for sparse conversion and operation tests.
//!
//! The leaf modules retain their original value-semantic and differential
//! assertions; one Cargo target replaces the previous flat target-per-file
//! topology.

#[path = "sparse_ops/conversions.rs"]
mod conversions;
#[path = "sparse_ops/differential.rs"]
mod differential;
#[path = "sparse_ops/invariants.rs"]
mod invariants;
