//! Hierarchical integration harness for tensor construction and layout tests.
//!
//! The leaf modules retain their original property and value-semantic tests;
//! one Cargo target replaces the previous flat target-per-file topology.

#[path = "tensor_ops/backend.rs"]
mod backend;
#[path = "tensor_ops/checkpoint.rs"]
mod checkpoint;
#[path = "tensor_ops/constructors.rs"]
mod constructors;
#[path = "tensor_ops/layout.rs"]
mod layout;
#[path = "tensor_ops/operations.rs"]
mod operations;
#[path = "tensor_ops/properties.rs"]
mod properties;
