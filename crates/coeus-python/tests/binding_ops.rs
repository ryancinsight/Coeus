//! Hierarchical integration harness for Python binding contract tests.
//!
//! The leaf modules retain their value-semantic binding assertions; one Cargo
//! target replaces the previous flat target-per-file topology. Python parity
//! scripts and the shared test lock module remain outside this Rust harness.

#[path = "binding_ops/activations.rs"]
mod activations;
#[path = "binding_ops/autodiff.rs"]
mod autodiff;
#[path = "common/mod.rs"]
mod common;
#[path = "binding_ops/distributed.rs"]
mod distributed;
#[path = "binding_ops/nn.rs"]
mod nn;
#[path = "binding_ops/operations.rs"]
mod operations;
#[path = "binding_ops/optim.rs"]
mod optim;
