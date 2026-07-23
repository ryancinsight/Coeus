//! Hierarchical integration harness for the standalone autograd operations.
//!
//! The established autograd module tree and the newer operation-family modules
//! share one Cargo integration target so the package has one canonical test
//! binary and one ownership boundary.

#[path = "autograd/mod.rs"]
mod autograd;

#[path = "autograd_ops/grid_sample.rs"]
mod grid_sample;
#[path = "autograd_ops/interpolation.rs"]
mod interpolation;
#[path = "autograd_ops/scan.rs"]
mod scan;
