//! Hierarchical integration harness for the standalone autograd operations.
//!
//! The existing `autograd_tests` target remains the canonical harness for its
//! larger module tree. These three leaf modules retain their original test
//! bodies while sharing one Cargo integration target.

#[path = "autograd_ops/grid_sample.rs"]
mod grid_sample;
#[path = "autograd_ops/interpolation.rs"]
mod interpolation;
#[path = "autograd_ops/scan.rs"]
mod scan;
