//! Hierarchical integration harness for the standalone autograd operations.
//!
//! The established autograd module tree and the newer operation-family modules
//! share one Cargo integration target so their generic tensor and backward
//! kernels are monomorphized once per scalar/backend pair.

#![expect(
    clippy::unwrap_used,
    reason = "test/bench/example code asserts and reports by design"
)]

#[path = "autograd/mod.rs"]
mod autograd;

#[path = "autograd_ops/grid_sample.rs"]
mod grid_sample;
#[path = "autograd_ops/interpolation.rs"]
mod interpolation;
#[path = "autograd_ops/rotate_half.rs"]
mod rotate_half;
#[path = "autograd_ops/scan.rs"]
mod scan;

#[path = "autograd_ops/pow_exponent_contract.rs"]
mod pow_exponent_contract;
#[path = "autograd_ops/pow_negative_contract.rs"]
mod pow_negative_contract;
