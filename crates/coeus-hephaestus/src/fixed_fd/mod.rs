//! Generic Coeus fixed-scheme sweep dispatch through Hephaestus.
//!
//! The accelerator half of the fixed-scheme finite-difference seam.
//! Consumers bind `coeus_ops::FiniteDifference3DOps` and reach either the CPU
//! implementation over Leto or, through this module, whichever Hephaestus
//! provider a backend selects — one call site, either device.

mod dispatch;
mod implementation;
mod provider;

pub use dispatch::{adjoint as fixed_fd_adjoint, sweep as fixed_fd_sweep};
pub use provider::{FixedFdBackend, FixedFdProvider};
