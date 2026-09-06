//! Generic Coeus staggered gradient/divergence dispatch through Hephaestus.
//!
//! The accelerator half of the finite-difference seam. Consumers bind
//! `coeus_ops::StaggeredPairOps` and reach either the CPU implementation over
//! Leto or, through this module, whichever Hephaestus provider a backend
//! selects — one call site, either device.

mod dispatch;
mod implementation;
mod provider;

pub use dispatch::{
    divergence as staggered_divergence, gradient as staggered_gradient, PreparedStaggeredPair,
};
pub use provider::{StaggeredBackend, StaggeredProvider};
