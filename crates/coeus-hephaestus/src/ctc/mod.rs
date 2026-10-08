//! Coeus dispatch for the provider CTC loss seam.
//!
//! Forwards assemble the shared [`CtcProblem`](hephaestus_core::CtcProblem),
//! run the provider forward, write the host-side loss scalar, and return
//! the retained device state; backwards read the seed and reachability
//! from the device before touching the gradient, so an impossible
//! alignment errors exactly as the CPU implementation does.

mod dispatch;
mod implementation;
mod provider;

pub use dispatch::{ctc_backward, ctc_forward};
pub use provider::{CtcBackend, CtcProvider};
