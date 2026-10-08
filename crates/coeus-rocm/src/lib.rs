//! Coeus ROCm backend integration through the Hephaestus ROCm provider.
//!
//! The crate exposes the generic `HephaestusBackend<RocmProvider>` for
//! elementwise, scalar-power, axis-reduction, scan, random, rotate-half,
//! stateful-update, cross-entropy, staggered, fixed-scheme finite-difference,
//! and CTC dispatch through the shared Coeus-Hephaestus bridge. Unsupported
//! ranks are returned as typed errors at the Coeus layout boundary; no host
//! fallback is used.
#![deny(missing_docs)]

mod backend;

pub use backend::{HephaestusBackend, RocmProvider};
