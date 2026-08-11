//! Coeus Metal backend integration through the Hephaestus Metal provider.
//!
//! The crate exposes the generic `HephaestusBackend<MetalProvider>` for
//! elementwise, scalar-power, axis-reduction, scan, random, rotate-half,
//! stateful-update, and cross-entropy dispatch through the shared
//! Coeus-Hephaestus bridge, with no host fallback for unsupported layouts.
#![deny(missing_docs)]

mod backend;

pub use backend::{HephaestusBackend, MetalProvider};
