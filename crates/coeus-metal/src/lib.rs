//! Coeus Metal backend integration through the Hephaestus Metal provider.
//!
//! The backend currently exposes the native rank-2 reduction and cumulative
//! scan/product contract. It uses the shared Coeus-Hephaestus storage and
//! dispatch layer, with no host fallback for unsupported layouts.
#![deny(missing_docs)]

mod backend;

pub use backend::{MetalBackend, MetalProvider};
