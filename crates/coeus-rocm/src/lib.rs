//! Coeus ROCm backend integration through the Hephaestus ROCm provider.
//!
//! This increment exposes the native rank-2 reduction and cumulative
//! scan/product contract. Unsupported ranks are returned as typed errors at
//! the Coeus layout boundary; no host fallback is used.
#![deny(missing_docs)]

mod backend;

pub use backend::{RocmBackend, RocmProvider};
