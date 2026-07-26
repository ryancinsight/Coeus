//! Generic Coeus integration for Hephaestus device providers.
//!
//! This crate owns storage, transfer, layout validation, and the Coeus
//! reduction/scan dispatch contract once. Vendor crates implement
//! [`HephaestusProvider`] and the scalar-specific [`ReductionProvider`] seam;
//! they do not copy the consumer-side operation orchestration.
#![deny(missing_docs)]

mod elementwise;
mod error;
mod layout;
mod reduction;
mod storage;

pub use elementwise::ElementwiseProvider;
pub use error::HephaestusBackendError;
pub use reduction::HephaestusBackend;
pub use reduction::{HephaestusProvider, RankedOperand, ReductionProvider, ScanOperation};
pub use storage::HephaestusStorage;
