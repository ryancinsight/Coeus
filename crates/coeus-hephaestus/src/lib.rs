//! Generic Coeus integration for Hephaestus device providers.
//!
//! This crate owns storage, transfer, layout validation, and Coeus dispatch
//! orchestration once. Vendor crates implement [`HephaestusProvider`] plus the
//! operation-specific [`ReductionProvider`], [`AttentionProvider`], and
//! [`AttentionBackend`] seams; they do not copy consumer-side request assembly.
#![deny(missing_docs)]

mod attention;
mod convolution;
mod elementwise;
mod error;
mod layout;
mod reduction;
mod storage;

pub use attention::{AttentionBackend, AttentionProvider};
pub use convolution::{
    regular_backward as convolution_backward, regular_forward as convolution_forward,
    transposed_backward as convolution_transposed_backward,
    transposed_forward as convolution_transposed_forward, ConvolutionBackend,
    ConvolutionBackwardDispatch, ConvolutionForwardDispatch, ConvolutionProvider,
};
pub use elementwise::ElementwiseProvider;
pub use error::HephaestusBackendError;
pub use reduction::HephaestusBackend;
pub use reduction::{HephaestusProvider, RankedOperand, ReductionProvider, ScanOperation};
pub use storage::HephaestusStorage;
