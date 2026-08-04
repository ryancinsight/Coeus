//! Generic Coeus integration for Hephaestus device providers.
//!
//! This crate owns storage, transfer, layout validation, and Coeus dispatch
//! orchestration once. Vendor crates implement [`HephaestusProvider`] plus the
//! operation-specific [`ReductionProvider`], [`AttentionProvider`], and
//! [`AttentionBackend`] seams; they do not copy consumer-side request assembly.
#![deny(missing_docs)]

mod attention;
mod convolution;
mod cross_entropy;
mod elementwise;
mod error;
mod layout;
mod random_init;
mod reduction;
mod rotate_half;
mod stateful_update;
mod storage;

pub use attention::{AttentionBackend, AttentionProvider};
pub use convolution::{
    regular_backward as convolution_backward, regular_forward as convolution_forward,
    transposed_backward as convolution_transposed_backward,
    transposed_forward as convolution_transposed_forward, ConvolutionBackend,
    ConvolutionBackwardDispatch, ConvolutionForwardDispatch, ConvolutionProvider,
};
pub use cross_entropy::{
    prepare_targets as prepare_cross_entropy_targets, CrossEntropyBackend, CrossEntropyProvider,
};
pub use elementwise::{parameterized_unary, ElementwiseProvider, ParameterizedElementwiseProvider};
pub use error::HephaestusBackendError;
pub use random_init::{normal as random_normal, uniform as random_uniform, RandomInitProvider};
pub use reduction::HephaestusBackend;
pub use reduction::{HephaestusProvider, RankedOperand, ReductionProvider, ScanOperation};
pub use rotate_half::{rotate_half, RotateHalfProvider};
pub use stateful_update::{StatefulUpdateBackend, StatefulUpdateProvider};
pub use storage::HephaestusStorage;
