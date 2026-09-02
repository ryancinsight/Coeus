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
mod matmul;
mod pooling;
mod random_init;
mod reduction;
mod rotate_half;
mod stateful_update;
mod storage;
mod unfold_fold;

pub use attention::{AttentionBackend, AttentionProvider};
pub use convolution::{
    ConvolutionBackend, ConvolutionBackwardDispatch, ConvolutionForwardDispatch,
    ConvolutionProvider, regular_backward as convolution_backward,
    regular_forward as convolution_forward, transposed_backward as convolution_transposed_backward,
    transposed_forward as convolution_transposed_forward,
};
pub use cross_entropy::{
    CrossEntropyBackend, CrossEntropyProvider, prepare_candidate,
    prepare_targets as prepare_cross_entropy_targets,
};
pub use elementwise::{
    ActivationUnaryOperations, ArithmeticUnaryOperations, BinaryElementwiseDispatch,
    ElementwiseProvider, ParameterizedElementwiseProvider, ScalarPowerDispatch,
    ScalarPowerProvider, UnaryElementwiseDispatch, parameterized_unary,
};
pub use error::HephaestusBackendError;
pub use matmul::{MatmulBackend, MatmulProvider, matmul};
pub use pooling::{PoolingBackend, PoolingProvider, pooling_backward, pooling_forward};
pub use random_init::{RandomInitProvider, normal as random_normal, uniform as random_uniform};
pub use reduction::HephaestusBackend;
pub use reduction::{
    AxisReductionDispatch, HephaestusProvider, RankedOperand, ReductionProvider, ScanDispatch,
    ScanOperation,
};
pub use rotate_half::{RotateHalfProvider, rotate_half};
pub use stateful_update::{StatefulUpdateBackend, StatefulUpdateProvider};
pub use storage::HephaestusStorage;
pub use unfold_fold::{
    UnfoldFoldBackend, UnfoldFoldProvider, unfold_fold_fold, unfold_fold_unfold,
};
