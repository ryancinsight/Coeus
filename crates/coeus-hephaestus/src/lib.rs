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
    regular_backward as convolution_backward, regular_forward as convolution_forward,
    transposed_backward as convolution_transposed_backward,
    transposed_forward as convolution_transposed_forward, ConvolutionBackend,
    ConvolutionBackwardDispatch, ConvolutionForwardDispatch, ConvolutionProvider,
};
pub use cross_entropy::{
    prepare_candidate, prepare_targets as prepare_cross_entropy_targets, CrossEntropyBackend,
    CrossEntropyProvider,
};
pub use elementwise::{
    parameterized_unary, ActivationUnaryOperations, ArithmeticUnaryOperations,
    BinaryElementwiseDispatch, ElementwiseProvider, ParameterizedElementwiseProvider,
    ScalarPowerDispatch, ScalarPowerProvider, UnaryElementwiseDispatch,
};
pub use error::HephaestusBackendError;
pub use matmul::{matmul, MatmulBackend, MatmulProvider};
pub use pooling::{pooling_backward, pooling_forward, PoolingBackend, PoolingProvider};
pub use random_init::{normal as random_normal, uniform as random_uniform, RandomInitProvider};
pub use reduction::HephaestusBackend;
pub use reduction::{
    AxisReductionDispatch, HephaestusProvider, RankedOperand, ReductionProvider, ScanDispatch,
    ScanOperation,
};
pub use rotate_half::{rotate_half, RotateHalfProvider};
pub use stateful_update::{StatefulUpdateBackend, StatefulUpdateProvider};
pub use storage::HephaestusStorage;
pub use unfold_fold::{
    unfold_fold_fold, unfold_fold_unfold, UnfoldFoldBackend, UnfoldFoldProvider,
};
