//! Hierarchical integration harness for the flat Coeus-NN operation tests.
//!
//! The existing `nn_tests` target remains a separate harness for its legacy
//! module tree. These operation-family modules preserve the moved test bodies
//! while reducing the 33 flat binaries to one Cargo target.

#[path = "nn_ops/activations.rs"]
mod activations;
#[path = "nn_ops/attention.rs"]
mod attention;
#[path = "nn_ops/convolution.rs"]
mod convolution;
#[path = "nn_ops/embedding.rs"]
mod embedding;
#[path = "nn_ops/interpolation.rs"]
mod interpolation;
#[path = "nn_ops/losses.rs"]
mod losses;
#[path = "nn_ops/normalization.rs"]
mod normalization;
#[path = "nn_ops/pooling.rs"]
mod pooling;
#[path = "nn_ops/recurrent.rs"]
mod recurrent;
#[path = "nn_ops/tensor.rs"]
mod tensor;
