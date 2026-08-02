//! Hierarchical integration harness for the flat Coeus-NN operation tests.
//!
//! The established NN module tree and the operation-family modules share one
//! Cargo integration target so the package has one canonical test binary and
//! one ownership boundary.

#[path = "nn/mod.rs"]
mod nn;

#[path = "nn_ops/activations.rs"]
mod activations;
#[path = "nn_ops/attention.rs"]
mod attention;
#[path = "nn_ops/convolution.rs"]
mod convolution;
#[path = "nn_ops/embedding.rs"]
mod embedding;
#[path = "nn_ops/evidence_manifest.rs"]
mod evidence_manifest;
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
