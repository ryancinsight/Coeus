//! Probability distributions and sampling utilities for Coeus.
//!
//! This crate provides small, validated distribution specifications suitable for
//! hyperparameter search and research workflows. Public constructors enforce
//! invariants; deserialization rejects invalid parameters.

#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

/// Error types for this crate.
pub mod error;
/// Validated distribution specifications for sampling.
pub mod parameter;

pub use error::{Error, Result};
pub use parameter::{FiniteF64, NonNegativeFiniteF64, ParameterDistribution, PositiveFiniteF64};
