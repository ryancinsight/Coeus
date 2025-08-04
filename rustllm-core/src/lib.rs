//! # `RustLLM` Core
//!
//! A minimal dependency, high-performance Large Language Model (LLM) building library
//! written in pure Rust with zero external dependencies.
//!
//! ## Design Philosophy
//!
//! This library follows elite programming practices including:
//! - **SOLID**: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion
//! - **CUPID**: Composable, Unix Philosophy, Predictable, Idiomatic, Domain-based
//! - **GRASP**: General Responsibility Assignment Software Patterns
//! - **KISS**: Keep It Simple, Stupid
//! - **DRY**: Don't Repeat Yourself
//! - **YAGNI**: You Aren't Gonna Need It
//!
//! ## Architecture
//!
//! The library is organized into three main layers:
//!
//! 1. **Foundation Layer**: Core utilities, error handling, and memory management
//! 2. **Core API Layer**: Traits and abstractions for extensibility
//! 3. **Plugin Layer**: Extensible implementations via the plugin system
//!
//! ## Zero Dependencies
//!
//! This core library has absolutely zero external dependencies, ensuring:
//! - Fast compilation times
//! - Minimal attack surface
//! - Maximum portability
//! - Complete control over the codebase
//!
//! ## Example
//!
//! ```rust,ignore
//! use rustllm_core::prelude::*;
//! 
//! // Create a tokenizer
//! let tokenizer = BasicTokenizer::new();
//! 
//! // Tokenize text
//! let tokens: Vec<_> = tokenizer
//!     .tokenize_str("Hello, world!")
//!     .filter(|token| token.as_str().map(|s| s.len() > 3).unwrap_or(false))
//!     .collect();
//! 
//! // Build a model
//! let config = BasicModelConfig::default();
//! let model = builder.build(config)?;
//! 
//! // Process tokens through the model
//! let output = model.forward(tokens)?;
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    missing_docs,
    rustdoc::all
)]
#![allow(
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    unsafe_code // Allow unsafe code for memory management
)]

//! # `RustLLM` Core
//!
//! A minimal dependency, high-performance Large Language Model (LLM) building
//! library written in pure Rust.

// Required for no_std builds
#[cfg(not(feature = "std"))]
extern crate alloc;

// Foundation layer modules
pub mod foundation {
    //! Foundation layer providing core utilities and abstractions.
    
    pub mod error;
    pub mod iterator;
    pub mod memory;
    pub mod types;
}

// Core API layer modules
pub mod core {
    //! Core API layer defining traits and abstractions.
    pub mod config;
    pub mod model;
    pub mod plugin;
    pub mod tokenizer;
    pub mod traits;
    pub mod serialization;
}

// Plugin system module
pub mod plugins {
    //! Plugin system for extensibility.
    
    pub mod manager;
    pub mod registry;
}

// Re-exports for convenience
pub mod prelude {
    //! Common imports for users of the library.
    pub use crate::core::{
        config::{ConfigValue, ConfigBuilder, ConfigStore, PluginConfigManager},
        model::{
            Model, ModelBuilder, ModelConfig, BasicModelConfig, Transformer250MConfig,
            InferenceModel, GenerativeModel, TrainableModel, DiffusionModel,
            OptimizerState, Loss, NoiseSchedule, DiffusionSampler,
            GenerationConfig, BasicGenerationConfig,
        },
        plugin::Plugin,
        serialization::{ModelSerializable, ModelHeader, ModelMetadata, ParameterSerializer, calculate_checksum},
        tokenizer::{Token, Tokenizer},
        traits::*,
    };
    pub use crate::foundation::{
        error::{Error, Result},
        iterator::*,
        memory::{Arena, CowStr, ZeroCopyStringBuilder, SliceView, LazyAlloc},
        types::*,
    };
    pub use crate::plugins::manager::PluginManager;
}

// Version information
/// The version of the RustLLM Core library.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The minimum supported Rust version.
pub const MSRV: &str = "1.70.0";
