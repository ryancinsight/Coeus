//! Foundation Model Training Infrastructure (Sprint MS-45)
//!
//! This module provides comprehensive foundation model training capabilities
//! supporting transformers, distributed training, memory optimization, and
//! advanced parallelism strategies for training large language and vision models.

// Re-export the main module
// pub use crate::*;

// Error handling
pub mod error;
pub use error::{NNError, Result};

// Core training infrastructure
pub mod trainer;
pub mod training;

// Transformer architectures
pub mod transformers;

// Distributed training
pub mod distributed;

// Memory optimization
pub mod memory;

// Parallelism strategies
// pub mod parallelism;

// Training optimization
pub mod optimization;

// Performance monitoring
pub mod monitoring;

// Data loading and processing
pub mod data;
