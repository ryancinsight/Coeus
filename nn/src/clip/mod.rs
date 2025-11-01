//! CLIP (Contrastive Language-Image Pretraining) implementation
//!
//! This module provides a complete implementation of CLIP, enabling:
//! - Vision-language understanding through contrastive learning
//! - Zero-shot classification and image-text retrieval
//! - Multi-modal embeddings for downstream tasks
//!
//! ## Architecture
//! CLIP consists of two encoders (vision and text) that are trained to predict
//! which images are paired with which texts through contrastive learning.
//!
//! ## Usage
//! ```rust
//! use nn::clip::{ClipModel, ClipConfig};
//!
//! let config = ClipConfig::default();
//! let model = ClipModel::new(config).unwrap();
//!
//! // Get embeddings for text and image
//! let text_embedding = model.encode_text(&["a photo of a cat"]).unwrap();
//! let image_embedding = model.encode_image(&image_tensor).unwrap();
//!
//! // Compute similarity
//! let similarity = text_embedding.dot(&image_embedding);
//! ```

pub mod config;
pub mod model;
pub mod loss;
pub mod preprocessing;
pub mod trainer;
pub mod enhanced_trainer;
pub mod zero_shot;
pub mod validation;

// Re-exports
pub use config::{ClipConfig, VisionConfig, TextConfig};
pub use model::ClipModel;
pub use loss::InfoNCELoss;
pub use preprocessing::{ImageProcessor, TextProcessor};
pub use trainer::ClipTrainer;
pub use enhanced_trainer::{EnhancedClipTrainer, EnhancedClipTrainingConfig};
pub use zero_shot::{
    ZeroShotClassifier, ZeroShotConfig, ClassificationResult, BatchClassificationResult
};
pub use validation::ZeroShotResults;

#[cfg(test)]
mod tests;
