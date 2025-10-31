//! # Multimodal Neural Networks
//!
//! This module implements multimodal architectures that can jointly process
//! multiple modalities (vision, language, audio) in a unified framework, enabling
//! cross-modal understanding and generation.
//!
//! ## Architecture Overview
//!
//! The multimodal transformer consists of several key components:
//!
//! - **Encoders**: Specialized encoders for each modality (vision, language, audio)
//! - **Attention**: Attention mechanisms that allow different modalities to attend to each other
//! - **Fusion Strategies**: Various approaches for combining information across modalities
//! - **Task Outputs**: Specialized outputs for different downstream tasks
//!
//! ## Key Features
//!
//! - **Extensible Modality Support**: Addition of new modalities through the `Modality` enum
//! - **Flexible Fusion**: Multiple fusion strategies including early, late, hierarchical, and attention-based fusion
//! - **Cross-Modal Understanding**: Bidirectional attention between all modality pairs
//! - **Task-Aware Processing**: Specialized outputs for classification, regression, generation, and retrieval
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use nn::multimodal::{MultimodalTransformer, MultimodalConfig, Modality, Task};
//! use backend::CpuBackend;
//! use dtype::Float32;
//!
//! // Configure multimodal transformer
//! let config = MultimodalConfig {
//!     modalities: vec![Modality::Vision, Modality::Language],
//!     hidden_dim: 768,
//!     num_fusion_layers: 6,
//!     fusion_strategy: FusionStrategy::HierarchicalFusion,
//!     dropout: 0.1,
//! };
//!
//! // Create transformer
//! let mut transformer = MultimodalTransformer::<CpuBackend<Float32>, _, Float32>::new(config)?;
//!
//! // Add classification task
//! let classification_task = Task::Classification(Classifier::new(768, 10)?);
//! transformer.add_task("classification".to_string(), classification_task)?;
//!
//! // Process multimodal inputs
//! let mut inputs = HashMap::new();
//! inputs.insert(Modality::Vision, vision_tensor);
//! inputs.insert(Modality::Language, language_tensor);
//!
//! let output = transformer.forward(&inputs, "classification", None)?;
//! ```
//!
//! ## Implementation Details
//!
//! - **Zero-Copy Operations**: Efficient tensor operations with minimal allocations
//! - **Generic Backend Support**: Works with any backend implementation (CPU, GPU, etc.)
//! - **Memory Safety**: Full Rust ownership and borrowing guarantees
//! - **Performance Optimized**: SIMD acceleration and parallel processing support

pub mod modality;
pub mod attention;
pub mod fusion;
pub mod encoder;
pub mod task;
pub mod transformer;

// Re-export main types for convenience
pub use modality::{Modality, ModalityConfig};
pub use attention::CrossModalAttention;
pub use fusion::{FusionStrategy, Fusion, FusionLayer, FusionBlock, FeedForward};
pub use encoder::{Encoder, Layer};
pub use task::{Task, Classifier, Generator, Retriever, SimilarityType};
pub use transformer::{MultimodalTransformer, MultimodalConfig};


