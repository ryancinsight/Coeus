//! # Modality Support
//!
//! Core modality definitions and configurations for multimodal processing.
//! Provides extensible modality support with type-safe configuration.

use std::collections::HashMap;

/// Supported modalities in the multimodal system
///
/// Each modality represents a different type of input data that can be processed
/// by the multimodal transformer. The system is designed to be extensible,
/// allowing new modalities to be added through the `Custom` variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Modality {
    /// Vision modality for processing images and video data
    /// Typically uses pre-trained vision encoders like CLIP vision or ResNet
    Vision,
    /// Language modality for processing text data
    /// Uses transformer-based language models like BERT or GPT
    Language,
    /// Audio modality for processing speech and audio data
    /// Can use spectrogram-based or waveform-based audio encoders
    Audio,
    /// Custom modality for extending the system with new data types
    /// The string identifier allows for custom modality-specific processing
    Custom(String),
}

impl Modality {
    /// Get string representation for modality
    pub fn as_str(&self) -> &str {
        match self {
            Modality::Vision => "vision",
            Modality::Language => "language",
            Modality::Audio => "audio",
            Modality::Custom(s) => s,
        }
    }
}

/// Configuration for a modality-specific encoder
///
/// This struct defines all the parameters needed to configure an encoder
/// for a specific modality. Different modalities may require different
/// architectural choices (e.g., position embeddings for sequential data).
#[derive(Debug, Clone)]
pub struct ModalityConfig {
    /// The type of modality this encoder will process
    pub modality: Modality,
    /// Input dimensionality of the raw modality features
    /// (e.g., 768 for BERT embeddings, 2048 for CLIP vision features)
    pub input_dim: usize,
    /// Hidden dimension for internal representations
    /// Should match across modalities for cross-modal fusion
    pub hidden_dim: usize,
    /// Number of transformer layers in the encoder
    pub num_layers: usize,
    /// Number of attention heads in each transformer layer
    pub num_heads: usize,
    /// Maximum sequence length this encoder can handle
    /// Important for memory allocation and positional embeddings
    pub max_seq_len: usize,
    /// Dropout probability applied during training
    pub dropout: f64,
    /// Additional modality-specific parameters
    /// Can be used for custom configuration options
    pub params: HashMap<String, f64>,
}

impl Default for ModalityConfig {
    fn default() -> Self {
        Self {
            modality: Modality::Vision,
            input_dim: 768,
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            max_seq_len: 512,
            dropout: 0.1,
            params: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modality_config() {
        let config = ModalityConfig::default();
        assert_eq!(config.modality, Modality::Vision);
        assert_eq!(config.input_dim, 768);
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.max_seq_len, 512);
        assert_eq!(config.dropout, 0.1);
    }

    #[test]
    fn test_modality_as_str() {
        assert_eq!(Modality::Vision.as_str(), "vision");
        assert_eq!(Modality::Language.as_str(), "language");
        assert_eq!(Modality::Audio.as_str(), "audio");
        assert_eq!(Modality::Custom("test".to_string()).as_str(), "test");
    }
}












