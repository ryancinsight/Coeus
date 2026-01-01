//! CLIP configuration and hyperparameters
//!
//! This module defines the configuration structures for CLIP models,
//! including vision and text encoder settings.

use serde::{Deserialize, Serialize};

/// Main CLIP model configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClipConfig {
    /// Embedding dimension (projected feature dimension)
    pub embed_dim: usize,
    /// Vision encoder configuration
    pub vision_config: VisionConfig,
    /// Text encoder configuration
    pub text_config: TextConfig,
    /// Projection dimension (512 for CLIP)
    pub projection_dim: usize,
    /// Temperature parameter for contrastive loss
    pub temperature: f64,
    /// Whether to cache text features during training
    pub cache_text_features: bool,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: Option<f64>,
}

impl Default for ClipConfig {
    fn default() -> Self {
        Self {
            embed_dim: 512,
            vision_config: VisionConfig::default(),
            text_config: TextConfig::default(),
            projection_dim: 512,
            temperature: 0.07, // Standard CLIP temperature
            cache_text_features: true,
            max_grad_norm: Some(1.0),
        }
    }
}

impl ClipConfig {
    /// Create ViT-B/32 configuration (matches OpenAI CLIP)
    pub fn vit_b32() -> Self {
        Self {
            embed_dim: 512,
            vision_config: VisionConfig::vit_b32(),
            text_config: TextConfig::default(),
            projection_dim: 512,
            temperature: 0.07,
            cache_text_features: true,
            max_grad_norm: Some(1.0),
        }
    }

    /// Create ViT-B/16 configuration
    pub fn vit_b16() -> Self {
        Self {
            embed_dim: 512,
            vision_config: VisionConfig::vit_b16(),
            text_config: TextConfig::default(),
            projection_dim: 512,
            temperature: 0.07,
            cache_text_features: true,
            max_grad_norm: Some(1.0),
        }
    }

    /// Create ViT-L/14 configuration
    pub fn vit_l14() -> Self {
        Self {
            embed_dim: 512,
            vision_config: VisionConfig::vit_l14(),
            text_config: TextConfig::default(),
            projection_dim: 512,
            temperature: 0.07,
            cache_text_features: true,
            max_grad_norm: Some(1.0),
        }
    }
}

/// Vision encoder configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisionConfig {
    /// Input image size (assumes square images)
    pub image_size: usize,
    /// Patch size for vision transformer
    pub patch_size: usize,
    /// Number of input channels (3 for RGB)
    pub num_channels: usize,
    /// Hidden dimension in transformer layers
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// MLP dimension in transformer layers
    pub mlp_dim: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Attention dropout probability
    pub attention_dropout: f64,
    /// Number of patches (computed automatically)
    pub num_patches: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self::vit_b32()
    }
}

impl VisionConfig {
    /// ViT-Base/32 configuration
    pub fn vit_b32() -> Self {
        Self {
            image_size: 224,
            patch_size: 32,
            num_channels: 3,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_dim: 3072,
            dropout: 0.1,
            attention_dropout: 0.1,
            num_patches: (224 / 32) * (224 / 32), // 49
        }
    }

    /// ViT-Base/16 configuration
    pub fn vit_b16() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_dim: 3072,
            dropout: 0.1,
            attention_dropout: 0.1,
            num_patches: (224 / 16) * (224 / 16), // 196
        }
    }

    /// ViT-Large/14 configuration
    pub fn vit_l14() -> Self {
        Self {
            image_size: 224,
            patch_size: 14,
            num_channels: 3,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            mlp_dim: 4096,
            dropout: 0.1,
            attention_dropout: 0.1,
            num_patches: (224 / 14) * (224 / 14), // 256
        }
    }

    /// Compute number of patches based on image and patch size
    pub fn compute_num_patches(image_size: usize, patch_size: usize) -> usize {
        (image_size / patch_size).pow(2)
    }
}

/// Text encoder configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextConfig {
    /// Vocabulary size for tokenizer
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Hidden dimension in transformer layers
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// MLP dimension in transformer layers
    pub mlp_dim: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Attention dropout probability
    pub attention_dropout: f64,
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 49408,           // CLIP's BPE vocabulary size
            max_position_embeddings: 77, // CLIP standard context length
            hidden_size: 512,
            num_layers: 12,
            num_heads: 8,
            mlp_dim: 2048,
            dropout: 0.1,
            attention_dropout: 0.1,
        }
    }
}

impl TextConfig {
    /// GPT-2 style configuration (alternative to CLIP default)
    pub fn gpt2_style() -> Self {
        Self {
            vocab_size: 50257, // GPT-2 vocabulary
            max_position_embeddings: 1024,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_dim: 3072,
            dropout: 0.1,
            attention_dropout: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_config_defaults() {
        let config = ClipConfig::default();
        assert_eq!(config.embed_dim, 512);
        assert_eq!(config.projection_dim, 512);
        assert_eq!(config.temperature, 0.07);
        assert!(config.cache_text_features);
        assert_eq!(config.max_grad_norm, Some(1.0));
    }

    #[test]
    fn test_vision_config_variants() {
        let b32 = VisionConfig::vit_b32();
        assert_eq!(b32.patch_size, 32);
        assert_eq!(b32.num_patches, 49);

        let b16 = VisionConfig::vit_b16();
        assert_eq!(b16.patch_size, 16);
        assert_eq!(b16.num_patches, 196);

        let l14 = VisionConfig::vit_l14();
        assert_eq!(l14.patch_size, 14);
        assert_eq!(l14.hidden_size, 1024);
        assert_eq!(l14.num_heads, 16);
    }

    #[test]
    fn test_clip_config_presets() {
        let b32 = ClipConfig::vit_b32();
        assert_eq!(b32.vision_config.patch_size, 32);
        assert_eq!(b32.vision_config.num_patches, 49);

        let b16 = ClipConfig::vit_b16();
        assert_eq!(b16.vision_config.patch_size, 16);
        assert_eq!(b16.vision_config.num_patches, 196);

        let l14 = ClipConfig::vit_l14();
        assert_eq!(l14.vision_config.patch_size, 14);
        assert_eq!(l14.vision_config.hidden_size, 1024);
    }
}
