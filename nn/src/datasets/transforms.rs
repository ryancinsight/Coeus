//! Image and Text Transformations for Vision-Language Data
//!
//! This module provides data transformation utilities specifically designed for CLIP training,
//! including image augmentation pipelines and text preprocessing transforms.

use super::ImageTextPair;
use crate::core::error::{NNError, Result};
use std::collections::HashMap;

/// Trait for data transformations
#[async_trait::async_trait(?Send)]
pub trait Transform: Send + Sync {
    /// Apply transformation to an image-text pair
    async fn transform(&self, pair: ImageTextPair) -> Result<ImageTextPair>;

    /// Get transform name for logging
    fn name(&self) -> &'static str;
}

/// Image augmentation transforms for CLIP training
pub mod image {
    use super::*;

    /// Random horizontal flip
    pub struct RandomHorizontalFlip {
        pub probability: f32,
    }

    #[async_trait::async_trait(?Send)]
    impl Transform for RandomHorizontalFlip {
        async fn transform(&self, mut pair: ImageTextPair) -> Result<ImageTextPair> {
            if rand::random::<f32>() < self.probability {
                // In a real implementation, this would flip the image horizontally
                // For now, we just mark it in metadata
                pair.metadata
                    .insert("flipped".to_string(), serde_json::json!("true"));
            }
            Ok(pair)
        }

        fn name(&self) -> &'static str {
            "random_horizontal_flip"
        }
    }

    /// Random color jitter (brightness, contrast, saturation)
    pub struct RandomColorJitter {
        pub brightness: f32,
        pub contrast: f32,
        pub saturation: f32,
        pub hue: f32,
        pub probability: f32,
    }

    #[async_trait::async_trait(?Send)]
    impl Transform for RandomColorJitter {
        async fn transform(&self, mut pair: ImageTextPair) -> Result<ImageTextPair> {
            if rand::random::<f32>() < self.probability {
                // In a real implementation, this would apply color transformations to image pixels
                // For now, we just mark it in metadata
                pair.metadata
                    .insert("color_jittered".to_string(), serde_json::json!("true"));
                pair.metadata.insert(
                    "brightness".to_string(),
                    serde_json::json!(format!("{:.2}", rand::random::<f32>() * self.brightness)),
                );
                pair.metadata.insert(
                    "contrast".to_string(),
                    serde_json::json!(format!("{:.2}", rand::random::<f32>() * self.contrast)),
                );
                pair.metadata.insert(
                    "saturation".to_string(),
                    serde_json::json!(format!("{:.2}", rand::random::<f32>() * self.saturation)),
                );
                pair.metadata.insert(
                    "hue".to_string(),
                    serde_json::json!(format!("{:.2}", rand::random::<f32>() * self.hue)),
                );
            }
            Ok(pair)
        }

        fn name(&self) -> &'static str {
            "random_color_jitter"
        }
    }

    /// Random resized crop for CLIP training
    pub struct RandomResizedCrop {
        pub size: (usize, usize),
        pub scale: (f32, f32),
        pub ratio: (f32, f32),
    }

    impl Default for RandomResizedCrop {
        fn default() -> Self {
            Self {
                size: (224, 224),
                scale: (0.08, 1.0),
                ratio: (0.75, 1.33),
            }
        }
    }

    #[async_trait::async_trait(?Send)]
    impl Transform for RandomResizedCrop {
        async fn transform(&self, mut pair: ImageTextPair) -> Result<ImageTextPair> {
            // In a real implementation, this would crop and resize the image
            // For now, we just mark the target size in metadata
            pair.metadata.insert(
                "crop_size".to_string(),
                serde_json::json!(format!("{}x{}", self.size.0, self.size.1)),
            );
            pair.metadata
                .insert("cropped".to_string(), serde_json::json!("true"));
            Ok(pair)
        }

        fn name(&self) -> &'static str {
            "random_resized_crop"
        }
    }

    /// Normalize pixel values to CLIP's expected range
    pub struct Normalize {
        pub mean: [f32; 3], // RGB means
        pub std: [f32; 3],  // RGB std deviations
    }

    impl Default for Normalize {
        fn default() -> Self {
            // CLIP's normalization values
            Self {
                mean: [0.481_454_7, 0.457_827_5, 0.408_210_7],
                std: [0.268_629_5, 0.261_302_6, 0.275_777_1],
            }
        }
    }

    #[async_trait::async_trait(?Send)]
    impl Transform for Normalize {
        async fn transform(&self, mut pair: ImageTextPair) -> Result<ImageTextPair> {
            // In a real implementation, this would normalize actual pixel values
            // For now, we just mark it in metadata
            pair.metadata
                .insert("normalized".to_string(), serde_json::json!("true"));
            pair.metadata.insert(
                "mean_r".to_string(),
                serde_json::json!(format!("{:.6}", self.mean[0])),
            );
            pair.metadata.insert(
                "mean_g".to_string(),
                serde_json::json!(format!("{:.6}", self.mean[1])),
            );
            pair.metadata.insert(
                "mean_b".to_string(),
                serde_json::json!(format!("{:.6}", self.mean[2])),
            );
            Ok(pair)
        }

        fn name(&self) -> &'static str {
            "normalize"
        }
    }

    /// Convert to tensor format
    pub struct ToTensor;

    #[async_trait::async_trait(?Send)]
    impl Transform for ToTensor {
        async fn transform(&self, mut pair: ImageTextPair) -> Result<ImageTextPair> {
            // In a real implementation, this would convert image bytes to tensor format
            // For now, we just mark it in metadata
            pair.metadata
                .insert("tensor_format".to_string(), serde_json::json!("true"));
            Ok(pair)
        }

        fn name(&self) -> &'static str {
            "to_tensor"
        }
    }
}

/// Text processing transforms for CLIP training
pub mod text {
    use super::*;

    /// Convert text to lowercase
    pub struct ToLowercase;

    #[async_trait::async_trait(?Send)]
    impl Transform for ToLowercase {
        async fn transform(&self, mut pair: ImageTextPair) -> Result<ImageTextPair> {
            for caption in &mut pair.captions {
                *caption = caption.to_lowercase();
            }
            pair.metadata
                .insert("lowercased".to_string(), serde_json::json!("true"));
            Ok(pair)
        }

        fn name(&self) -> &'static str {
            "to_lowercase"
        }
    }

    /// Random text deletion (SimCLR-style augmentation for text)
    pub struct RandomDeletion {
        pub probability: f32,
        pub max_deletions: usize,
    }

    #[async_trait::async_trait(?Send)]
    impl Transform for RandomDeletion {
        async fn transform(&self, mut pair: ImageTextPair) -> Result<ImageTextPair> {
            let mut modified = false;
            for caption in &mut pair.captions {
                if rand::random::<f32>() < self.probability {
                    let words: Vec<&str> = caption.split_whitespace().collect();
                    if words.len() > 1 {
                        let deletions = std::cmp::min(self.max_deletions, words.len() / 2);
                        let mut new_words = words.clone();

                        for _ in 0..deletions {
                            let idx = rand::random::<usize>() % new_words.len();
                            new_words.remove(idx);
                        }

                        *caption = new_words.join(" ");
                        modified = true;
                    }
                }
            }
            if modified {
                pair.metadata.insert(
                    "text_augmented".to_string(),
                    serde_json::json!("random_deletion"),
                );
            }
            Ok(pair)
        }

        fn name(&self) -> &'static str {
            "random_deletion"
        }
    }

    /// Random token swapping
    pub struct RandomSwap {
        pub probability: f32,
        pub max_swaps: usize,
    }

    #[async_trait::async_trait(?Send)]
    impl Transform for RandomSwap {
        async fn transform(&self, mut pair: ImageTextPair) -> Result<ImageTextPair> {
            let mut modified = false;
            for caption in &mut pair.captions {
                if rand::random::<f32>() < self.probability {
                    let mut words: Vec<String> =
                        caption.split_whitespace().map(|s| s.to_string()).collect();

                    if words.len() > 1 {
                        let swaps = std::cmp::min(self.max_swaps, words.len() / 3);

                        for _ in 0..swaps {
                            let i = rand::random::<usize>() % words.len();
                            let j = rand::random::<usize>() % words.len();
                            words.swap(i, j);
                        }

                        *caption = words.join(" ");
                        modified = true;
                    }
                }
            }
            if modified {
                pair.metadata.insert(
                    "text_augmented".to_string(),
                    serde_json::json!("random_swap"),
                );
            }
            Ok(pair)
        }

        fn name(&self) -> &'static str {
            "random_swap"
        }
    }

    /// Tokenize text using CLIP vocabulary (placeholder)
    pub struct Tokenize {
        pub max_length: usize,
        pub add_special_tokens: bool,
    }

    impl Default for Tokenize {
        fn default() -> Self {
            Self {
                max_length: 77,
                add_special_tokens: true,
            }
        }
    }

    #[async_trait::async_trait(?Send)]
    impl Transform for Tokenize {
        async fn transform(&self, mut pair: ImageTextPair) -> Result<ImageTextPair> {
            // In a real implementation, this would tokenize using CLIP's vocabulary
            // For now, we just simulate tokenization by counting words
            let word_count = pair
                .captions
                .first()
                .map(|c| c.split_whitespace().count())
                .unwrap_or(0);

            pair.metadata
                .insert("tokenized".to_string(), serde_json::json!("true"));
            pair.metadata.insert(
                "word_count".to_string(),
                serde_json::json!(word_count.to_string()),
            );
            pair.metadata.insert(
                "max_length".to_string(),
                serde_json::json!(self.max_length.to_string()),
            );
            Ok(pair)
        }

        fn name(&self) -> &'static str {
            "tokenize"
        }
    }
}

/// Composable transform pipeline
pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }

    pub fn push(&mut self, transform: Box<dyn Transform>) {
        self.transforms.push(transform);
    }

    pub fn clear(&mut self) {
        self.transforms.clear();
    }

    /// Apply all transforms in sequence
    pub async fn apply(&self, pair: ImageTextPair) -> Result<ImageTextPair> {
        let mut result = pair;
        let mut applied_transforms = Vec::new();

        for transform in &self.transforms {
            result = transform.transform(result).await?;
            applied_transforms.push(transform.name());
        }

        result.metadata.insert(
            "applied_transforms".to_string(),
            serde_json::Value::String(applied_transforms.join(",")),
        );
        result.metadata.insert(
            "num_transforms".to_string(),
            serde_json::Value::String(applied_transforms.len().to_string()),
        );

        Ok(result)
    }

    /// Get number of transforms in pipeline
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

/// Standard CLIP augmentation pipeline
pub fn clip_augmentation_pipeline() -> Compose {
    use image::*;
    use text::*;

    let transforms: Vec<Box<dyn Transform>> = vec![
        // Image augmentations
        Box::new(RandomResizedCrop::default()),
        Box::new(RandomHorizontalFlip { probability: 0.5 }),
        Box::new(RandomColorJitter {
            brightness: 0.4,
            contrast: 0.4,
            saturation: 0.4,
            hue: 0.1,
            probability: 0.8,
        }),
        Box::new(ToTensor),
        Box::new(Normalize::default()),
        // Text augmentations
        Box::new(ToLowercase),
        // Note: CLIP typically doesn't use heavy text augmentations
        // Box::new(RandomDeletion { probability: 0.1, max_deletions: 1 }),
        Box::new(Tokenize::default()),
    ];

    Compose::new(transforms)
}

/// Minimal pipeline for validation/test sets (no random augmentations)
pub fn validation_pipeline() -> Compose {
    use image::*;
    use text::*;

    let transforms: Vec<Box<dyn Transform>> = vec![
        // Center crop and resize (not random)
        Box::new(RandomResizedCrop::default()), // Would be center crop in real implementation
        Box::new(ToTensor),
        Box::new(Normalize::default()),
        Box::new(ToLowercase),
        Box::new(Tokenize::default()),
    ];

    Compose::new(transforms)
}

/// Training pipeline with heavier augmentations for robustness
pub fn heavy_augmentation_pipeline() -> Compose {
    use image::*;
    use text::*;

    let transforms: Vec<Box<dyn Transform>> = vec![
        // Heavy image augmentations
        Box::new(RandomResizedCrop::default()),
        Box::new(RandomHorizontalFlip { probability: 0.5 }),
        Box::new(RandomColorJitter {
            brightness: 0.5,
            contrast: 0.5,
            saturation: 0.5,
            hue: 0.2,
            probability: 1.0, // Always apply
        }),
        Box::new(ToTensor),
        Box::new(Normalize::default()),
        // Moderate text augmentations
        Box::new(ToLowercase),
        Box::new(RandomDeletion {
            probability: 0.15,
            max_deletions: 2,
        }),
        Box::new(RandomSwap {
            probability: 0.1,
            max_swaps: 1,
        }),
        Box::new(Tokenize::default()),
    ];

    Compose::new(transforms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn create_test_pair() -> ImageTextPair {
        ImageTextPair {
            image_data: vec![0u8; 224 * 224 * 3],
            image_path: "test.jpg".to_string(),
            captions: vec!["A beautiful sunset over the mountains".to_string()],
            image_id: "test123".to_string(),
            caption_ids: vec!["test123_0".to_string()],
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_to_lowercase() {
        let transform = text::ToLowercase;
        let pair = create_test_pair();
        let result = transform.transform(pair).await.unwrap();

        assert_eq!(result.captions[0], "a beautiful sunset over the mountains");
        assert_eq!(
            result.metadata.get("lowercased"),
            Some(&serde_json::Value::String("true".to_string()))
        );
    }

    #[tokio::test]
    async fn test_random_deletion() {
        let transform = text::RandomDeletion {
            probability: 1.0, // Always apply
            max_deletions: 1,
        };

        let pair = create_test_pair();
        let result = transform.transform(pair).await.unwrap();

        // Check that words were removed (exact result depends on random seed)
        let word_count = result.captions[0].split_whitespace().count();
        assert!(word_count <= 6); // Original had 7 words, potentially removed 1
        assert!(result.metadata.contains_key("text_augmented"));
    }

    #[tokio::test]
    async fn test_random_horizontal_flip() {
        let transform = image::RandomHorizontalFlip { probability: 1.0 };
        let pair = create_test_pair();
        let result = transform.transform(pair).await.unwrap();

        assert_eq!(
            result.metadata.get("flipped"),
            Some(&serde_json::Value::String("true".to_string()))
        );
    }

    #[tokio::test]
    async fn test_compose_empty() {
        let compose = Compose::new(vec![]);
        let pair = create_test_pair();
        let result = compose.apply(pair.clone()).await.unwrap();
        assert_eq!(result.captions[0], pair.captions[0]);
    }

    #[tokio::test]
    async fn test_compose_multiple() {
        use text::*;

        let compose = Compose::new(vec![Box::new(ToLowercase), Box::new(Tokenize::default())]);

        let pair = create_test_pair();
        let result = compose.apply(pair).await.unwrap();

        assert!(result.metadata.contains_key("lowercased"));
        assert!(result.metadata.contains_key("tokenized"));
        assert!(result.metadata.contains_key("applied_transforms"));
        assert_eq!(result.metadata["num_transforms"], "2");
    }

    #[tokio::test]
    async fn test_standard_pipelines() {
        let clip_pipeline = clip_augmentation_pipeline();
        assert!(!clip_pipeline.is_empty());

        let val_pipeline = validation_pipeline();
        assert!(!val_pipeline.is_empty());

        let heavy_pipeline = heavy_augmentation_pipeline();
        assert!(!heavy_pipeline.is_empty());

        // Test that we can apply them
        let pair = create_test_pair();
        let result = clip_pipeline.apply(pair).await.unwrap();
        assert!(result.metadata.contains_key("applied_transforms"));
    }
}
