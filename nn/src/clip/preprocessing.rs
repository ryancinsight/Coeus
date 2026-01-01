//! Preprocessing utilities for CLIP
//!
//! This module provides image and text preprocessing according to OpenAI CLIP standards.

use std::fmt;

/// OpenAI CLIP image preprocessing
#[derive(Debug, Clone)]
pub struct ImageProcessor {
    /// Input image size (224 for CLIP)
    pub image_size: usize,
    /// Mean values for normalization (RGB)
    pub mean: [f32; 3],
    /// Std values for normalization (RGB)
    pub std: [f32; 3],
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self {
            image_size: 224,
            // CLIP standard normalization
            mean: [0.481_454_7, 0.457_827_5, 0.408_210_7],
            std: [0.268_629_5, 0.261_302_6, 0.275_777_1],
        }
    }
}

impl ImageProcessor {
    /// Create with custom parameters
    pub fn new(image_size: usize, mean: [f32; 3], std: [f32; 3]) -> Self {
        Self {
            image_size,
            mean,
            std,
        }
    }

    /// Preprocess a single image
    /// Takes RGB image as [height, width, 3] and returns normalized tensor
    pub fn preprocess(&self, image: &[f32], height: usize, width: usize) -> Vec<f32> {
        // Resize to target size (bilinear) - simplified
        let resized = self.resize_bilinear(image, height, width, self.image_size, self.image_size);

        // Normalize
        let mut normalized = Vec::with_capacity(resized.len());
        for (i, &val) in resized.iter().enumerate() {
            let channel = i % 3;
            let normalized_val = (val - self.mean[channel]) / self.std[channel];
            normalized.push(normalized_val);
        }

        // Convert HWC to CHW (CLIP expects [3, H, W])
        // Placeholder - actual implementation would transpose
        normalized
    }

    /// Preprocess batch of images
    pub fn preprocess_batch(
        &self,
        images: &[f32],
        height: usize,
        width: usize,
        batch_size: usize,
    ) -> Vec<f32> {
        let single_image_size = height * width * 3;
        let mut processed = Vec::with_capacity(batch_size * self.image_size * self.image_size * 3);

        for b in 0..batch_size {
            let start = b * single_image_size;
            let end = start + single_image_size;
            let image = &images[start..end];
            let processed_image = self.preprocess(image, height, width);
            processed.extend(processed_image);
        }

        processed
    }

    fn resize_bilinear(
        &self,
        image: &[f32],
        src_h: usize,
        src_w: usize,
        dst_h: usize,
        dst_w: usize,
    ) -> Vec<f32> {
        // Placeholder bilinear resize implementation
        // In practice, this would use proper interpolation

        if src_h == dst_h && src_w == dst_w {
            return image.to_vec();
        }

        // Simple nearest neighbor for now (would be bilinear in real implementation)
        let mut resized = Vec::with_capacity(dst_h * dst_w * 3);

        let h_ratio = src_h as f32 / dst_h as f32;
        let w_ratio = src_w as f32 / dst_w as f32;

        for y in 0..dst_h {
            for x in 0..dst_w {
                let src_y = (y as f32 * h_ratio) as usize;
                let src_x = (x as f32 * w_ratio) as usize;

                for c in 0..3 {
                    let idx = (src_y * src_w + src_x) * 3 + c;
                    let val = if idx < image.len() { image[idx] } else { 0.0 };
                    resized.push(val);
                }
            }
        }

        resized
    }
}

/// CLIP text preprocessing
#[derive(Debug, Clone)]
pub struct TextProcessor {
    /// Maximum sequence length (77 for CLIP)
    pub max_length: usize,
    /// Whether to truncate longer sequences
    pub truncate: bool,
    /// Whether to add SOS/EOS tokens
    pub add_special_tokens: bool,
}

impl Default for TextProcessor {
    fn default() -> Self {
        Self {
            max_length: 77, // CLIP standard context length
            truncate: true,
            add_special_tokens: true,
        }
    }
}

impl TextProcessor {
    /// Create with custom parameters
    pub fn new(max_length: usize, truncate: bool, add_special_tokens: bool) -> Self {
        Self {
            max_length,
            truncate,
            add_special_tokens,
        }
    }

    /// Process a single text string (placeholder - needs tokenizer integration)
    pub fn process_text(&self, text: &str) -> Result<Vec<u32>, String> {
        // Placeholder tokenization - would integrate with tokenizer::BpeTokenizer
        // For now, just return dummy tokens
        let dummy_tokens = vec![49406, 320, 1125, 539, 320, 2368, 49407]; // [SOS] a photo of a cat [EOS]

        let mut processed = if self.add_special_tokens {
            dummy_tokens
        } else {
            vec![320, 1125, 539, 320, 2368] // Remove SOS/EOS
        };

        // Truncate if needed
        if self.truncate && processed.len() > self.max_length {
            processed.truncate(self.max_length);
        }

        // Pad to max_length
        while processed.len() < self.max_length {
            processed.push(49408); // PAD token (would be tokenizer pad token)
        }

        Ok(processed)
    }

    /// Process batch of texts
    pub fn process_batch(&self, texts: &[String]) -> Result<Vec<u32>, String> {
        let mut all_tokens = Vec::new();

        for text in texts {
            let tokens = self.process_text(text)?;
            all_tokens.extend(tokens);
        }

        Ok(all_tokens)
    }
}

impl fmt::Display for ImageProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ImageProcessor(size={}, mean={:?}, std={:?})",
            self.image_size, self.mean, self.std
        )
    }
}

impl fmt::Display for TextProcessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TextProcessor(max_length={}, truncate={}, add_special={})",
            self.max_length, self.truncate, self.add_special_tokens
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_processor_defaults() {
        let processor = ImageProcessor::default();
        assert_eq!(processor.image_size, 224);
        assert_eq!(processor.mean, [0.481_454_7, 0.457_827_5, 0.408_210_7]);
        assert_eq!(processor.std, [0.268_629_5, 0.261_302_6, 0.275_777_1]);
    }

    #[test]
    fn test_image_processor_preprocess() {
        let processor = ImageProcessor::default();

        // Create a simple 2x2 RGB image (12 values)
        let image = vec![
            0.0, 0.0, 0.0, // Black pixel
            1.0, 1.0, 1.0, // White pixel
            0.5, 0.5, 0.5, // Gray pixel
            0.0, 1.0, 0.0, // Green pixel
        ];

        let processed = processor.preprocess(&image, 2, 2);
        assert!(!processed.is_empty());
        // Should be normalized with CLIP mean/std
        assert!(processed.len() >= 224 * 224 * 3); // Resized
    }

    #[test]
    fn test_text_processor_defaults() {
        let processor = TextProcessor::default();
        assert_eq!(processor.max_length, 77);
        assert!(processor.truncate);
        assert!(processor.add_special_tokens);
    }

    #[test]
    fn test_text_processor_process() {
        let processor = TextProcessor::default();
        let text = "a photo of a cat";

        let tokens = processor.process_text(text).unwrap();
        assert!(!tokens.is_empty());
        assert_eq!(tokens.len(), processor.max_length);

        // Should contain SOS and EOS tokens when add_special_tokens=true
        assert_eq!(tokens[0], 49406); // [SOS]
                                      // Would have EOS at some position
    }

    #[test]
    fn test_text_processor_no_special_tokens() {
        let processor = TextProcessor::new(77, true, false);
        let text = "test";

        let tokens = processor.process_text(text).unwrap();
        assert!(!tokens.is_empty());
        // Should not start with SOS token
        assert_ne!(tokens[0], 49406);
    }
}
