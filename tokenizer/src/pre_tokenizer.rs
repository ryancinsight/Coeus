//! Pre-tokenization pipeline for text preprocessing.

use crate::error::{Result, TokenizerError};
use unicode_normalization::UnicodeNormalization;
use unicode_segmentation::UnicodeSegmentation;

/// Pre-tokenization trait for text preprocessing.
///
/// Pre-tokenizers handle Unicode normalization, whitespace handling,
/// and initial text segmentation before tokenization.
pub trait PreTokenizer {
    /// Pre-process text into initial token segments.
    ///
    /// # Arguments
    /// * `text` - Raw input text
    ///
    /// # Returns
    /// Vector of pre-tokenized segments
    ///
    /// # Errors
    /// Returns `TokenizerError` if preprocessing fails
    fn pre_tokenize(&self, text: &str) -> Result<Vec<String>>;
}

/// Basic pre-tokenizer with Unicode normalization and whitespace handling.
#[derive(Debug, Clone)]
pub struct BasicPreTokenizer {
    /// Whether to normalize Unicode (NFC by default).
    pub normalize: bool,
    /// Whether to convert to lowercase.
    pub lowercase: bool,
    /// Whether to strip leading/trailing whitespace.
    pub strip_accents: bool,
}

impl BasicPreTokenizer {
    /// Create a new basic pre-tokenizer with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            normalize: true,
            lowercase: false,
            strip_accents: false,
        }
    }

    /// Create a pre-tokenizer with Unicode normalization.
    #[must_use]
    pub fn with_normalization() -> Self {
        Self::new()
    }

    /// Create a pre-tokenizer with lowercase conversion.
    #[must_use]
    pub fn with_lowercase() -> Self {
        Self {
            normalize: true,
            lowercase: true,
            strip_accents: false,
        }
    }
}

impl Default for BasicPreTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PreTokenizer for BasicPreTokenizer {
    fn pre_tokenize(&self, text: &str) -> Result<Vec<String>> {
        let mut processed = text.to_string();

        // Unicode normalization
        if self.normalize {
            processed = processed.nfc().collect();
        }

        // Lowercase conversion
        if self.lowercase {
            processed = processed.to_lowercase();
        }

        // Strip accents if requested (basic implementation)
        if self.strip_accents {
            processed = strip_accents(&processed);
        }

        // Split on whitespace and filter empty strings
        let segments: Vec<String> = processed
            .unicode_words()
            .map(std::string::ToString::to_string)
            .collect();

        if segments.is_empty() && !text.trim().is_empty() {
            // Fallback for languages without word boundaries
            return Err(TokenizerError::unicode(
                "Text contains no word boundaries and cannot be segmented",
            ));
        }

        Ok(segments)
    }
}

/// Whitespace pre-tokenizer that splits on whitespace only.
#[derive(Debug, Clone)]
pub struct WhitespacePreTokenizer {
    /// Whether to normalize Unicode.
    pub normalize: bool,
}

impl WhitespacePreTokenizer {
    /// Create a new whitespace pre-tokenizer.
    #[must_use]
    pub fn new() -> Self {
        Self { normalize: true }
    }
}

impl Default for WhitespacePreTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PreTokenizer for WhitespacePreTokenizer {
    fn pre_tokenize(&self, text: &str) -> Result<Vec<String>> {
        let processed = if self.normalize {
            text.nfc().collect::<String>()
        } else {
            text.to_string()
        };

        // Split on whitespace
        let segments: Vec<String> = processed
            .split_whitespace()
            .map(std::string::ToString::to_string)
            .filter(|s| !s.is_empty())
            .collect();

        Ok(segments)
    }
}

/// Byte-level pre-tokenizer for character-level processing.
#[derive(Debug, Clone)]
pub struct ByteLevelPreTokenizer {
    /// Whether to add prefix space to continuation bytes.
    pub add_prefix_space: bool,
}

impl ByteLevelPreTokenizer {
    /// Create a new byte-level pre-tokenizer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            add_prefix_space: false,
        }
    }

    /// Create with prefix space handling.
    #[must_use]
    pub fn with_prefix_space() -> Self {
        Self {
            add_prefix_space: true,
        }
    }
}

impl Default for ByteLevelPreTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl PreTokenizer for ByteLevelPreTokenizer {
    fn pre_tokenize(&self, text: &str) -> Result<Vec<String>> {
        // For byte-level processing, we want to preserve spaces as separate tokens
        // when add_prefix_space is true (like GPT-2), otherwise split on whitespace
        if self.add_prefix_space {
            // Split on whitespace and keep spaces as separate tokens
            let mut segments = Vec::new();
            let mut current_segment = String::new();

            for ch in text.chars() {
                if ch.is_whitespace() {
                    if !current_segment.is_empty() {
                        segments.push(current_segment);
                        current_segment = String::new();
                    }
                    segments.push(ch.to_string());
                } else {
                    current_segment.push(ch);
                }
            }

            if !current_segment.is_empty() {
                segments.push(current_segment);
            }

            // Filter out empty strings
            Ok(segments.into_iter().filter(|s| !s.is_empty()).collect())
        } else {
            // Split on whitespace like normal pre-tokenizer
            Ok(text
                .split_whitespace()
                .map(std::string::ToString::to_string)
                .filter(|s| !s.is_empty())
                .collect())
        }
    }
}

/// Strip accents from text (basic implementation).
fn strip_accents(text: &str) -> String {
    // This is a basic implementation - a full implementation would need
    // proper Unicode accent stripping with libraries like `unidecode`
    text.chars()
        .filter(|&ch| ch.is_alphanumeric() || ch.is_whitespace())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_pre_tokenizer() {
        let tokenizer = BasicPreTokenizer::new();
        let result = tokenizer.pre_tokenize("Hello, world!").unwrap();
        assert_eq!(result, vec!["Hello", "world"]);

        let result = tokenizer.pre_tokenize("").unwrap();
        assert_eq!(result, Vec::<String>::new());
    }

    #[test]
    fn test_basic_pre_tokenizer_lowercase() {
        let tokenizer = BasicPreTokenizer::with_lowercase();
        let result = tokenizer.pre_tokenize("Hello, WORLD!").unwrap();
        assert_eq!(result, vec!["hello", "world"]);
    }

    #[test]
    fn test_whitespace_pre_tokenizer() {
        let tokenizer = WhitespacePreTokenizer::new();
        let result = tokenizer.pre_tokenize("Hello   world\ttab").unwrap();
        assert_eq!(result, vec!["Hello", "world", "tab"]);
    }

    #[test]
    fn test_byte_level_pre_tokenizer() {
        let tokenizer = ByteLevelPreTokenizer::new();
        let result = tokenizer.pre_tokenize("Hello world").unwrap();
        assert_eq!(result, vec!["Hello", "world"]);

        let tokenizer = ByteLevelPreTokenizer::with_prefix_space();
        let result = tokenizer.pre_tokenize("Hello world").unwrap();
        assert_eq!(result, vec!["Hello", " ", "world"]);
    }

    #[test]
    fn test_unicode_normalization() {
        let tokenizer = BasicPreTokenizer::with_normalization();
        // Test with combining characters
        let text = "café"; // é is U+00E9
        let normalized = text.nfc().collect::<String>();
        let result = tokenizer.pre_tokenize(&normalized).unwrap();
        assert_eq!(result, vec!["café"]);
    }
}
