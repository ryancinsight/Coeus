//! High-level encoding interface for tokenizers

use crate::error::{Result, TokenizerError};
use crate::tokenizer::{DecodeOptions, TokenizeOptions, Tokenizer};
use crate::vocabulary::Vocabulary;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// High-level tokenizer interface providing tiktoken-compatible API
pub struct Encoding {
    /// The underlying tokenizer implementation
    tokenizer: Box<dyn Tokenizer + Send + Sync>,
}

impl Encoding {
    /// Create a new encoding for a specific model
    ///
    /// # Errors
    /// Returns `TokenizerError::ModelError` if the model is not supported or not yet implemented.
    /// Returns `TokenizerError::ConfigError` if the model configuration is invalid.
    pub fn new(model_name: &str) -> Result<Self> {
        // For now, return an error since no model implementations exist yet
        Err(TokenizerError::model_error(format!(
            "Model '{model_name}' not yet implemented. Enable features to add model support. Available models: {}",
            Self::available_models().join(", ")
        )))
    }

    /// Create encoding from an existing tokenizer
    pub fn from_tokenizer<T: Tokenizer + Send + Sync + 'static>(tokenizer: T) -> Self {
        Self {
            tokenizer: Box::new(tokenizer),
        }
    }

    /// Create encoding for encoding only (no decoding capability)
    ///
    /// # Errors
    /// Returns `TokenizerError::ModelError` since no model implementations exist yet
    pub fn for_encoding_only(model_name: &str) -> Result<Self> {
        // For now, return an error since no model implementations exist yet
        Err(TokenizerError::model_error(format!(
            "Model '{model_name}' not yet implemented. Enable features to add model support."
        )))
    }

    /// Get list of available models
    #[must_use]
    pub fn available_models() -> Vec<String> {
        let mut models = Vec::new();

        #[cfg(feature = "gpt2")]
        models.push("gpt2".to_string());

        #[cfg(feature = "gpt3")]
        {
            models.push("gpt-3.5-turbo".to_string());
            models.push("gpt-4".to_string());
        }

        #[cfg(feature = "clip")]
        models.push("clip".to_string());

        #[cfg(feature = "bert")]
        {
            models.push("bert-base".to_string());
            models.push("bert-large".to_string());
        }

        if models.is_empty() {
            models.push("none (enable features to add models)".to_string());
        }

        models
    }

    /// Get the underlying tokenizer
    #[must_use]
    pub fn tokenizer(&self) -> &dyn Tokenizer {
        self.tokenizer.as_ref()
    }

    /// Get the model name
    #[must_use]
    pub fn model_name(&self) -> &str {
        self.tokenizer.name()
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    /// Encode text to token IDs (tiktoken-compatible API)
    ///
    /// # Errors
    /// Returns `TokenizerError` if encoding fails
    pub fn encode(&self, text: &str) -> Result<Vec<usize>> {
        self.tokenizer.encode(text, false)
    }

    /// Encode text with special tokens (tiktoken-compatible API)
    ///
    /// # Errors
    /// Returns `TokenizerError` if encoding fails
    pub fn encode_with_special_tokens(&self, text: &str) -> Result<Vec<usize>> {
        self.tokenizer.encode(text, true)
    }

    /// Encode text with options
    ///
    /// # Errors
    /// Returns `TokenizerError` if encoding fails
    pub fn encode_with_options(&self, text: &str, options: &TokenizeOptions) -> Result<Vec<usize>> {
        self.tokenizer.encode_with_options(text, options)
    }

    /// Encode a batch of texts
    ///
    /// # Errors
    /// Returns `TokenizerError` if encoding fails
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<usize>>> {
        self.tokenizer.encode_batch(texts, false)
    }

    /// Encode a batch with special tokens
    ///
    /// # Errors
    /// Returns `TokenizerError` if encoding fails
    pub fn encode_batch_with_special_tokens(&self, texts: &[&str]) -> Result<Vec<Vec<usize>>> {
        self.tokenizer.encode_batch(texts, true)
    }

    /// Encode a batch with options
    ///
    /// # Errors
    /// Returns `TokenizerError` if encoding fails
    pub fn encode_batch_with_options(
        &self,
        texts: &[&str],
        options: &TokenizeOptions,
    ) -> Result<Vec<Vec<usize>>> {
        self.tokenizer.encode_batch_with_options(texts, options)
    }

    /// Decode token IDs to text (tiktoken-compatible API)
    ///
    /// # Errors
    /// Returns `TokenizerError` if decoding fails
    pub fn decode(&self, token_ids: &[usize]) -> Result<String> {
        self.tokenizer.decode(token_ids, false)
    }

    /// Decode with special token handling
    ///
    /// # Errors
    /// Returns `TokenizerError` if decoding fails
    pub fn decode_with_special_tokens(&self, token_ids: &[usize]) -> Result<String> {
        self.tokenizer.decode(token_ids, true)
    }

    /// Decode with options
    ///
    /// # Errors
    /// Returns `TokenizerError` if decoding fails
    pub fn decode_with_options(
        &self,
        token_ids: &[usize],
        options: &DecodeOptions,
    ) -> Result<String> {
        self.tokenizer.decode_with_options(token_ids, options)
    }

    /// Decode a batch of token sequences
    ///
    /// # Errors
    /// Returns `TokenizerError` if decoding fails
    pub fn decode_batch(&self, batch_token_ids: &[&[usize]]) -> Result<Vec<String>> {
        self.tokenizer.decode_batch(batch_token_ids, false)
    }

    /// Decode a batch with special token handling
    ///
    /// # Errors
    /// Returns `TokenizerError` if decoding fails
    pub fn decode_batch_with_special_tokens(
        &self,
        batch_token_ids: &[&[usize]],
    ) -> Result<Vec<String>> {
        self.tokenizer.decode_batch(batch_token_ids, true)
    }

    /// Decode a batch with options
    ///
    /// # Errors
    /// Returns `TokenizerError` if decoding fails
    pub fn decode_batch_with_options(
        &self,
        batch_token_ids: &[&[usize]],
        options: &DecodeOptions,
    ) -> Result<Vec<String>> {
        self.tokenizer
            .decode_batch_with_options(batch_token_ids, options)
    }

    /// Convert token IDs to tokens
    ///
    /// # Errors
    /// Returns `TokenizerError` if conversion fails
    pub fn convert_ids_to_tokens(&self, token_ids: &[usize]) -> Result<Vec<String>> {
        self.tokenizer.convert_ids_to_tokens(token_ids)
    }

    /// Convert tokens to token IDs
    ///
    /// # Errors
    /// Returns `TokenizerError` if conversion fails
    pub fn convert_tokens_to_ids(&self, tokens: &[&str]) -> Result<Vec<usize>> {
        self.tokenizer.convert_tokens_to_ids(tokens)
    }

    /// Get special token IDs
    #[must_use]
    pub fn special_tokens(&self) -> &HashMap<String, usize> {
        self.tokenizer.special_tokens()
    }

    /// Get BOS token ID
    #[must_use]
    pub fn bos_token_id(&self) -> Option<usize> {
        self.tokenizer.bos_token_id()
    }

    /// Get EOS token ID
    #[must_use]
    pub fn eos_token_id(&self) -> Option<usize> {
        self.tokenizer.eos_token_id()
    }

    /// Get PAD token ID
    #[must_use]
    pub fn pad_token_id(&self) -> Option<usize> {
        self.tokenizer.pad_token_id()
    }

    /// Get UNK token ID
    #[must_use]
    pub fn unk_token_id(&self) -> Option<usize> {
        self.tokenizer.unk_token_id()
    }

    /// Get CLS token ID
    #[must_use]
    pub fn cls_token_id(&self) -> Option<usize> {
        self.tokenizer.cls_token_id()
    }

    /// Get SEP token ID
    #[must_use]
    pub fn sep_token_id(&self) -> Option<usize> {
        self.tokenizer.sep_token_id()
    }

    /// Get MASK token ID
    #[must_use]
    pub fn mask_token_id(&self) -> Option<usize> {
        self.tokenizer.mask_token_id()
    }

    /// Validate the encoding
    ///
    /// # Errors
    /// Returns `TokenizerError` if validation fails
    pub fn validate(&self) -> Result<()> {
        self.tokenizer.validate()
    }

    /// Get encoding information
    #[must_use]
    pub fn info(&self) -> EncodingInfo {
        EncodingInfo {
            model_name: self.model_name().to_string(),
            vocab_size: self.vocab_size(),
            special_tokens: self.special_tokens().clone(),
            max_sequence_length: crate::MAX_SEQUENCE_LENGTH,
        }
    }
}

/// Information about an encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingInfo {
    /// Model name
    pub model_name: String,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Special tokens mapping
    pub special_tokens: HashMap<String, usize>,
    /// Maximum sequence length
    pub max_sequence_length: usize,
}

/// Configuration for creating encodings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingConfig {
    /// Model name
    pub model_name: String,
    /// Custom vocabulary (optional)
    pub vocabulary: Option<Vocabulary>,
    /// Additional special tokens
    pub additional_special_tokens: Vec<String>,
    /// Maximum sequence length
    pub max_sequence_length: Option<usize>,
}

impl EncodingConfig {
    /// Create a new encoding configuration
    #[must_use]
    pub const fn new(model_name: String) -> Self {
        Self {
            model_name,
            vocabulary: None,
            additional_special_tokens: Vec::new(),
            max_sequence_length: None,
        }
    }

    /// Set custom vocabulary
    #[must_use]
    pub fn with_vocabulary(mut self, vocabulary: Vocabulary) -> Self {
        self.vocabulary = Some(vocabulary);
        self
    }

    /// Add special tokens
    #[must_use]
    pub fn with_special_tokens(mut self, tokens: Vec<String>) -> Self {
        self.additional_special_tokens = tokens;
        self
    }

    /// Set maximum sequence length
    #[must_use]
    pub const fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_sequence_length = Some(max_length);
        self
    }
}

impl Encoding {
    /// Create encoding from configuration
    ///
    /// # Errors
    /// Returns `TokenizerError` if configuration is invalid or model creation fails
    pub fn from_config(config: EncodingConfig) -> Result<Self> {
        let mut encoding = Self::new(&config.model_name)?;

        // Add custom vocabulary if provided
        if let Some(vocab) = config.vocabulary {
            *encoding.tokenizer.vocabulary_mut() = vocab;
        }

        // Add additional special tokens
        for token in config.additional_special_tokens {
            encoding.tokenizer.vocabulary_mut().add_special_token(token);
        }

        // Validate the configuration
        encoding.validate()?;

        Ok(encoding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_available_models() {
        let models = Encoding::available_models();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_encoding_config_creation() {
        let config = EncodingConfig::new("test".to_string());
        assert_eq!(config.model_name, "test");
        assert!(config.vocabulary.is_none());
        assert!(config.additional_special_tokens.is_empty());
        assert!(config.max_sequence_length.is_none());
    }

    #[test]
    fn test_encoding_config_with_options() {
        let vocab = Vocabulary::new();
        let config = EncodingConfig::new("test".to_string())
            .with_vocabulary(vocab)
            .with_special_tokens(vec!["[TEST]".to_string()])
            .with_max_length(512);

        assert!(config.vocabulary.is_some());
        assert_eq!(config.additional_special_tokens, vec!["[TEST]"]);
        assert_eq!(config.max_sequence_length, Some(512));
    }

    #[test]
    fn test_unknown_model_error() {
        let result = Encoding::new("unknown-model");
        assert!(result.is_err());
        // Check that it's an error without unwrapping (avoids Debug requirement)
        match result {
            Err(TokenizerError::ModelError { .. }) => {} // Expected error type
            _ => panic!("Expected ModelError"),
        }
    }
}
