//! Core tokenizer traits and implementations

use crate::error::{Result, TokenizerError};
use crate::vocabulary::Vocabulary;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Options for tokenization operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenizeOptions {
    /// Maximum sequence length (tokens)
    pub max_length: Option<usize>,
    /// Whether to add special tokens (e.g., BOS, EOS)
    pub add_special_tokens: bool,
    /// Whether to pad sequences to `max_length`
    pub pad_to_max_length: bool,
    /// Truncation strategy
    pub truncation_strategy: TruncationStrategy,
    /// Return token offsets
    pub return_offsets: bool,
}

/// Truncation strategy for sequences that exceed `max_length`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TruncationStrategy {
    /// Do not truncate (return error if too long)
    #[default]
    DoNotTruncate,
    /// Truncate from the beginning
    TruncateFirst,
    /// Truncate from the end
    TruncateLast,
    /// Truncate from both ends equally
    TruncateMiddle,
}

/// Options for decoding operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecodeOptions {
    /// Whether to skip special tokens during decoding
    pub skip_special_tokens: bool,
    /// Whether to clean up spaces around special tokens
    pub clean_up_tokenization_spaces: bool,
}

/// Tokenization result with optional offset information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizationResult {
    /// Token IDs
    pub token_ids: Vec<usize>,
    /// Attention mask (1 for real tokens, 0 for padding)
    pub attention_mask: Vec<usize>,
    /// Token type IDs (for models like BERT)
    pub token_type_ids: Option<Vec<usize>>,
    /// Character offsets for each token (if requested)
    pub offsets: Option<Vec<(usize, usize)>>,
    /// Special tokens mask (1 for special tokens, 0 for regular tokens)
    pub special_tokens_mask: Option<Vec<usize>>,
}

/// Core tokenizer trait defining the interface for all tokenizer implementations
pub trait Tokenizer {
    /// Get the tokenizer's vocabulary
    fn vocabulary(&self) -> &Vocabulary;

    /// Get a mutable reference to the vocabulary
    fn vocabulary_mut(&mut self) -> &mut Vocabulary;

    /// Get the tokenizer's name/model type
    fn name(&self) -> &str;

    /// Encode text to token IDs
    ///
    /// # Errors
    /// Returns `TokenizerError` if encoding fails
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<usize>> {
        self.encode_with_options(
            text,
            &TokenizeOptions {
                add_special_tokens,
                ..Default::default()
            },
        )
    }

    /// Encode text with detailed options
    ///
    /// # Errors
    /// Returns `TokenizerError` if encoding fails
    fn encode_with_options(&self, text: &str, options: &TokenizeOptions) -> Result<Vec<usize>>;

    /// Encode a batch of texts
    ///
    /// # Errors
    /// Returns `TokenizerError` if any encoding fails
    fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<usize>>> {
        texts
            .iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }

    /// Encode a batch of texts with options
    ///
    /// # Errors
    /// Returns `TokenizerError` if any encoding fails
    fn encode_batch_with_options(
        &self,
        texts: &[&str],
        options: &TokenizeOptions,
    ) -> Result<Vec<Vec<usize>>> {
        texts
            .iter()
            .map(|text| self.encode_with_options(text, options))
            .collect()
    }

    /// Encode text to a detailed result
    ///
    /// # Errors
    /// Returns `TokenizerError` if encoding fails
    fn encode_to_result(&self, text: &str, options: &TokenizeOptions)
        -> Result<TokenizationResult>;

    /// Decode token IDs to text
    ///
    /// # Errors
    /// Returns `TokenizerError::UnknownToken` if any token ID is not found in the vocabulary.
    /// Returns `TokenizerError::EncodingError` if the decoding process fails.
    fn decode(&self, token_ids: &[usize], skip_special_tokens: bool) -> Result<String> {
        self.decode_with_options(
            token_ids,
            &DecodeOptions {
                skip_special_tokens,
                ..Default::default()
            },
        )
    }

    /// Decode token IDs with options
    ///
    /// # Errors
    /// Returns `TokenizerError::UnknownToken` if any token ID is not found in the vocabulary.
    /// Returns `TokenizerError::EncodingError` if the decoding process fails.
    fn decode_with_options(&self, token_ids: &[usize], options: &DecodeOptions) -> Result<String>;

    /// Decode a batch of token sequences
    ///
    /// # Errors
    /// Returns `TokenizerError::UnknownToken` if any token ID in the batch is not found in the vocabulary.
    /// Returns `TokenizerError::BatchError` if batch processing fails.
    fn decode_batch(
        &self,
        batch_token_ids: &[&[usize]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        batch_token_ids
            .iter()
            .map(|token_ids| self.decode(token_ids, skip_special_tokens))
            .collect()
    }

    /// Decode a batch with options
    ///
    /// # Errors
    /// Returns `TokenizerError::UnknownToken` if any token ID in the batch is not found in the vocabulary.
    /// Returns `TokenizerError::BatchError` if batch processing fails.
    fn decode_batch_with_options(
        &self,
        batch_token_ids: &[&[usize]],
        options: &DecodeOptions,
    ) -> Result<Vec<String>> {
        batch_token_ids
            .iter()
            .map(|token_ids| self.decode_with_options(token_ids, options))
            .collect()
    }

    /// Convert token IDs to tokens
    ///
    /// # Errors
    /// Returns `TokenizerError::InvalidTokenId` if any token ID is not found in the vocabulary.
    fn convert_ids_to_tokens(&self, token_ids: &[usize]) -> Result<Vec<String>>;

    /// Convert tokens to token IDs
    ///
    /// # Errors
    /// Returns `TokenizerError::UnknownToken` if any token is not found in the vocabulary.
    fn convert_tokens_to_ids(&self, tokens: &[&str]) -> Result<Vec<usize>>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize {
        self.vocabulary().size()
    }

    /// Get the special tokens map
    fn special_tokens(&self) -> &HashMap<String, usize>;

    /// Check if a token is a special token
    fn is_special_token(&self, token: &str) -> bool {
        self.vocabulary().is_special_token(token)
    }

    /// Get special token ID
    fn get_special_token_id(&self, token: &str) -> Option<usize> {
        self.vocabulary().get_special_token_id(token)
    }

    /// Get the BOS (beginning of sequence) token ID
    fn bos_token_id(&self) -> Option<usize> {
        self.get_special_token_id(crate::special_tokens::BOS)
    }

    /// Get the EOS (end of sequence) token ID
    fn eos_token_id(&self) -> Option<usize> {
        self.get_special_token_id(crate::special_tokens::EOS)
    }

    /// Get the PAD token ID
    fn pad_token_id(&self) -> Option<usize> {
        self.get_special_token_id(crate::special_tokens::PAD)
    }

    /// Get the UNK (unknown) token ID
    fn unk_token_id(&self) -> Option<usize> {
        self.get_special_token_id(crate::special_tokens::UNK)
    }

    /// Get the CLS token ID
    fn cls_token_id(&self) -> Option<usize> {
        self.get_special_token_id(crate::special_tokens::CLS)
    }

    /// Get the SEP token ID
    fn sep_token_id(&self) -> Option<usize> {
        self.get_special_token_id(crate::special_tokens::SEP)
    }

    /// Get the MASK token ID
    fn mask_token_id(&self) -> Option<usize> {
        self.get_special_token_id(crate::special_tokens::MASK)
    }

    /// Validate tokenizer configuration
    ///
    /// # Errors
    /// Returns `TokenizerError::SpecialTokenError` if special tokens are not found in vocabulary.
    /// Returns `TokenizerError::VocabularyError` if vocabulary integrity is compromised.
    fn validate(&self) -> Result<()> {
        // Validate vocabulary
        self.vocabulary().validate()?;

        // Validate special tokens
        for (token, &id) in self.special_tokens() {
            if self.vocabulary().get_token_id(token).is_none() {
                return Err(TokenizerError::special_token_error(format!(
                    "Special token '{token}' not found in vocabulary"
                )));
            }

            if self.vocabulary().get_token(id).is_none() {
                return Err(TokenizerError::special_token_error(format!(
                    "Special token ID {id} not found in vocabulary"
                )));
            }
        }

        Ok(())
    }
}

/// Base tokenizer implementation providing common functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseTokenizer {
    /// The tokenizer's vocabulary
    vocabulary: Vocabulary,
    /// Model name/type
    model_name: String,
}

impl BaseTokenizer {
    /// Create a new base tokenizer
    #[must_use]
    pub fn new(model_name: String) -> Self {
        Self {
            vocabulary: Vocabulary::new(),
            model_name,
        }
    }

    /// Create a new base tokenizer with vocabulary
    #[must_use]
    pub const fn with_vocabulary(model_name: String, vocabulary: Vocabulary) -> Self {
        Self {
            vocabulary,
            model_name,
        }
    }

    /// Get the model name
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Set the model name
    pub fn set_model_name(&mut self, name: String) {
        self.model_name = name;
    }
}

impl Tokenizer for BaseTokenizer {
    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn vocabulary_mut(&mut self) -> &mut Vocabulary {
        &mut self.vocabulary
    }

    fn name(&self) -> &str {
        &self.model_name
    }

    fn encode_with_options(&self, _text: &str, _options: &TokenizeOptions) -> Result<Vec<usize>> {
        // This is a placeholder implementation - concrete tokenizers should override
        Err(TokenizerError::unsupported_operation(
            "encode_with_options",
            self.name(),
        ))
    }

    fn encode_to_result(
        &self,
        text: &str,
        options: &TokenizeOptions,
    ) -> Result<TokenizationResult> {
        let token_ids = self.encode_with_options(text, options)?;

        let attention_mask = vec![1; token_ids.len()];
        let special_tokens_mask = Some(
            token_ids
                .iter()
                .map(|&id| {
                    self.vocabulary()
                        .get_token(id)
                        .map_or(0, |token| usize::from(self.is_special_token(token)))
                })
                .collect(),
        );

        Ok(TokenizationResult {
            token_ids,
            attention_mask,
            token_type_ids: None,
            offsets: None,
            special_tokens_mask,
        })
    }

    fn decode_with_options(
        &self,
        _token_ids: &[usize],
        _options: &DecodeOptions,
    ) -> Result<String> {
        // This is a placeholder implementation - concrete tokenizers should override
        Err(TokenizerError::unsupported_operation(
            "decode_with_options",
            self.name(),
        ))
    }

    fn convert_ids_to_tokens(&self, token_ids: &[usize]) -> Result<Vec<String>> {
        token_ids
            .iter()
            .map(|&id| {
                self.vocabulary()
                    .get_token(id)
                    .ok_or_else(|| TokenizerError::invalid_token_id(id))
                    .map(std::string::ToString::to_string)
            })
            .collect()
    }

    fn convert_tokens_to_ids(&self, tokens: &[&str]) -> Result<Vec<usize>> {
        tokens
            .iter()
            .map(|&token| {
                self.vocabulary()
                    .get_token_id(token)
                    .ok_or_else(|| TokenizerError::unknown_token(token.to_string()))
            })
            .collect()
    }

    fn special_tokens(&self) -> &HashMap<String, usize> {
        self.vocabulary().special_tokens()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocabulary::Vocabulary;

    #[test]
    fn test_base_tokenizer_creation() {
        let vocab = Vocabulary::new();
        let tokenizer = BaseTokenizer::with_vocabulary("test".to_string(), vocab);

        assert_eq!(tokenizer.name(), "test");
        assert_eq!(tokenizer.vocab_size(), 0);
    }

    #[test]
    fn test_tokenize_options_default() {
        let options = TokenizeOptions::default();
        assert_eq!(options.max_length, None);
        assert!(!options.add_special_tokens);
        assert!(!options.pad_to_max_length);
        assert_eq!(
            options.truncation_strategy,
            TruncationStrategy::DoNotTruncate
        );
        assert!(!options.return_offsets);
    }

    #[test]
    fn test_decode_options_default() {
        let options = DecodeOptions::default();
        assert!(!options.skip_special_tokens);
        assert!(!options.clean_up_tokenization_spaces);
    }

    #[test]
    fn test_tokenization_result_creation() {
        let result = TokenizationResult {
            token_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            token_type_ids: None,
            offsets: None,
            special_tokens_mask: Some(vec![0, 1, 0]),
        };

        assert_eq!(result.token_ids, vec![1, 2, 3]);
        assert_eq!(result.attention_mask, vec![1, 1, 1]);
        assert!(result.token_type_ids.is_none());
        assert!(result.offsets.is_none());
        assert_eq!(result.special_tokens_mask, Some(vec![0, 1, 0]));
    }

    #[test]
    fn test_truncation_strategy() {
        assert_eq!(TruncationStrategy::DoNotTruncate as u8, 0);
        assert_eq!(TruncationStrategy::TruncateFirst as u8, 1);
        assert_eq!(TruncationStrategy::TruncateLast as u8, 2);
        assert_eq!(TruncationStrategy::TruncateMiddle as u8, 3);
    }
}
