//! BERT tokenizer implementation with `WordPiece` algorithm and pre-trained vocabulary support

use crate::downloader::VocabDownloader;
use crate::error::{Result, TokenizerError};
use crate::tokenizer::{DecodeOptions, TokenizationResult, TokenizeOptions, Tokenizer};
use crate::vocabulary::Vocabulary;
use regex::Regex;

/// BERT tokenizer with `WordPiece` algorithm and pre-trained vocabulary
pub struct BERTTokenizer {
    /// Vocabulary containing token-to-ID mappings
    vocabulary: Vocabulary,
    /// Regex pattern for basic tokenization
    basic_pattern: Regex,
    /// Model name
    model_name: String,
}

impl BERTTokenizer {
    /// Create a new BERT tokenizer with downloaded vocabulary
    ///
    /// # Errors
    /// Returns `TokenizerError::IoError` if vocabulary download fails
    /// Returns `TokenizerError::VocabularyError` if vocabulary parsing fails
    ///
    /// # Panics
    /// Panics if regex pattern compilation fails
    pub fn new(model_name: &str) -> Result<Self> {
        let downloader = VocabDownloader::new();

        // Download BERT vocabulary
        let vocab_data = downloader.download_bert_vocab(model_name)?;

        // Create vocabulary from downloaded data
        let vocabulary = downloader.create_vocab_from_bert_data(&vocab_data)?;

        // Basic pattern for splitting text (similar to BERT's basic tokenizer)
        let basic_pattern = Regex::new(r"(\w+|[^\w\s]+|\s+)").unwrap();

        Ok(Self {
            vocabulary,
            basic_pattern,
            model_name: model_name.to_string(),
        })
    }

    /// Create BERT tokenizer with custom vocabulary
    ///
    /// # Errors
    /// Returns `TokenizerError` if vocabulary setup fails
    ///
    /// # Panics
    /// Panics if regex pattern compilation fails
    pub fn with_vocabulary(model_name: String, vocabulary: Vocabulary) -> Result<Self> {
        let basic_pattern = Regex::new(r"(\w+|[^\w\s]+|\s+)").unwrap();

        Ok(Self {
            vocabulary,
            basic_pattern,
            model_name,
        })
    }

    /// Get vocabulary size
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocabulary.size()
    }

    /// Get model name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.model_name
    }

    /// Get CLS token ID
    #[must_use]
    pub fn cls_token_id(&self) -> Option<usize> {
        self.vocabulary.get_special_token_id("[CLS]")
    }

    /// Get SEP token ID
    #[must_use]
    pub fn sep_token_id(&self) -> Option<usize> {
        self.vocabulary.get_special_token_id("[SEP]")
    }

    /// Get MASK token ID
    #[must_use]
    pub fn mask_token_id(&self) -> Option<usize> {
        self.vocabulary.get_special_token_id("[MASK]")
    }

    /// Get PAD token ID
    #[must_use]
    pub fn pad_token_id(&self) -> Option<usize> {
        self.vocabulary.get_special_token_id("[PAD]")
    }

    /// Get UNK token ID
    #[must_use]
    pub fn unk_token_id(&self) -> Option<usize> {
        self.vocabulary.get_special_token_id("[UNK]")
    }

    /// Get BOS token ID (BERT doesn't have BOS)
    #[must_use]
    pub const fn bos_token_id(&self) -> Option<usize> {
        None
    }

    /// Get EOS token ID (BERT doesn't have EOS)
    #[must_use]
    pub const fn eos_token_id(&self) -> Option<usize> {
        None
    }

    /// `WordPiece` tokenization of a single word
    fn wordpiece_tokenize(&self, word: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut remaining = word.to_string();

        // Try to find the longest prefix that exists in vocabulary
        while !remaining.is_empty() {
            let mut found = false;

            // Try different prefixes
            for i in (1..=remaining.len()).rev() {
                let prefix = &remaining[..i];

                // Check if prefix exists in vocabulary
                if self.vocabulary.get_token_id(prefix).is_some() {
                    tokens.push(prefix.to_string());
                    remaining = remaining[i..].to_string();
                    found = true;
                    break;
                }

                // Check if prefix with ## exists in vocabulary
                let prefixed = format!("##{prefix}");
                if self.vocabulary.get_token_id(&prefixed).is_some() {
                    tokens.push(prefixed);
                    remaining = remaining[i..].to_string();
                    found = true;
                    break;
                }
            }

            // If no valid prefix found, use [UNK] token
            if !found {
                if let Some(unk_token) = self.vocabulary.get_token(self.unk_token_id().unwrap_or(0))
                {
                    tokens.push(unk_token.to_string());
                } else {
                    tokens.push("[UNK]".to_string());
                }
                break;
            }
        }

        tokens
    }

    /// Basic tokenization (split on whitespace and punctuation)
    fn basic_tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();

        for cap in self.basic_pattern.captures_iter(text) {
            let token = cap.get(1).unwrap().as_str();
            if !token.chars().all(char::is_whitespace) {
                tokens.push(token.to_string());
            }
        }

        tokens
    }
}

impl Tokenizer for BERTTokenizer {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<usize>> {
        let basic_tokens = self.basic_tokenize(text);
        let mut tokens = Vec::new();

        for basic_token in basic_tokens {
            let wordpiece_tokens = self.wordpiece_tokenize(&basic_token);
            for wp_token in wordpiece_tokens {
                if let Some(id) = self.vocabulary.get_token_id(&wp_token) {
                    tokens.push(id);
                } else {
                    // Fallback to UNK
                    if let Some(unk_id) = self.unk_token_id() {
                        tokens.push(unk_id);
                    } else {
                        return Err(TokenizerError::unknown_token(wp_token));
                    }
                }
            }
        }

        if add_special_tokens {
            let mut result = Vec::new();
            // Add CLS
            if let Some(cls_id) = self.cls_token_id() {
                result.push(cls_id);
            }
            result.extend(tokens);
            // Add SEP
            if let Some(sep_id) = self.sep_token_id() {
                result.push(sep_id);
            }
            Ok(result)
        } else {
            Ok(tokens)
        }
    }

    fn encode_with_options(&self, text: &str, options: &TokenizeOptions) -> Result<Vec<usize>> {
        self.encode(text, options.add_special_tokens)
    }

    fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<usize>>> {
        texts
            .iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }

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

    fn decode(&self, token_ids: &[usize], skip_special_tokens: bool) -> Result<String> {
        let mut text = String::new();

        for &token_id in token_ids {
            if let Some(token) = self.vocabulary.get_token(token_id) {
                if skip_special_tokens && self.vocabulary.is_special_token(token) {
                    continue;
                }

                // Remove ## prefix for WordPiece continuation tokens
                let display_token = token.strip_prefix("##").unwrap_or(token);
                text.push_str(display_token);
            } else {
                return Err(TokenizerError::invalid_token_id(token_id));
            }
        }

        Ok(text)
    }

    fn decode_with_options(&self, token_ids: &[usize], options: &DecodeOptions) -> Result<String> {
        self.decode(token_ids, options.skip_special_tokens)
    }

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

    fn convert_ids_to_tokens(&self, token_ids: &[usize]) -> Result<Vec<String>> {
        token_ids
            .iter()
            .map(|&id| {
                self.vocabulary
                    .get_token(id)
                    .ok_or_else(|| TokenizerError::invalid_token_id(id))
                    .map(str::to_string)
            })
            .collect()
    }

    fn convert_tokens_to_ids(&self, tokens: &[&str]) -> Result<Vec<usize>> {
        tokens
            .iter()
            .map(|token| {
                self.vocabulary
                    .get_token_id(token)
                    .ok_or_else(|| TokenizerError::unknown_token((*token).to_string()))
            })
            .collect()
    }

    fn special_tokens(&self) -> &std::collections::HashMap<String, usize> {
        self.vocabulary.special_tokens()
    }

    fn is_special_token(&self, token: &str) -> bool {
        self.vocabulary.is_special_token(token)
    }

    fn bos_token_id(&self) -> Option<usize> {
        self.bos_token_id()
    }

    fn eos_token_id(&self) -> Option<usize> {
        self.eos_token_id()
    }

    fn pad_token_id(&self) -> Option<usize> {
        self.pad_token_id()
    }

    fn unk_token_id(&self) -> Option<usize> {
        self.unk_token_id()
    }

    fn cls_token_id(&self) -> Option<usize> {
        self.cls_token_id()
    }

    fn sep_token_id(&self) -> Option<usize> {
        self.sep_token_id()
    }

    fn mask_token_id(&self) -> Option<usize> {
        self.mask_token_id()
    }

    fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    fn vocabulary_mut(&mut self) -> &mut Vocabulary {
        &mut self.vocabulary
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

    fn validate(&self) -> Result<()> {
        self.vocabulary.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_tokenizer_creation() {
        // Skip test if network is unavailable
        let result = BERTTokenizer::new("bert-base");
        match result {
            Ok(tokenizer) => {
                assert_eq!(tokenizer.name(), "bert-base");
                assert!(tokenizer.vocab_size() > 0);
                assert!(tokenizer.cls_token_id().is_some());
                assert!(tokenizer.sep_token_id().is_some());
                assert!(tokenizer.mask_token_id().is_some());
            }
            Err(TokenizerError::IoError { .. }) => {
                // Expected when no internet connection
                println!("Skipping BERT test due to network unavailability");
            }
            Err(e) => panic!("Unexpected error: {e}"),
        }
    }

    #[test]
    fn test_bert_special_tokens() {
        // Test with mock vocabulary since we can't download in tests
        let mut vocab = Vocabulary::new();
        vocab.add_special_token("[CLS]".to_string());
        vocab.add_special_token("[SEP]".to_string());
        vocab.add_special_token("[MASK]".to_string());
        vocab.add_special_token("[PAD]".to_string());
        vocab.add_special_token("[UNK]".to_string());

        let tokenizer = BERTTokenizer::with_vocabulary("bert-base".to_string(), vocab).unwrap();
        assert!(tokenizer.cls_token_id().is_some());
        assert!(tokenizer.sep_token_id().is_some());
        assert!(tokenizer.mask_token_id().is_some());
        assert!(tokenizer.pad_token_id().is_some());
        assert!(tokenizer.unk_token_id().is_some());
    }

    #[test]
    fn test_bert_basic_tokenize() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("hello".to_string());
        vocab.add_token("world".to_string());
        vocab.add_token("[UNK]".to_string());

        let tokenizer = BERTTokenizer::with_vocabulary("bert-base".to_string(), vocab).unwrap();
        let tokens = tokenizer.basic_tokenize("hello world!");

        // Should split on whitespace and punctuation
        assert!(tokens.len() >= 2); // "hello", "world", "!"
    }
}
