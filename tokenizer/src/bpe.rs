//! Byte-Pair Encoding (BPE) implementation

use crate::error::{Result, TokenizerError};
use crate::vocabulary::Vocabulary;
use ahash::AHashMap;
use regex::Regex;
use std::collections::HashSet;

/// BPE tokenizer implementation
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// The vocabulary containing token-to-ID mappings
    vocabulary: Vocabulary,
    /// Regex pattern for tokenization
    pattern: Regex,
    /// Merge rules for BPE (in application order)
    merges: Vec<(String, String)>,
    /// Special tokens
    special_tokens: HashSet<String>,
    /// Model name
    model_name: String,
}

impl BpeTokenizer {
    /// Get the model name
    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get the vocabulary
    #[must_use]
    pub const fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }

    /// Get mutable access to the vocabulary
    #[allow(clippy::missing_const_for_fn)]
    pub fn vocabulary_mut(&mut self) -> &mut Vocabulary {
        &mut self.vocabulary
    }

    /// Create a new BPE tokenizer
    #[must_use]
    pub fn new(model_name: String) -> Self {
        Self {
            vocabulary: Vocabulary::new(),
            pattern: Self::default_pattern(),
            merges: Vec::new(),
            special_tokens: HashSet::new(),
            model_name,
        }
    }

    /// Create BPE tokenizer with custom pattern
    ///
    /// # Errors
    /// Returns `TokenizerError::ConfigError` if the provided regex pattern is invalid.
    pub fn with_pattern(model_name: String, pattern: &str) -> Result<Self> {
        let pattern = Regex::new(pattern)
            .map_err(|e| TokenizerError::config_error(format!("Invalid regex pattern: {e}")))?;

        Ok(Self {
            vocabulary: Vocabulary::new(),
            pattern,
            merges: Vec::new(),
            special_tokens: HashSet::new(),
            model_name,
        })
    }

    /// Get the default regex pattern for BPE tokenization
    ///
    /// # Panics
    /// Panics if the default BPE regex pattern is invalid (should never happen in practice).
    #[must_use]
    pub fn default_pattern() -> Regex {
        // GPT-like pattern: split on whitespace and punctuation (simplified without look-ahead)
        Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
            .expect("Default BPE pattern should be valid")
    }

    /// Add special tokens
    pub fn add_special_tokens(&mut self, tokens: &[String]) {
        for token in tokens {
            self.special_tokens.insert(token.clone());
            self.vocabulary.add_special_token(token.clone());
        }
    }

    /// Load pre-trained vocabulary and merges
    ///
    /// # Errors
    /// Returns `TokenizerError::VocabularyError` if vocabulary integrity is compromised during loading.
    pub fn load_vocab_and_merges(
        &mut self,
        vocab: AHashMap<String, usize>,
        merges: Vec<(String, String)>,
    ) -> Result<()> {
        // Clear existing vocabulary
        self.vocabulary.clear();

        // Add all tokens from vocab
        let mut sorted_vocab: Vec<_> = vocab.into_iter().collect();
        sorted_vocab.sort_by_key(|(_, id)| *id);

        for (token, id) in sorted_vocab {
            // Reserve space for the ID if needed
            while self.vocabulary.id_to_token.len() <= id {
                self.vocabulary.id_to_token.push(String::new());
            }
            self.vocabulary.id_to_token[id].clone_from(&token);
            self.vocabulary.token_to_id.insert(token, id);
        }

        // Set next_id to the highest ID + 1
        self.vocabulary.next_id = self.vocabulary.id_to_token.len();

        // Load merges in the correct application order
        self.merges = merges;

        self.validate()?;
        Ok(())
    }

    /// Train BPE on a corpus of text
    ///
    /// # Errors
    /// Returns `TokenizerError::InvalidInput` if the corpus is empty.
    /// Returns `TokenizerError::BpeMergeError` if BPE merge operations fail.
    /// Returns `TokenizerError::VocabularyError` if vocabulary operations fail.
    pub fn train(
        &mut self,
        corpus: &[String],
        vocab_size: usize,
        num_merges: Option<usize>,
    ) -> Result<()> {
        if corpus.is_empty() {
            return Err(TokenizerError::invalid_input(
                "Empty corpus provided for training",
            ));
        }

        // Step 1: Pre-tokenize the corpus
        let mut word_freqs = AHashMap::default();
        for text in corpus {
            let words = self.pre_tokenize(text);
            for word in words {
                *word_freqs.entry(word).or_insert(0) += 1;
            }
        }

        // Step 2: Initialize vocabulary with byte-level tokens
        let mut vocab = Self::initialize_byte_vocab();
        let mut merges = Vec::new();

        // Step 3: Convert words to token sequences
        let mut word_tokens: AHashMap<String, Vec<String>> = AHashMap::default();
        for word in word_freqs.keys() {
            word_tokens.insert(word.clone(), Self::word_to_bytes(word));
        }

        // Step 4: Perform merges
        let target_merges = num_merges.unwrap_or_else(|| vocab_size.saturating_sub(256));
        let mut merge_count = 0;

        while merge_count < target_merges {
            // Find the most frequent pair
            let pair_stats = Self::get_pair_stats(&word_tokens, &word_freqs);
            if pair_stats.is_empty() {
                break; // No more pairs to merge
            }

            // Get the most frequent pair
            let best_pair = pair_stats
                .iter()
                .max_by_key(|(_, freq)| *freq)
                .map(|(pair, _)| pair.clone())
                .ok_or_else(|| TokenizerError::bpe_merge_error("No pairs found for merging"))?;

            // Perform the merge
            let new_token = format!("{}{}", best_pair.0, best_pair.1);
            Self::merge_pair(&mut word_tokens, &best_pair, &new_token);

            // Add to merges and vocabulary
            let _new_id = vocab.size();
            vocab.add_token(new_token.clone());
            merges.push(best_pair.clone());

            merge_count += 1;
        }

        // Step 5: Store the results
        self.merges = merges;
        self.vocabulary = vocab;

        // Add special tokens if any
        for token in &self.special_tokens {
            self.vocabulary.add_special_token(token.clone());
        }

        Ok(())
    }

    /// Pre-tokenize text using regex pattern
    fn pre_tokenize(&self, text: &str) -> Vec<String> {
        let mut words = Vec::new();
        for cap in self.pattern.captures_iter(text) {
            if let Some(word) = cap.get(0) {
                words.push(word.as_str().to_string());
            }
        }
        words
    }

    /// Convert word to byte sequences
    fn word_to_bytes(word: &str) -> Vec<String> {
        word.chars().map(|c| format!("{:02x}", c as u32)).collect()
    }

    /// Initialize vocabulary with byte-level tokens
    fn initialize_byte_vocab() -> Vocabulary {
        let mut vocab = Vocabulary::new();

        // Add bytes 0-255 as initial vocabulary
        for byte in 0..=255u8 {
            let token = format!("{byte:02x}");
            vocab.add_token(token);
        }

        vocab
    }

    /// Get statistics for all pairs in the current tokenization
    fn get_pair_stats(
        word_tokens: &AHashMap<String, Vec<String>>,
        word_freqs: &AHashMap<String, u64>,
    ) -> AHashMap<(String, String), u64> {
        let mut pair_stats = AHashMap::default();

        for (word, tokens) in word_tokens {
            let freq = word_freqs.get(word).copied().unwrap_or(0);

            if tokens.len() >= 2 {
                for pair in tokens.windows(2) {
                    if pair.len() == 2 {
                        let pair_key = (pair[0].clone(), pair[1].clone());
                        *pair_stats.entry(pair_key).or_insert(0) += freq;
                    }
                }
            }
        }

        pair_stats
    }

    /// Merge a pair in all words
    fn merge_pair(
        word_tokens: &mut AHashMap<String, Vec<String>>,
        pair: &(String, String),
        new_token: &str,
    ) {
        for tokens in word_tokens.values_mut() {
            let mut i = 0;
            while i < tokens.len() - 1 {
                if tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                    tokens[i] = new_token.to_string();
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
    }

    /// Encode text using BPE
    ///
    /// # Errors
    /// Returns `TokenizerError::EncodingError` if BPE encoding fails.
    pub fn encode_bpe(&self, text: &str) -> Result<Vec<String>> {
        let words = self.pre_tokenize(text);
        let mut tokens = Vec::new();

        for word in words {
            let word_tokens = self.encode_word_bpe(&word);
            tokens.extend(word_tokens);
        }

        Ok(tokens)
    }

    /// Encode a single word using BPE
    fn encode_word_bpe(&self, word: &str) -> Vec<String> {
        let mut tokens = Self::word_to_bytes(word);

        // Apply merges in the correct order (most frequent first)
        for (token1, token2) in &self.merges {
            let mut i = 0;
            while i < tokens.len() - 1 {
                if tokens[i] == *token1 && tokens[i + 1] == *token2 {
                    let merged = format!("{token1}{token2}");
                    tokens[i] = merged;
                    tokens.remove(i + 1);
                    // Don't increment i here, check the same position again
                    // in case we can merge more
                } else {
                    i += 1;
                }
            }
        }

        tokens
    }

    /// Decode BPE tokens back to text
    ///
    /// # Errors
    /// Returns `TokenizerError::EncodingError` if BPE decoding fails.
    pub fn decode_bpe(&self, tokens: &[String]) -> Result<String> {
        let mut text = String::new();

        for (i, token) in tokens.iter().enumerate() {
            // Convert hex back to character if it's a byte token
            if token.len() == 2 && token.chars().all(|c| c.is_ascii_hexdigit()) {
                if let Ok(byte) = u32::from_str_radix(token, 16) {
                    if let Some(ch) = char::from_u32(byte) {
                        text.push(ch);
                    }
                }
            } else {
                text.push_str(token);
            }

            // Add space between words (simple heuristic)
            if i < tokens.len() - 1 && !token.ends_with(' ') {
                text.push(' ');
            }
        }

        Ok(text.trim().to_string())
    }

    /// Get the merges
    #[must_use]
    pub const fn merges(&self) -> &Vec<(String, String)> {
        &self.merges
    }

    /// Validate BPE tokenizer configuration
    ///
    /// # Errors
    /// Returns `TokenizerError::VocabularyError` if vocabulary integrity is compromised.
    /// Returns `TokenizerError::BpeMergeError` if merge rules reference non-existent tokens.
    pub fn validate(&self) -> Result<()> {
        self.vocabulary.validate()?;

        // Validate that all merge pairs exist in vocabulary
        for (token1, token2) in &self.merges {
            if self.vocabulary.get_token_id(token1).is_none() {
                return Err(TokenizerError::bpe_merge_error(format!(
                    "Merge token '{token1}' not found in vocabulary"
                )));
            }
            if self.vocabulary.get_token_id(token2).is_none() {
                return Err(TokenizerError::bpe_merge_error(format!(
                    "Merge token '{token2}' not found in vocabulary"
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_tokenizer_creation() {
        let tokenizer = BpeTokenizer::new("test".to_string());
        assert_eq!(tokenizer.model_name, "test");
        assert_eq!(tokenizer.vocabulary.size(), 0);
        assert!(tokenizer.merges.is_empty());
    }

    #[test]
    fn test_pre_tokenize() {
        let tokenizer = BpeTokenizer::new("test".to_string());
        let text = "Hello, world!";
        let words = tokenizer.pre_tokenize(text);
        assert!(!words.is_empty());
    }

    #[test]
    fn test_word_to_bytes() {
        let bytes = BpeTokenizer::word_to_bytes("A");
        assert_eq!(bytes, vec!["41"]); // 'A' is 0x41 in hex
    }

    #[test]
    fn test_initialize_byte_vocab() {
        let vocab = BpeTokenizer::initialize_byte_vocab();
        assert_eq!(vocab.size(), 256); // 0-255 bytes
    }

    #[test]
    fn test_validation() {
        let tokenizer = BpeTokenizer::new("test".to_string());
        assert!(tokenizer.validate().is_ok());
    }

    #[test]
    fn test_custom_pattern() {
        let pattern = r"\w+|\W+";
        let tokenizer = BpeTokenizer::with_pattern("test".to_string(), pattern).unwrap();
        assert_eq!(tokenizer.model_name, "test");
    }

    #[test]
    fn test_invalid_pattern() {
        let result = BpeTokenizer::with_pattern("test".to_string(), r"[invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_bpe_edge_cases() {
        let tokenizer = BpeTokenizer::new("test".to_string());

        // Test with empty text
        let result = tokenizer.encode_bpe("");
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());

        // Test with very long text
        let long_text = "word ".repeat(1000);
        let result = tokenizer.encode_bpe(&long_text);
        assert!(result.is_ok());

        // Test with special characters
        let special_text = "Hello, world! 🚀 你好 🌟";
        let result = tokenizer.encode_bpe(special_text);
        assert!(result.is_ok());

        // Test pre-tokenization edge cases
        let text_with_newlines = "Hello\nworld\r\ntest";
        let words = tokenizer.pre_tokenize(text_with_newlines);
        assert!(!words.is_empty());

        // Test with only whitespace
        let whitespace_text = "   \n\t  ";
        let words = tokenizer.pre_tokenize(whitespace_text);
        // Note: Regex captures whitespace sequences, so we get some tokens
        assert!(!words.is_empty()); // Regex captures whitespace sequences
    }

    #[test]
    fn test_bpe_training_edge_cases() {
        let mut tokenizer = BpeTokenizer::new("test".to_string());

        // Test training with empty corpus
        let result = tokenizer.train(&[], 100, Some(10));
        assert!(result.is_err());

        // Test training with very small corpus
        let corpus = vec!["a".to_string()];
        let result = tokenizer.train(&corpus, 100, Some(10));
        assert!(result.is_ok());

        // Test training with repeated tokens
        let corpus = vec!["hello world".to_string(); 100];
        let result = tokenizer.train(&corpus, 50, Some(10));
        assert!(result.is_ok());
        assert!(tokenizer.vocabulary.size() > 256); // Should have added tokens beyond base bytes
    }

    #[test]
    fn test_bpe_merge_operations() {
        let mut tokenizer = BpeTokenizer::new("test".to_string());

        // Add some initial vocabulary
        tokenizer.vocabulary.add_token("he".to_string());
        tokenizer.vocabulary.add_token("ll".to_string());
        tokenizer.vocabulary.add_token("o".to_string());

        // Test merge pair operation
        let word_tokens = [
            vec!["he".to_string(), "ll".to_string()],
            vec!["o".to_string()],
        ];

        let mut word_tokens_map = AHashMap::default();
        word_tokens_map.insert("hello".to_string(), word_tokens[0].clone());
        word_tokens_map.insert("o".to_string(), word_tokens[1].clone());

        let mut word_freqs = AHashMap::default();
        word_freqs.insert("hello".to_string(), 5);
        word_freqs.insert("o".to_string(), 3);

        let pair_stats = BpeTokenizer::get_pair_stats(&word_tokens_map, &word_freqs);
        assert!(!pair_stats.is_empty());

        // Test merge operation
        let best_pair = ("he".to_string(), "ll".to_string());
        BpeTokenizer::merge_pair(&mut word_tokens_map, &best_pair, "hell");

        // Verify merge occurred
        if let Some(tokens) = word_tokens_map.get("hello") {
            assert!(tokens.contains(&"hell".to_string()));
        }
    }

    #[test]
    fn test_bpe_decode_edge_cases() {
        let tokenizer = BpeTokenizer::new("test".to_string());

        // Test decode with empty tokens
        let result = tokenizer.decode_bpe(&[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "");

        // Test decode with invalid byte sequences
        let tokens = vec!["zz".to_string()]; // Invalid hex
        let result = tokenizer.decode_bpe(&tokens);
        assert!(result.is_ok()); // Should handle gracefully

        // Test decode with mixed valid/invalid
        let tokens = vec!["48".to_string(), "invalid".to_string(), "65".to_string()]; // 'H', invalid, 'e'
        let result = tokenizer.decode_bpe(&tokens);
        assert!(result.is_ok());
    }
}
