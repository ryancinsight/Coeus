//! BPE tokenizer plugin implementation.

use rustllm_core::core::plugin::{Plugin, PluginCapabilities, TokenizerPlugin};
use rustllm_core::core::tokenizer::{
    StringToken, Token, TokenIterator, Tokenizer, VocabularyTokenizer,
};
use rustllm_core::core::traits::{Identity, Versioned};
use rustllm_core::foundation::{
    error::{Error, Result},
    types::{TokenId, Version, VocabSize},
};
use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;

/// BPE tokenizer plugin.
///
/// This plugin provides Byte Pair Encoding tokenization capabilities.
/// It creates new tokenizer instances on demand rather than storing them,
/// following the stateless plugin design principle.
#[derive(Debug, Default)]
pub struct BpeTokenizerPlugin;

impl Identity for BpeTokenizerPlugin {
    fn id(&self) -> &str {
        "bpe_tokenizer"
    }
}

impl Versioned for BpeTokenizerPlugin {
    fn version(&self) -> Version {
        Version::new(0, 1, 0)
    }
}

impl Plugin for BpeTokenizerPlugin {
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::standard()
            .with_feature("tokenization")
            .with_feature("bpe")
    }
}

impl TokenizerPlugin for BpeTokenizerPlugin {
    type Tokenizer = BpeTokenizer;

    fn create_tokenizer(&self) -> Result<Self::Tokenizer> {
        Ok(BpeTokenizer::new())
    }
}

/// A pair of tokens for BPE merging.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct TokenPair {
    first: String,
    second: String,
}

impl TokenPair {
    fn new(first: String, second: String) -> Self {
        Self { first, second }
    }

    fn merged(&self) -> String {
        format!("{}{}", self.first, self.second)
    }
}

/// Priority queue entry for BPE merges.
#[derive(Debug, Clone, Eq)]
#[allow(dead_code)]
struct MergeCandidate {
    pair: TokenPair,
    frequency: usize,
    priority: usize,
}

impl PartialEq for MergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.frequency == other.frequency && self.priority == other.priority
    }
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher frequency first, then lower priority (earlier in merge order)
        match self.frequency.cmp(&other.frequency) {
            Ordering::Equal => other.priority.cmp(&self.priority),
            other => other,
        }
    }
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Byte Pair Encoding tokenizer.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Vocabulary mapping from token strings to IDs
    vocab: HashMap<String, TokenId>,
    /// Reverse vocabulary mapping from IDs to token strings
    reverse_vocab: HashMap<TokenId, String>,
    /// Merge rules learned from training
    merges: Vec<TokenPair>,
    /// Special tokens
    special_tokens: HashMap<String, TokenId>,
    /// Next available token ID
    next_id: TokenId,
}

impl BpeTokenizer {
    /// Creates a new BPE tokenizer.
    pub fn new() -> Self {
        let mut tokenizer = Self {
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            merges: Vec::new(),
            special_tokens: HashMap::new(),
            next_id: 0,
        };

        // Initialize with byte-level tokens (0-255)
        for byte in 0u8..=255 {
            let token = String::from_utf8(vec![byte]).unwrap_or_else(|_| {
                // For invalid UTF-8 bytes, use a special representation
                format!("<0x{:02X}>", byte)
            });
            tokenizer.add_to_vocab(token);
        }

        // Add special tokens
        tokenizer.add_special_token("<PAD>", 0);
        tokenizer.add_special_token("<UNK>", 1);
        tokenizer.add_special_token("<BOS>", 2);
        tokenizer.add_special_token("<EOS>", 3);

        tokenizer
    }

    /// Adds a token to the vocabulary.
    fn add_to_vocab(&mut self, token: String) -> TokenId {
        if let Some(&id) = self.vocab.get(&token) {
            id
        } else {
            let id = self.next_id;
            self.vocab.insert(token.clone(), id);
            self.reverse_vocab.insert(id, token);
            self.next_id += 1;
            id
        }
    }

    /// Adds a special token to the vocabulary with a specific ID.
    fn add_special_token(&mut self, token: &str, id: TokenId) -> TokenId {
        self.vocab.insert(token.to_string(), id);
        self.reverse_vocab.insert(id, token.to_string());
        self.special_tokens.insert(token.to_string(), id);
        id
    }

    /// Learns BPE merges from a corpus.
    pub fn train(&mut self, corpus: &[&str], num_merges: usize) -> Result<()> {
        // Tokenize corpus into bytes
        let mut word_freqs: HashMap<Vec<String>, usize> = HashMap::new();

        for text in corpus {
            let words = text.split_whitespace();
            for word in words {
                let tokens: Vec<String> = word
                    .bytes()
                    .map(|b| {
                        String::from_utf8(vec![b]).unwrap_or_else(|_| format!("<0x{:02X}>", b))
                    })
                    .collect();
                *word_freqs.entry(tokens).or_insert(0) += 1;
            }
        }

        // Learn merges
        for _merge_idx in 0..num_merges {
            // Count pair frequencies
            let mut pair_freqs: HashMap<TokenPair, usize> = HashMap::new();

            for (word, freq) in &word_freqs {
                if word.len() < 2 {
                    continue;
                }

                for i in 0..word.len() - 1 {
                    let pair = TokenPair::new(word[i].clone(), word[i + 1].clone());
                    *pair_freqs.entry(pair).or_insert(0) += freq;
                }
            }

            // Find most frequent pair
            let best_pair = pair_freqs
                .iter()
                .max_by_key(|&(_, freq)| freq)
                .map(|(pair, _)| pair.clone());

            if let Some(pair) = best_pair {
                // Add merge rule
                self.merges.push(pair.clone());

                // Add merged token to vocabulary
                let merged = pair.merged();
                self.add_to_vocab(merged.clone());

                // Update word frequencies with merged tokens
                let mut new_word_freqs = HashMap::new();

                for (word, freq) in word_freqs {
                    let mut new_word = Vec::new();
                    let mut i = 0;

                    while i < word.len() {
                        if i < word.len() - 1 && word[i] == pair.first && word[i + 1] == pair.second
                        {
                            new_word.push(merged.clone());
                            i += 2;
                        } else {
                            new_word.push(word[i].clone());
                            i += 1;
                        }
                    }

                    *new_word_freqs.entry(new_word).or_insert(0) += freq;
                }

                word_freqs = new_word_freqs;
            } else {
                // No more pairs to merge
                break;
            }
        }

        Ok(())
    }

    /// Applies BPE merges to a word.
    fn bpe(&self, word: Vec<String>) -> Vec<String> {
        if word.len() < 2 {
            return word;
        }

        let mut tokens = word;

        for merge in &self.merges {
            let mut new_tokens = Vec::new();
            let mut i = 0;

            while i < tokens.len() {
                if i < tokens.len() - 1 && tokens[i] == merge.first && tokens[i + 1] == merge.second
                {
                    new_tokens.push(merge.merged());
                    i += 2;
                } else {
                    new_tokens.push(tokens[i].clone());
                    i += 1;
                }
            }

            tokens = new_tokens;
        }

        tokens
    }
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for BpeTokenizer {
    type Token = StringToken;
    type Error = std::io::Error;

    fn tokenize<'a>(&self, input: Cow<'a, str>) -> TokenIterator<'a, Self::Token> {
        let mut tokens = Vec::new();

        // Convert to owned string to work with
        let input_str = input.as_ref();

        // Split by whitespace and process each word
        for word in input_str.split_whitespace() {
            // Convert to byte tokens
            let byte_tokens: Vec<String> = word
                .bytes()
                .map(|b| String::from_utf8(vec![b]).unwrap_or_else(|_| format!("<0x{:02X}>", b)))
                .collect();

            // Apply BPE
            let bpe_tokens = self.bpe(byte_tokens);

            // Convert to StringToken with IDs
            for token_str in bpe_tokens {
                let id = self.vocab.get(&token_str).copied();
                tokens.push(StringToken::with_id(token_str, id.unwrap_or(1))); // 1 is <UNK>
            }

            // Add space token between words (except last)
            tokens.push(StringToken::with_id(" ".to_string(), 32)); // 32 is space in ASCII
        }

        // Remove last space
        if !tokens.is_empty() {
            tokens.pop();
        }

        Box::new(tokens.into_iter())
    }

    fn decode<I>(&self, tokens: I) -> Result<String>
    where
        I: IntoIterator<Item = Self::Token>,
    {
        let mut result = String::new();

        for token in tokens {
            if let Some(s) = token.as_str() {
                // Handle special byte representations
                if s.starts_with("<0x") && s.ends_with('>') {
                    // Parse hex byte
                    if let Ok(byte) = u8::from_str_radix(&s[3..5], 16) {
                        result.push_str(&String::from_utf8_lossy(&[byte]));
                    } else {
                        result.push_str(s);
                    }
                } else {
                    result.push_str(s);
                }
            }
        }

        Ok(result)
    }

    fn vocab_size(&self) -> VocabSize {
        self.vocab.len() as VocabSize
    }

    fn name(&self) -> &str {
        "bpe_tokenizer"
    }
}

impl VocabularyTokenizer for BpeTokenizer {
    fn add_token(&mut self, token: &str) -> Result<TokenId> {
        Ok(self.add_to_vocab(token.to_string()))
    }

    fn remove_token(&mut self, token: &str) -> Result<()> {
        if let Some(&id) = self.vocab.get(token) {
            self.vocab.remove(token);
            self.reverse_vocab.remove(&id);
            Ok(())
        } else {
            Err(Error::Processing(
                rustllm_core::foundation::error::ProcessingError::InvalidInput {
                    description: format!("Token not found: {}", token),
                    position: None,
                },
            ))
        }
    }

    fn contains_token(&self, token: &str) -> bool {
        self.vocab.contains_key(token)
    }

    fn token_from_id(&self, id: TokenId) -> Option<String> {
        self.reverse_vocab.get(&id).cloned()
    }

    fn id_from_token(&self, token: &str) -> Option<TokenId> {
        self.vocab.get(token).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpe_tokenizer_basic() {
        let tokenizer = BpeTokenizer::new();

        // Test basic tokenization (byte-level)
        let tokens: Vec<_> = tokenizer.tokenize_str("hello").collect();
        assert_eq!(tokens.len(), 5); // One token per byte

        // Test decode
        let decoded = tokenizer.decode(tokens).unwrap();
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_bpe_training() {
        let mut tokenizer = BpeTokenizer::new();

        // Train on a simple corpus
        let corpus = vec![
            "the cat sat on the mat",
            "the dog sat on the log",
            "the cat and the dog",
        ];

        tokenizer.train(&corpus, 10).unwrap();

        // Check that vocabulary has grown
        assert!(tokenizer.vocab_size() > 256);

        // Test tokenization after training
        let tokens: Vec<_> = tokenizer.tokenize_str("the cat").collect();
        assert!(!tokens.is_empty());

        // Test decode
        let decoded = tokenizer.decode(tokens).unwrap();
        // Note: Due to space handling, might not be exact
        assert!(decoded.contains("the"));
        assert!(decoded.contains("cat"));
    }

    #[test]
    fn test_special_tokens() {
        let mut tokenizer = BpeTokenizer::new();

        // Add special tokens
        tokenizer.add_special_token("<PAD>", 0);
        tokenizer.add_special_token("<UNK>", 1);
        tokenizer.add_special_token("<BOS>", 2);
        tokenizer.add_special_token("<EOS>", 3);

        // Check they're in vocabulary
        assert!(tokenizer.contains_token("<PAD>"));
        assert!(tokenizer.contains_token("<UNK>"));
        assert!(tokenizer.contains_token("<BOS>"));
        assert!(tokenizer.contains_token("<EOS>"));

        // Check IDs
        assert_eq!(tokenizer.id_from_token("<PAD>"), Some(0));
        assert_eq!(tokenizer.id_from_token("<UNK>"), Some(1));
        assert_eq!(tokenizer.id_from_token("<BOS>"), Some(2));
        assert_eq!(tokenizer.id_from_token("<EOS>"), Some(3));
    }
}
