//! Tokenizer traits and abstractions.
//!
//! This module defines the core tokenizer interfaces following the
//! Single Responsibility Principle (SRP) and Interface Segregation Principle (ISP).

use crate::foundation::{
    error::Result,
    types::{TokenId, VocabSize},
};
use core::fmt::Debug;
use std::borrow::Cow;

/// Type alias for token iterators.
pub type TokenIterator<'a, T> = Box<dyn Iterator<Item = T> + 'a>;

/// Trait representing a token.
pub trait Token: Debug + Clone + Send + Sync {
    /// Returns the token ID.
    fn id(&self) -> TokenId;

    /// Returns the token as a string reference if available.
    fn as_str(&self) -> Option<&str>;

    /// Returns the token's byte representation if available.
    fn as_bytes(&self) -> Option<&[u8]>;
}

/// Basic string token implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct StringToken {
    value: String,
    id: Option<TokenId>,
}

impl StringToken {
    /// Creates a new string token.
    pub fn new(value: String) -> Self {
        Self { value, id: None }
    }

    /// Creates a new string token with an ID.
    pub fn with_id(value: String, id: TokenId) -> Self {
        Self {
            value,
            id: Some(id),
        }
    }

    /// Returns the length of the token in bytes.
    pub fn len(&self) -> usize {
        self.value.len()
    }

    /// Checks if the token is empty.
    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }
}

impl Token for StringToken {
    fn id(&self) -> TokenId {
        self.id.unwrap_or(0) // Return 0 as default ID instead of panicking
    }

    fn as_str(&self) -> Option<&str> {
        Some(&self.value)
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        Some(self.value.as_bytes())
    }
}

/// Token with position information.
#[derive(Debug, Clone)]
pub struct PositionedToken<T: Token> {
    token: T,
    start: usize,
    end: usize,
}

impl<T: Token> PositionedToken<T> {
    /// Creates a new positioned token.
    pub fn new(token: T, start: usize, end: usize) -> Self {
        Self { token, start, end }
    }

    /// Returns the inner token.
    pub fn token(&self) -> &T {
        &self.token
    }

    /// Returns the start position.
    pub fn start(&self) -> usize {
        self.start
    }

    /// Returns the end position.
    pub fn end(&self) -> usize {
        self.end
    }
}

impl<T: Token> Token for PositionedToken<T> {
    fn id(&self) -> TokenId {
        self.token.id()
    }

    fn as_str(&self) -> Option<&str> {
        self.token.as_str()
    }

    fn as_bytes(&self) -> Option<&[u8]> {
        self.token.as_bytes()
    }
}

/// Main tokenizer trait.
pub trait Tokenizer: Send + Sync {
    /// The token type produced by this tokenizer.
    type Token: Token;

    /// The error type for this tokenizer.
    type Error: Debug + Send + Sync + 'static;

    /// Tokenizes the input text.
    ///
    /// Accepts a `Cow<'_, str>` to handle both borrowed and owned strings efficiently.
    fn tokenize<'a>(&self, input: Cow<'a, str>) -> TokenIterator<'a, Self::Token>;

    /// Tokenizes the input text (convenience method for borrowed strings).
    fn tokenize_str<'a>(&self, input: &'a str) -> TokenIterator<'a, Self::Token> {
        self.tokenize(Cow::Borrowed(input))
    }

    /// Decodes tokens back into text.
    fn decode<I>(&self, tokens: I) -> Result<String>
    where
        I: IntoIterator<Item = Self::Token>;

    /// Returns the vocabulary size.
    fn vocab_size(&self) -> VocabSize;

    /// Returns the tokenizer name.
    fn name(&self) -> &str;
}

/// Trait for tokenizers that support vocabulary management.
pub trait VocabularyTokenizer: Tokenizer {
    /// Adds a token to the vocabulary.
    fn add_token(&mut self, token: &str) -> Result<TokenId>;

    /// Removes a token from the vocabulary.
    fn remove_token(&mut self, token: &str) -> Result<()>;

    /// Checks if a token exists in the vocabulary.
    fn contains_token(&self, token: &str) -> bool;

    /// Returns the token for a given ID.
    fn token_from_id(&self, id: TokenId) -> Option<String>;

    /// Returns the ID for a given token.
    fn id_from_token(&self, token: &str) -> Option<TokenId>;
}

/// Trait for tokenizers that support special tokens.
pub trait SpecialTokenizer: Tokenizer {
    /// Returns the padding token if available.
    fn pad_token(&self) -> Option<Self::Token>;

    /// Returns the unknown token if available.
    fn unk_token(&self) -> Option<Self::Token>;

    /// Returns the beginning of sequence token if available.
    fn bos_token(&self) -> Option<Self::Token>;

    /// Returns the end of sequence token if available.
    fn eos_token(&self) -> Option<Self::Token>;

    /// Returns the mask token if available.
    fn mask_token(&self) -> Option<Self::Token>;
}

/// Trait for tokenizers that support normalization.
pub trait NormalizingTokenizer: Tokenizer {
    /// Normalizes the input text before tokenization.
    fn normalize(&self, text: &str) -> String;

    /// Tokenizes with normalization.
    fn tokenize_normalized<'a>(&self, input: &'a str) -> TokenIterator<'a, Self::Token> {
        let normalized = self.normalize(input);
        // Use Cow::Owned to pass the normalized string without leaking
        self.tokenize(Cow::Owned(normalized))
    }
}

/// Trait for tokenizers that support pre-tokenization.
pub trait PreTokenizer: Send + Sync {
    /// Pre-tokenizes the input into chunks.
    fn pre_tokenize<'a>(&self, input: &'a str) -> Box<dyn Iterator<Item = &'a str> + 'a>;
}

/// Trait for tokenizers that support post-processing.
pub trait PostProcessor: Send + Sync {
    /// The token type.
    type Token: Token;

    /// Post-processes the tokens.
    fn post_process(&self, tokens: Vec<Self::Token>) -> Result<Vec<Self::Token>>;
}

/// Configuration for tokenizers.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Maximum sequence length.
    pub max_length: Option<usize>,

    /// Whether to add special tokens.
    pub add_special_tokens: bool,

    /// Whether to lowercase input.
    pub lowercase: bool,

    /// Whether to strip accents.
    pub strip_accents: bool,

    /// Padding strategy.
    pub padding: PaddingStrategy,

    /// Truncation strategy.
    pub truncation: TruncationStrategy,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            max_length: None,
            add_special_tokens: true,
            lowercase: false,
            strip_accents: false,
            padding: PaddingStrategy::None,
            truncation: TruncationStrategy::None,
        }
    }
}

/// Padding strategy for tokenization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// No padding.
    None,

    /// Pad to maximum length.
    MaxLength,

    /// Pad to longest sequence in batch.
    Longest,
}

/// Truncation strategy for tokenization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// No truncation.
    None,

    /// Truncate to maximum length.
    MaxLength,

    /// Truncate longest sequence first.
    LongestFirst,

    /// Only truncate second sequence in pair.
    OnlySecond,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_token() {
        let token = StringToken::new("hello".to_string());
        assert_eq!(token.as_str(), Some("hello"));
        assert_eq!(token.id(), 0); // Default ID
        assert_eq!(token.as_bytes(), Some(b"hello".as_ref()));
        assert_eq!(token.len(), 5);
        assert!(!token.is_empty());

        let token_with_id = StringToken::with_id("world".to_string(), 42);
        assert_eq!(token_with_id.id(), 42);
    }

    #[test]
    fn test_positioned_token() {
        let inner = StringToken::with_id("test".to_string(), 10);
        let positioned = PositionedToken::new(inner.clone(), 5, 9);

        assert_eq!(positioned.as_str(), Some("test"));
        assert_eq!(positioned.id(), 10);
        assert_eq!(positioned.start(), 5);
        assert_eq!(positioned.end(), 9);
        assert_eq!(positioned.token.len(), 4);
    }

    #[test]
    fn test_tokenizer_config() {
        let config = TokenizerConfig::default();
        assert_eq!(config.max_length, None);
        assert!(config.add_special_tokens);
        assert!(!config.lowercase);
        assert!(!config.strip_accents);
        assert_eq!(config.padding, PaddingStrategy::None);
        assert_eq!(config.truncation, TruncationStrategy::None);
    }
}
