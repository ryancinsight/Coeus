//! Tokenizer traits and abstractions.
//!
//! This module defines the core tokenizer interfaces following the
//! Single Responsibility Principle (SRP) and Interface Segregation Principle (ISP).

use crate::foundation::{
    error::Result,
    iterator::TokenIterator,
    types::{TokenId, VocabSize},
};
use core::fmt::Debug;

/// Trait representing a token in the tokenization process.
pub trait Token: Debug + Clone + Send + Sync {
    /// Returns the token as a string slice if possible.
    fn as_str(&self) -> Option<&str>;
    
    /// Returns the token ID if available.
    fn id(&self) -> Option<TokenId>;
    
    /// Returns the byte representation of the token.
    fn as_bytes(&self) -> &[u8];
    
    /// Returns the length of the token in bytes.
    fn len(&self) -> usize {
        self.as_bytes().len()
    }
    
    /// Checks if the token is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Basic string token implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
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
}

impl Token for StringToken {
    fn as_str(&self) -> Option<&str> {
        Some(&self.value)
    }
    
    fn id(&self) -> Option<TokenId> {
        self.id
    }
    
    fn as_bytes(&self) -> &[u8] {
        self.value.as_bytes()
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
    fn as_str(&self) -> Option<&str> {
        self.token.as_str()
    }
    
    fn id(&self) -> Option<TokenId> {
        self.token.id()
    }
    
    fn as_bytes(&self) -> &[u8] {
        self.token.as_bytes()
    }
}

/// Main tokenizer trait.
pub trait Tokenizer: Send + Sync {
    /// The token type produced by this tokenizer.
    type Token: Token;
    
    /// The error type for this tokenizer.
    #[cfg(feature = "std")]
    type Error: std::error::Error + Send + Sync + 'static;
    
    #[cfg(not(feature = "std"))]
    type Error: core::fmt::Debug + core::fmt::Display + Send + Sync + 'static;
    
    /// Tokenizes the input text into an iterator of tokens.
    fn tokenize<'a>(&self, input: &'a str) -> TokenIterator<'a, Self::Token>;
    
    /// Decodes tokens back into text.
    fn decode<I>(&self, tokens: I) -> Result<String>
    where
        I: IntoIterator<Item = Self::Token>;
    
    /// Returns the vocabulary size if known.
    fn vocab_size(&self) -> Option<VocabSize> {
        None
    }
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
        // Note: This leaks the normalized string, but it's a simplified example
        // In production, you'd need a better solution
        let leaked = Box::leak(normalized.into_boxed_str());
        self.tokenize(leaked)
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
        assert_eq!(token.id(), None);
        assert_eq!(token.as_bytes(), b"hello");
        assert_eq!(token.len(), 5);
        assert!(!token.is_empty());
        
        let token_with_id = StringToken::with_id("world".to_string(), 42);
        assert_eq!(token_with_id.id(), Some(42));
    }
    
    #[test]
    fn test_positioned_token() {
        let inner = StringToken::new("test".to_string());
        let positioned = PositionedToken::new(inner, 10, 14);
        
        assert_eq!(positioned.start(), 10);
        assert_eq!(positioned.end(), 14);
        assert_eq!(positioned.as_str(), Some("test"));
    }
    
    #[test]
    fn test_tokenizer_config() {
        let config = TokenizerConfig::default();
        assert_eq!(config.max_length, None);
        assert!(config.add_special_tokens);
        assert!(!config.lowercase);
        assert_eq!(config.padding, PaddingStrategy::None);
    }
}