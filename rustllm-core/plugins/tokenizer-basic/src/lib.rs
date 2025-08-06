//! Basic tokenizer plugin implementation.

use rustllm_core::core::{
    plugin::{Plugin, TokenizerPlugin, PluginCapabilities},
    tokenizer::{Tokenizer, Token, StringToken, TokenIterator},
    traits::{Identity, Versioned},
};
use rustllm_core::foundation::{
    error::Result,
    types::{Version, VocabSize},
};
use std::borrow::Cow;

/// Basic whitespace tokenizer plugin.
#[derive(Debug, Default)]
pub struct BasicTokenizerPlugin;

impl Identity for BasicTokenizerPlugin {
    fn id(&self) -> &str {
        "basic_tokenizer"
    }
}

impl Versioned for BasicTokenizerPlugin {
    fn version(&self) -> Version {
        Version::new(0, 1, 0)
    }
}

impl Plugin for BasicTokenizerPlugin {
    fn capabilities(&self) -> PluginCapabilities {
        PluginCapabilities::standard()
            .with_feature("tokenization")
            .with_feature("whitespace")
    }
}

impl TokenizerPlugin for BasicTokenizerPlugin {
    type Tokenizer = BasicTokenizer;
    
    fn create_tokenizer(&self) -> Result<Self::Tokenizer> {
        Ok(BasicTokenizer::new())
    }
}

/// Basic whitespace tokenizer.
#[derive(Debug, Clone)]
pub struct BasicTokenizer;

impl BasicTokenizer {
    /// Creates a new basic tokenizer.
    pub fn new() -> Self {
        Self
    }
}

impl Default for BasicTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for BasicTokenizer {
    type Token = StringToken;
    type Error = std::io::Error;
    
    fn tokenize<'a>(&self, input: Cow<'a, str>) -> TokenIterator<'a, Self::Token> {
        // Convert to owned string to ensure lifetime independence
        let owned_input = input.into_owned();
        Box::new(
            owned_input
                .split_whitespace()
                .map(|s| StringToken::new(s.to_string()))
                .collect::<Vec<_>>()
                .into_iter()
        )
    }
    
    fn decode<I>(&self, tokens: I) -> Result<String>
    where
        I: IntoIterator<Item = Self::Token>,
    {
        Ok(tokens
            .into_iter()
            .filter_map(|t| t.as_str().map(|s| s.to_string()))
            .collect::<Vec<_>>()
            .join(" "))
    }
    
    fn vocab_size(&self) -> VocabSize {
        0 // Basic tokenizer has no fixed vocabulary
    }
    
    fn name(&self) -> &str {
        "basic_tokenizer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_tokenizer() {
        let tokenizer = BasicTokenizer::new();
        let tokens: Vec<_> = tokenizer.tokenize_str("hello world rust").collect();
        
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].as_str(), Some("hello"));
        assert_eq!(tokens[1].as_str(), Some("world"));
        assert_eq!(tokens[2].as_str(), Some("rust"));
        
        let decoded = tokenizer.decode(tokens).unwrap();
        assert_eq!(decoded, "hello world rust");
    }
}