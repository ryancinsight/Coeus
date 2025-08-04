//! Basic tokenizer plugin implementation.

use rustllm_core::core::plugin::{Plugin, TokenizerPlugin};
use rustllm_core::core::tokenizer::{Token, Tokenizer, StringToken};
use rustllm_core::foundation::{
    error::Result,
    iterator::TokenIterator,
    types::Version,
};

/// Basic whitespace tokenizer plugin.
#[derive(Debug, Default)]
pub struct BasicTokenizerPlugin;

impl Plugin for BasicTokenizerPlugin {
    fn name(&self) -> &str {
        "basic_tokenizer"
    }
    
    fn version(&self) -> Version {
        Version::new(0, 1, 0)
    }
    
    fn description(&self) -> &str {
        "Basic whitespace tokenizer"
    }
    
    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }
    
    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
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
    
    fn tokenize<'a>(&self, input: &'a str) -> TokenIterator<'a, Self::Token> {
        Box::new(
            input
                .split_whitespace()
                .map(|s| StringToken::new(s.to_string()))
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
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_tokenizer() {
        let tokenizer = BasicTokenizer::new();
        let tokens: Vec<_> = tokenizer.tokenize("hello world rust").collect();
        
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].as_str(), Some("hello"));
        assert_eq!(tokens[1].as_str(), Some("world"));
        assert_eq!(tokens[2].as_str(), Some("rust"));
        
        let decoded = tokenizer.decode(tokens).unwrap();
        assert_eq!(decoded, "hello world rust");
    }
}