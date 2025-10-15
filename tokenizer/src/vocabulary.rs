//! Vocabulary management for tokenizers.

use crate::error::{Result, TokenizerError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Vocabulary mapping tokens to IDs and vice versa.
///
/// Provides bidirectional mapping between token strings and their integer IDs,
/// along with special token management.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vocabulary {
    /// Mapping from token strings to IDs.
    token_to_id: HashMap<String, u32>,
    /// Mapping from IDs to token strings (for fast lookup).
    id_to_token: Vec<String>,
    /// Special tokens with reserved IDs.
    special_tokens: HashMap<String, u32>,
    /// Next available ID for new tokens.
    next_id: u32,
}

impl Vocabulary {
    /// Create a new empty vocabulary.
    #[must_use]
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
            special_tokens: HashMap::new(),
            next_id: 0,
        }
    }

    /// Create vocabulary from token-ID pairs.
    ///
    /// # Errors
    /// Returns an error if token addition fails due to conflicts.
    pub fn from_tokens(tokens: impl IntoIterator<Item = (String, u32)>) -> Result<Self> {
        let mut vocab = Self::new();
        for (token, id) in tokens {
            vocab.add_token(token, id)?;
        }
        Ok(vocab)
    }

    /// Add a token with specific ID.
    ///
    /// # Errors
    /// Returns error if token already exists or ID is already used.
    pub fn add_token(&mut self, token: String, id: u32) -> Result<()> {
        if self.token_to_id.contains_key(&token) {
            return Err(TokenizerError::vocabulary(format!(
                "Token '{token}' already exists"
            )));
        }

        // Ensure id_to_token has enough capacity
        if id as usize >= self.id_to_token.len() {
            self.id_to_token.resize(id as usize + 1, String::new());
        }

        if !self.id_to_token[id as usize].is_empty() {
            return Err(TokenizerError::vocabulary(format!("ID {id} already used")));
        }

        self.token_to_id.insert(token.clone(), id);
        self.id_to_token[id as usize] = token;
        self.next_id = self.next_id.max(id + 1);

        Ok(())
    }

    /// Add a special token with specific ID.
    ///
    /// # Errors
    /// Returns error if token already exists or ID is already used.
    pub fn add_special_token(&mut self, token: String, id: u32) -> Result<()> {
        self.add_token(token.clone(), id)?;
        self.special_tokens.insert(token, id);
        Ok(())
    }

    /// Get token ID for a token string.
    #[must_use]
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token string for an ID.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.id_to_token
            .get(id as usize)
            .cloned()
            .filter(|s| !s.is_empty())
    }

    /// Check if token is a special token.
    #[must_use]
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special_tokens.contains_key(token)
    }

    /// Get special token ID.
    #[must_use]
    pub fn special_token_id(&self, token: &str) -> Option<u32> {
        self.special_tokens.get(token).copied()
    }

    /// Get all special tokens.
    #[must_use]
    pub fn special_tokens(&self) -> &HashMap<String, u32> {
        &self.special_tokens
    }

    /// Get vocabulary size (number of tokens).
    #[must_use]
    pub fn size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Get next available ID.
    #[must_use]
    pub fn next_id(&self) -> u32 {
        self.next_id
    }

    /// Get all tokens as a vector.
    #[must_use]
    pub fn tokens(&self) -> Vec<String> {
        self.token_to_id.keys().cloned().collect()
    }

    /// Merge another vocabulary into this one.
    ///
    /// # Errors
    /// Returns error if there are conflicts.
    pub fn merge(&mut self, other: &Vocabulary) -> Result<()> {
        for (token, &id) in &other.token_to_id {
            if let Some(existing_id) = self.token_to_id.get(token) {
                if *existing_id != id {
                    return Err(TokenizerError::vocabulary(format!(
                        "Token '{token}' has conflicting IDs: {existing_id} vs {id}"
                    )));
                }
            } else {
                self.add_token(token.clone(), id)?;
            }
        }

        // Merge special tokens
        for (token, &id) in &other.special_tokens {
            if !self.special_tokens.contains_key(token) {
                self.special_tokens.insert(token.clone(), id);
            }
        }

        Ok(())
    }

    /// Create vocabulary from JSON.
    ///
    /// # Errors
    /// Returns an error if JSON parsing fails.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(TokenizerError::from)
    }

    /// Serialize vocabulary to JSON.
    ///
    /// # Errors
    /// Returns an error if JSON serialization fails.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(TokenizerError::from)
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_basic() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("hello".to_string(), 0).unwrap();
        vocab.add_token("world".to_string(), 1).unwrap();

        assert_eq!(vocab.token_to_id("hello"), Some(0));
        assert_eq!(vocab.token_to_id("world"), Some(1));
        assert_eq!(vocab.token_to_id("unknown"), None);

        assert_eq!(vocab.id_to_token(0), Some("hello".to_string()));
        assert_eq!(vocab.id_to_token(1), Some("world".to_string()));
        assert_eq!(vocab.id_to_token(2), None);

        assert_eq!(vocab.size(), 2);
    }

    #[test]
    fn test_vocabulary_special_tokens() {
        let mut vocab = Vocabulary::new();
        vocab.add_special_token("[PAD]".to_string(), 0).unwrap();
        vocab.add_special_token("[UNK]".to_string(), 1).unwrap();

        assert!(vocab.is_special_token("[PAD]"));
        assert!(vocab.is_special_token("[UNK]"));
        assert!(!vocab.is_special_token("hello"));

        assert_eq!(vocab.special_token_id("[PAD]"), Some(0));
        assert_eq!(vocab.special_tokens().len(), 2);
    }

    #[test]
    fn test_vocabulary_duplicate_token() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("hello".to_string(), 0).unwrap();
        assert!(vocab.add_token("hello".to_string(), 1).is_err());
    }

    #[test]
    fn test_vocabulary_duplicate_id() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("hello".to_string(), 0).unwrap();
        vocab.add_token("world".to_string(), 0).unwrap_err();
    }

    #[test]
    fn test_vocabulary_merge() {
        let mut vocab1 = Vocabulary::new();
        vocab1.add_token("hello".to_string(), 0).unwrap();

        let mut vocab2 = Vocabulary::new();
        vocab2.add_token("world".to_string(), 1).unwrap();

        vocab1.merge(&vocab2).unwrap();
        assert_eq!(vocab1.size(), 2);
        assert_eq!(vocab1.token_to_id("world"), Some(1));
    }

    #[test]
    fn test_vocabulary_serialization() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("hello".to_string(), 0).unwrap();
        vocab.add_special_token("[PAD]".to_string(), 1).unwrap();

        let json = vocab.to_json().unwrap();
        let deserialized = Vocabulary::from_json(&json).unwrap();

        assert_eq!(vocab, deserialized);
    }
}
