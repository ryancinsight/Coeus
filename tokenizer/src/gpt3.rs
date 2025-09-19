//! GPT-3 tokenizer implementation (placeholder)

use crate::error::{Result, TokenizerError};

/// GPT-3 tokenizer placeholder
pub struct GPT3Tokenizer;

impl GPT3Tokenizer {
    /// Create a new GPT-3 tokenizer
    ///
    /// # Errors
    /// Returns an error since this is not yet implemented
    pub fn new(_model_name: &str) -> Result<Self> {
        Err(TokenizerError::model_error(
            "GPT-3 tokenizer not yet implemented".to_string(),
        ))
    }
}
