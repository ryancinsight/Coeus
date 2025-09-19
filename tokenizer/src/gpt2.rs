//! GPT-2 tokenizer implementation (placeholder)

use crate::error::{Result, TokenizerError};

/// GPT-2 tokenizer placeholder
pub struct GPT2Tokenizer;

impl GPT2Tokenizer {
    /// Create a new GPT-2 tokenizer
    ///
    /// # Errors
    /// Returns an error since this is not yet implemented
    pub fn new() -> Result<Self> {
        Err(TokenizerError::model_error(
            "GPT-2 tokenizer not yet implemented".to_string(),
        ))
    }
}
