//! CLIP tokenizer implementation (placeholder)

use crate::error::{Result, TokenizerError};

/// CLIP tokenizer placeholder
pub struct CLIPTokenizer;

impl CLIPTokenizer {
    /// Create a new CLIP tokenizer
    ///
    /// # Errors
    /// Returns an error since this is not yet implemented
    pub fn new() -> Result<Self> {
        Err(TokenizerError::model_error(
            "CLIP tokenizer not yet implemented".to_string(),
        ))
    }
}
