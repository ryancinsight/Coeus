//! BERT tokenizer implementation (placeholder)

use crate::error::{Result, TokenizerError};

/// BERT tokenizer placeholder
pub struct BERTTokenizer;

impl BERTTokenizer {
    /// Create a new BERT tokenizer
    ///
    /// # Errors
    /// Returns an error since this is not yet implemented
    pub fn new(_model_name: &str) -> Result<Self> {
        Err(TokenizerError::model_error(
            "BERT tokenizer not yet implemented".to_string(),
        ))
    }
}
