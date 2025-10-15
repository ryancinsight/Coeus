//! Encoding structures for tokenized text.

use serde::{Deserialize, Serialize};

/// Single text encoding result.
///
/// Contains the tokenized representation of input text with all necessary metadata
/// for downstream processing in transformer models.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Encoding {
    /// Token IDs for model input.
    pub ids: Vec<u32>,
    /// Original token strings.
    pub tokens: Vec<String>,
    /// Character offsets (start, end) for each token.
    pub offsets: Vec<(usize, usize)>,
    /// Attention mask (1 for real tokens, 0 for padding).
    pub attention_mask: Vec<u32>,
    /// Token type IDs for multi-sequence inputs.
    pub token_type_ids: Vec<u32>,
    /// Special tokens mask (1 for special tokens, 0 for regular tokens).
    pub special_tokens_mask: Vec<u32>,
    /// Original input text length.
    pub length: usize,
}

impl Encoding {
    /// Create a new encoding.
    #[must_use]
    pub fn new(
        ids: Vec<u32>,
        tokens: Vec<String>,
        offsets: Vec<(usize, usize)>,
        attention_mask: Vec<u32>,
        token_type_ids: Vec<u32>,
        special_tokens_mask: Vec<u32>,
        length: usize,
    ) -> Self {
        Self {
            ids,
            tokens,
            offsets,
            attention_mask,
            token_type_ids,
            special_tokens_mask,
            length,
        }
    }

    /// Create a basic encoding with defaults.
    #[must_use]
    pub fn from_tokens(ids: Vec<u32>, tokens: Vec<String>, length: usize) -> Self {
        let n_tokens = ids.len();
        Self {
            ids,
            tokens,
            offsets: vec![(0, 0); n_tokens], // Placeholder offsets
            attention_mask: vec![1; n_tokens],
            token_type_ids: vec![0; n_tokens],
            special_tokens_mask: vec![0; n_tokens],
            length,
        }
    }

    /// Get the number of tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if encoding is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Truncate encoding to maximum length.
    pub fn truncate(&mut self, max_len: usize) {
        if self.len() > max_len {
            self.ids.truncate(max_len);
            self.tokens.truncate(max_len);
            self.offsets.truncate(max_len);
            self.attention_mask.truncate(max_len);
            self.token_type_ids.truncate(max_len);
            self.special_tokens_mask.truncate(max_len);
        }
    }

    /// Pad encoding to specified length.
    pub fn pad(&mut self, target_len: usize, pad_token_id: u32) {
        let pad_len = target_len.saturating_sub(self.len());
        if pad_len > 0 {
            self.ids
                .extend(std::iter::repeat(pad_token_id).take(pad_len));
            self.tokens
                .extend(std::iter::repeat("[PAD]".to_string()).take(pad_len));
            self.offsets.extend(std::iter::repeat((0, 0)).take(pad_len));
            self.attention_mask
                .extend(std::iter::repeat(0).take(pad_len));
            self.token_type_ids
                .extend(std::iter::repeat(0).take(pad_len));
            self.special_tokens_mask
                .extend(std::iter::repeat(1).take(pad_len));
        }
    }
}

/// Batch encoding result for multiple texts.
///
/// Contains encodings for a batch of texts with consistent padding/truncation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchEncoding {
    /// Individual encodings for each text in the batch.
    pub encodings: Vec<Encoding>,
    /// Maximum sequence length in the batch.
    pub max_len: usize,
    /// Padding token ID used.
    pub pad_token_id: u32,
    /// Whether padding was applied.
    pub padded: bool,
    /// Whether truncation was applied.
    pub truncated: bool,
}

impl BatchEncoding {
    /// Create a new batch encoding.
    #[must_use]
    pub fn new(
        encodings: Vec<Encoding>,
        max_len: usize,
        pad_token_id: u32,
        padded: bool,
        truncated: bool,
    ) -> Self {
        Self {
            encodings,
            max_len,
            pad_token_id,
            padded,
            truncated,
        }
    }

    /// Get batch size.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.encodings.len()
    }

    /// Get all token IDs as a 2D vector [`batch_size`][seq_len].
    #[must_use]
    pub fn input_ids(&self) -> Vec<Vec<u32>> {
        self.encodings.iter().map(|e| e.ids.clone()).collect()
    }

    /// Get all attention masks as a 2D vector [`batch_size`][seq_len].
    #[must_use]
    pub fn attention_mask(&self) -> Vec<Vec<u32>> {
        self.encodings
            .iter()
            .map(|e| e.attention_mask.clone())
            .collect()
    }

    /// Get all token type IDs as a 2D vector [`batch_size`][seq_len].
    #[must_use]
    pub fn token_type_ids(&self) -> Vec<Vec<u32>> {
        self.encodings
            .iter()
            .map(|e| e.token_type_ids.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_creation() {
        let encoding = Encoding::from_tokens(
            vec![1, 2, 3],
            vec!["hello".to_string(), "world".to_string(), "!".to_string()],
            12,
        );
        assert_eq!(encoding.len(), 3);
        assert_eq!(encoding.ids, vec![1, 2, 3]);
        assert_eq!(encoding.tokens, vec!["hello", "world", "!"]);
        assert_eq!(encoding.length, 12);
    }

    #[test]
    fn test_encoding_truncate() {
        let mut encoding = Encoding::from_tokens(
            vec![1, 2, 3, 4],
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            10,
        );
        encoding.truncate(2);
        assert_eq!(encoding.len(), 2);
        assert_eq!(encoding.ids, vec![1, 2]);
    }

    #[test]
    fn test_encoding_pad() {
        let mut encoding = Encoding::from_tokens(
            vec![1, 2],
            vec!["hello".to_string(), "world".to_string()],
            10,
        );
        encoding.pad(4, 0);
        assert_eq!(encoding.len(), 4);
        assert_eq!(encoding.ids, vec![1, 2, 0, 0]);
        assert_eq!(encoding.attention_mask, vec![1, 1, 0, 0]);
    }
}
