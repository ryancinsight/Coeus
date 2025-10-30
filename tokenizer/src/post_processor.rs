//! Post-processing pipeline for tokenized sequences.

use crate::encoding::Encoding;
use crate::error::{Result, TokenizerError};
use crate::vocabulary::Vocabulary;

/// Post-processor trait for sequence post-processing.
///
/// Handles special token insertion, padding, truncation, and other
/// sequence-level transformations.
pub trait PostProcessor {
    /// Post-process an encoding.
    ///
    /// # Arguments
    /// * `encoding` - Input encoding to process
    /// * `vocab` - Vocabulary for special token lookup
    ///
    /// # Returns
    /// Processed encoding
    ///
    /// # Errors
    /// Returns `TokenizerError` if post-processing fails
    fn post_process(&self, encoding: Encoding, vocab: &Vocabulary) -> Result<Encoding>;
}

/// Template for post-processing with special tokens.
#[derive(Debug, Clone)]
pub struct TemplatePostProcessor {
    /// Special tokens to add at the beginning.
    pub prefix: Vec<String>,
    /// Special tokens to add at the end.
    pub suffix: Vec<String>,
}

impl TemplatePostProcessor {
    /// Create a new template post-processor.
    #[must_use]
    pub fn new(prefix: Vec<String>, suffix: Vec<String>) -> Self {
        Self { prefix, suffix }
    }

    /// Create BERT-style post-processor with \[CLS\] and \[SEP\].
    #[must_use]
    pub fn bert() -> Self {
        Self {
            prefix: vec!["[CLS]".to_string()],
            suffix: vec!["[SEP]".to_string()],
        }
    }

    /// Create GPT-style post-processor (no special tokens).
    #[must_use]
    pub fn gpt() -> Self {
        Self {
            prefix: Vec::new(),
            suffix: Vec::new(),
        }
    }
}

impl PostProcessor for TemplatePostProcessor {
    fn post_process(&self, mut encoding: Encoding, vocab: &Vocabulary) -> Result<Encoding> {
        // Add prefix tokens
        for token in &self.prefix {
            if let Some(token_id) = vocab.special_token_id(token) {
                encoding.ids.insert(0, token_id);
                encoding.tokens.insert(0, token.clone());
                encoding.offsets.insert(0, (0, 0));
                encoding.attention_mask.insert(0, 1);
                encoding.token_type_ids.insert(0, 0);
                encoding.special_tokens_mask.insert(0, 1);
            } else {
                return Err(TokenizerError::vocabulary(format!(
                    "Special token '{token}' not found in vocabulary"
                )));
            }
        }

        // Add suffix tokens
        for token in &self.suffix {
            if let Some(token_id) = vocab.special_token_id(token) {
                encoding.ids.push(token_id);
                encoding.tokens.push(token.clone());
                encoding.offsets.push((encoding.length, encoding.length));
                encoding.attention_mask.push(1);
                encoding.token_type_ids.push(0);
                encoding.special_tokens_mask.push(1);
            } else {
                return Err(TokenizerError::vocabulary(format!(
                    "Special token '{token}' not found in vocabulary"
                )));
            }
        }

        Ok(encoding)
    }
}

/// Padding and truncation post-processor.
#[derive(Debug, Clone)]
pub struct PaddingPostProcessor {
    /// Target sequence length.
    pub max_len: Option<usize>,
    /// Padding token ID.
    pub pad_token_id: u32,
    /// Padding direction.
    pub direction: PaddingDirection,
    /// Whether to truncate if longer than `max_len`.
    pub truncate: bool,
}

/// Direction for padding sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingDirection {
    /// Pad on the right (after sequence).
    Right,
    /// Pad on the left (before sequence).
    Left,
}

impl PaddingPostProcessor {
    /// Create a new padding post-processor.
    #[must_use]
    pub fn new(
        max_len: Option<usize>,
        pad_token_id: u32,
        direction: PaddingDirection,
        truncate: bool,
    ) -> Self {
        Self {
            max_len,
            pad_token_id,
            direction,
            truncate,
        }
    }

    /// Create right-padding post-processor.
    #[must_use]
    pub fn right_padding(max_len: usize, pad_token_id: u32) -> Self {
        Self::new(Some(max_len), pad_token_id, PaddingDirection::Right, true)
    }

    /// Create left-padding post-processor.
    #[must_use]
    pub fn left_padding(max_len: usize, pad_token_id: u32) -> Self {
        Self::new(Some(max_len), pad_token_id, PaddingDirection::Left, true)
    }
}

impl PostProcessor for PaddingPostProcessor {
    fn post_process(&self, mut encoding: Encoding, _vocab: &Vocabulary) -> Result<Encoding> {
        // Truncate if needed
        if self.truncate {
            if let Some(max_len) = self.max_len {
                encoding.truncate(max_len);
            }
        }

        // Pad if needed
        if let Some(max_len) = self.max_len {
            let current_len = encoding.len();
            if current_len < max_len {
                let pad_len = max_len - current_len;

                match self.direction {
                    PaddingDirection::Right => {
                        // Pad on the right
                        encoding
                            .ids
                            .extend(std::iter::repeat(self.pad_token_id).take(pad_len));
                        encoding
                            .tokens
                            .extend(std::iter::repeat("[PAD]".to_string()).take(pad_len));
                        encoding
                            .offsets
                            .extend(std::iter::repeat((0, 0)).take(pad_len));
                        encoding
                            .attention_mask
                            .extend(std::iter::repeat(0u32).take(pad_len));
                        encoding
                            .token_type_ids
                            .extend(std::iter::repeat(0u32).take(pad_len));
                        encoding
                            .special_tokens_mask
                            .extend(std::iter::repeat(1u32).take(pad_len));
                    }
                    PaddingDirection::Left => {
                        // Pad on the left
                        let mut new_ids = std::iter::repeat(self.pad_token_id)
                            .take(pad_len)
                            .collect::<Vec<_>>();
                        new_ids.extend(encoding.ids);
                        encoding.ids = new_ids;

                        let mut new_tokens = std::iter::repeat("[PAD]".to_string())
                            .take(pad_len)
                            .collect::<Vec<_>>();
                        new_tokens.extend(encoding.tokens);
                        encoding.tokens = new_tokens;

                        let mut new_offsets =
                            std::iter::repeat((0, 0)).take(pad_len).collect::<Vec<_>>();
                        new_offsets.extend(encoding.offsets);
                        encoding.offsets = new_offsets;

                        let mut new_attention_mask =
                            std::iter::repeat(0u32).take(pad_len).collect::<Vec<_>>();
                        new_attention_mask.extend(encoding.attention_mask);
                        encoding.attention_mask = new_attention_mask;

                        let mut new_token_type_ids =
                            std::iter::repeat(0u32).take(pad_len).collect::<Vec<_>>();
                        new_token_type_ids.extend(encoding.token_type_ids);
                        encoding.token_type_ids = new_token_type_ids;

                        let mut new_special_tokens_mask =
                            std::iter::repeat(1u32).take(pad_len).collect::<Vec<_>>();
                        new_special_tokens_mask.extend(encoding.special_tokens_mask);
                        encoding.special_tokens_mask = new_special_tokens_mask;
                    }
                }
            }
        }

        Ok(encoding)
    }
}

/// Chain multiple post-processors together.
pub struct ChainedPostProcessor {
    /// The chain of post-processors to apply in order.
    pub processors: Vec<Box<dyn PostProcessor + Send + Sync>>,
}

impl ChainedPostProcessor {
    /// Create a new chained post-processor.
    #[must_use]
    pub fn new(processors: Vec<Box<dyn PostProcessor + Send + Sync>>) -> Self {
        Self { processors }
    }

    /// Add a post-processor to the chain.
    pub fn add_processor(&mut self, processor: Box<dyn PostProcessor + Send + Sync>) {
        self.processors.push(processor);
    }
}

impl PostProcessor for ChainedPostProcessor {
    fn post_process(&self, encoding: Encoding, vocab: &Vocabulary) -> Result<Encoding> {
        let mut result = encoding;
        for processor in &self.processors {
            result = processor.post_process(result, vocab)?;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocabulary::Vocabulary;

    #[test]
    fn test_template_post_processor_bert() {
        let mut vocab = Vocabulary::new();
        vocab.add_special_token("[CLS]".to_string(), 101).unwrap();
        vocab.add_special_token("[SEP]".to_string(), 102).unwrap();

        let processor = TemplatePostProcessor::bert();
        let encoding = Encoding::from_tokens(
            vec![1, 2, 3],
            vec!["hello".to_string(), "world".to_string(), "!".to_string()],
            12,
        );

        let result = processor.post_process(encoding, &vocab).unwrap();

        // Should have [CLS] at start and [SEP] at end
        assert_eq!(result.ids, vec![101, 1, 2, 3, 102]);
        assert_eq!(result.tokens, vec!["[CLS]", "hello", "world", "!", "[SEP]"]);
        assert_eq!(result.special_tokens_mask, vec![1, 0, 0, 0, 1]);
    }

    #[test]
    fn test_padding_post_processor_right() {
        let processor = PaddingPostProcessor::right_padding(5, 0);
        let encoding = Encoding::from_tokens(
            vec![1, 2],
            vec!["hello".to_string(), "world".to_string()],
            10,
        );
        let vocab = Vocabulary::new();

        let result = processor.post_process(encoding, &vocab).unwrap();

        assert_eq!(result.ids, vec![1, 2, 0, 0, 0]);
        assert_eq!(result.attention_mask, vec![1, 1, 0, 0, 0]);
    }

    #[test]
    fn test_padding_post_processor_left() {
        let processor = PaddingPostProcessor::left_padding(5, 0);
        let encoding = Encoding::from_tokens(
            vec![1, 2],
            vec!["hello".to_string(), "world".to_string()],
            10,
        );
        let vocab = Vocabulary::new();

        let result = processor.post_process(encoding, &vocab).unwrap();

        assert_eq!(result.ids, vec![0, 0, 0, 1, 2]);
        assert_eq!(result.attention_mask, vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn test_padding_with_truncation() {
        let processor = PaddingPostProcessor::right_padding(3, 0);
        let encoding = Encoding::from_tokens(
            vec![1, 2, 3, 4],
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string(),
            ],
            10,
        );
        let vocab = Vocabulary::new();

        let result = processor.post_process(encoding, &vocab).unwrap();

        // Should truncate to 3 tokens then pad to 3 (no padding needed)
        assert_eq!(result.ids, vec![1, 2, 3]);
        assert_eq!(result.attention_mask, vec![1, 1, 1]);
    }

    #[test]
    fn test_missing_special_token() {
        let vocab = Vocabulary::new();
        let processor = TemplatePostProcessor::bert();
        let encoding = Encoding::from_tokens(
            vec![1, 2],
            vec!["hello".to_string(), "world".to_string()],
            10,
        );

        assert!(processor.post_process(encoding, &vocab).is_err());
    }
}
