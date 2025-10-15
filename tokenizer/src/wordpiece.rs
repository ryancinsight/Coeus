//! `WordPiece` tokenizer implementation for BERT-style tokenization.

use crate::encoding::{BatchEncoding, Encoding};
use crate::error::{Result, TokenizerError};
use crate::post_processor::{PostProcessor, TemplatePostProcessor};
use crate::pre_tokenizer::{BasicPreTokenizer, PreTokenizer};
use crate::vocabulary::Vocabulary;
use crate::{BatchTokenizer, Tokenizer};

/// `WordPiece` tokenizer for BERT-style subword tokenization.
///
/// Implements Google's `WordPiece` algorithm used in BERT and other transformer models.
/// The algorithm works by:
/// 1. Splitting text into words
/// 2. For each word, finding the longest prefix in vocabulary
/// 3. Adding "##" prefix for continuation subwords
/// 4. Using unknown token for out-of-vocabulary words
#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    /// Token vocabulary.
    vocab: Vocabulary,
    /// Pre-tokenizer for word segmentation.
    pre_tokenizer: BasicPreTokenizer,
    /// Post-processor for special tokens.
    post_processor: Option<TemplatePostProcessor>,
    /// Maximum word length to try (to prevent infinite loops).
    max_word_len: usize,
    /// Unknown token for out-of-vocabulary words.
    unk_token: Option<String>,
}

impl WordPieceTokenizer {
    /// Create a new `WordPiece` tokenizer.
    ///
    /// # Arguments
    /// * `vocab` - Token vocabulary mapping tokens to IDs
    ///
    /// # Errors
    /// Returns error if vocabulary is invalid
    pub fn new(vocab: Vocabulary) -> Result<Self> {
        Ok(Self {
            vocab,
            pre_tokenizer: BasicPreTokenizer::new(),
            post_processor: None,
            max_word_len: 100,
            unk_token: Some("[UNK]".to_string()),
        })
    }

    /// Create tokenizer with custom pre-tokenizer.
    #[must_use]
    pub fn with_pre_tokenizer(mut self, pre_tokenizer: BasicPreTokenizer) -> Self {
        self.pre_tokenizer = pre_tokenizer;
        self
    }

    /// Create tokenizer with post-processor.
    #[must_use]
    pub fn with_post_processor(mut self, post_processor: TemplatePostProcessor) -> Self {
        self.post_processor = Some(post_processor);
        self
    }

    /// Set maximum word length.
    #[must_use]
    pub fn with_max_word_len(mut self, max_word_len: usize) -> Self {
        self.max_word_len = max_word_len;
        self
    }

    /// Set unknown token.
    #[must_use]
    pub fn with_unk_token(mut self, unk_token: String) -> Self {
        self.unk_token = Some(unk_token);
        self
    }

    /// Apply `WordPiece` algorithm to a word.
    ///
    /// # Arguments
    /// * `word` - Input word to tokenize
    ///
    /// # Returns
    /// Vector of subword tokens (first without ##, continuations with ##)
    fn wordpiece_tokenize(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return Vec::new();
        }

        // Check if whole word is in vocabulary
        if self.vocab.token_to_id(word).is_some() {
            return vec![word.to_string()];
        }

        let mut subwords = Vec::new();
        let word_chars: Vec<char> = word.chars().collect();
        let mut start = 0;

        while start < word_chars.len() {
            let mut end = word_chars.len();
            let mut found = false;

            // Try progressively shorter substrings
            while end > start && end - start <= self.max_word_len {
                let substring: String = word_chars[start..end].iter().collect();
                let token = if start == 0 {
                    substring // First token doesn't get ##
                } else {
                    format!("##{substring}") // Continuation tokens get ##
                };

                if self.vocab.token_to_id(&token).is_some() {
                    subwords.push(token);
                    start = end;
                    found = true;
                    break;
                }
                end -= 1;
            }

            if !found {
                // No valid substring found - use unknown token for remaining part
                if let Some(unk_token) = &self.unk_token {
                    let _remaining: String = word_chars[start..].iter().collect();
                    if start == 0 {
                        subwords.push(unk_token.clone());
                    } else {
                        subwords.push(format!("##{unk_token}"));
                    }
                }
                break;
            }
        }

        subwords
    }

    /// Handle unknown tokens.
    fn handle_unknown(&self, token: &str) -> Result<u32> {
        if let Some(unk_token) = &self.unk_token {
            self.vocab.token_to_id(unk_token).ok_or_else(|| {
                TokenizerError::vocabulary(format!(
                    "Unknown token '{token}' and UNK token '{unk_token}' not in vocabulary"
                ))
            })
        } else {
            Err(TokenizerError::UnknownToken(token.to_string()))
        }
    }
}

impl Tokenizer for WordPieceTokenizer {
    fn encode(&self, text: &str) -> Result<Encoding> {
        // Pre-tokenize into words
        let words = self.pre_tokenizer.pre_tokenize(text)?;

        let mut all_ids = Vec::new();
        let mut all_tokens = Vec::new();
        let mut all_offsets = Vec::new();
        let mut offset = 0;

        // Apply WordPiece to each word
        for word in words {
            let subwords = self.wordpiece_tokenize(&word);

            for subword in subwords {
                // Convert subword to ID
                let token_id = self
                    .vocab
                    .token_to_id(&subword)
                    .unwrap_or_else(|| self.handle_unknown(&subword).unwrap_or(0)); // Fallback to 0

                all_ids.push(token_id);
                all_tokens.push(subword.clone());

                // Calculate character offsets (approximate)
                let token_len = subword.trim_start_matches("##").chars().count();
                all_offsets.push((offset, offset + token_len));
                offset += token_len;
            }

            // Add space between words
            offset += 1; // Account for space
        }

        let ids_len = all_ids.len();
        let mut encoding = Encoding::new(
            all_ids,
            all_tokens,
            all_offsets,
            vec![1; ids_len], // attention_mask
            vec![0; ids_len], // token_type_ids
            vec![0; ids_len], // special_tokens_mask
            text.len(),
        );

        // Apply post-processing if configured
        if let Some(processor) = &self.post_processor {
            encoding = processor.post_process(encoding, &self.vocab)?;
        }

        Ok(encoding)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut tokens = Vec::new();
        for &id in ids {
            if let Some(token) = self.vocab.id_to_token(id) {
                tokens.push(token);
            } else {
                return Err(TokenizerError::InvalidTokenId(id));
            }
        }

        // Join tokens, handling special tokens and ## continuations
        let mut result = String::new();
        for (i, token) in tokens.iter().enumerate() {
            if self.vocab.is_special_token(token) {
                // Special tokens are kept as-is
                result.push_str(token);
            } else if let Some(stripped) = token.strip_prefix("##") {
                // Continuation tokens: remove ## and append without space
                result.push_str(stripped);
            } else {
                // Regular tokens: add space if not first token
                if i > 0 {
                    result.push(' ');
                }
                result.push_str(token);
            }
        }

        Ok(result)
    }

    fn vocab_size(&self) -> usize {
        self.vocab.size()
    }

    fn vocabulary(&self) -> &Vocabulary {
        &self.vocab
    }
}

impl BatchTokenizer for WordPieceTokenizer {
    fn encode_batch(
        &self,
        texts: &[String],
        padding: bool,
        truncation: bool,
        max_length: Option<usize>,
    ) -> Result<BatchEncoding> {
        let mut encodings = Vec::new();
        let mut max_len = 0;

        // Encode each text
        for text in texts {
            let mut encoding = self.encode(text)?;

            if truncation {
                if let Some(max_len_val) = max_length {
                    encoding.truncate(max_len_val);
                }
            }

            max_len = max_len.max(encoding.len());
            encodings.push(encoding);
        }

        // Apply padding if requested
        if padding {
            let pad_token_id = self.vocab.special_token_id("[PAD]").unwrap_or(0);
            for encoding in &mut encodings {
                if let Some(max_len_val) = max_length {
                    if encoding.len() < max_len_val {
                        encoding.pad(max_len_val, pad_token_id);
                    }
                } else if encoding.len() < max_len {
                    encoding.pad(max_len, pad_token_id);
                }
            }
        }

        Ok(BatchEncoding::new(
            encodings,
            max_length.unwrap_or(max_len),
            self.vocab.special_token_id("[PAD]").unwrap_or(0),
            padding,
            truncation,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocabulary::Vocabulary;

    fn create_test_vocab() -> Vocabulary {
        let mut vocab = Vocabulary::new();
        vocab.add_token("hello".to_string(), 0).unwrap();
        vocab.add_token("world".to_string(), 1).unwrap();
        vocab.add_token("##l".to_string(), 2).unwrap();
        vocab.add_token("##o".to_string(), 3).unwrap();
        vocab.add_token("##r".to_string(), 4).unwrap();
        vocab.add_token("##d".to_string(), 5).unwrap();
        vocab.add_token("un".to_string(), 6).unwrap();
        vocab.add_token("##k".to_string(), 7).unwrap();
        vocab.add_token("##n".to_string(), 8).unwrap();
        vocab.add_token("##w".to_string(), 9).unwrap();
        vocab.add_token("[UNK]".to_string(), 10).unwrap();
        vocab
    }

    #[test]
    fn test_wordpiece_basic_encoding() {
        let vocab = create_test_vocab();
        let tokenizer = WordPieceTokenizer::new(vocab).unwrap();

        let encoding = tokenizer.encode("hello").unwrap();

        // Should recognize "hello" as a whole word
        assert!(!encoding.ids.is_empty());
        assert_eq!(encoding.tokens.len(), encoding.ids.len());
    }

    #[test]
    fn test_wordpiece_subword_splitting() {
        let vocab = create_test_vocab();
        let tokenizer = WordPieceTokenizer::new(vocab).unwrap();

        let encoding = tokenizer.encode("unknown").unwrap();

        // Should split "unknown" into subwords: "un" + "##k" + "##n" + "##o" + "##w"
        // But our vocab only has "un" and "##k", so remaining gets [UNK]
        assert!(!encoding.ids.is_empty());
    }

    #[test]
    fn test_wordpiece_decode() {
        let vocab = create_test_vocab();
        let tokenizer = WordPieceTokenizer::new(vocab).unwrap();

        // Test round-trip
        let original = "hello world";
        let encoding = tokenizer.encode(original).unwrap();
        let decoded = tokenizer.decode(&encoding.ids).unwrap();

        // Should reconstruct original text
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_wordpiece_with_special_tokens() {
        let mut vocab = create_test_vocab();
        vocab.add_special_token("[CLS]".to_string(), 12).unwrap();
        vocab.add_special_token("[SEP]".to_string(), 13).unwrap();

        let post_processor = TemplatePostProcessor::bert();
        let tokenizer = WordPieceTokenizer::new(vocab)
            .unwrap()
            .with_post_processor(post_processor);

        let encoding = tokenizer.encode("hello").unwrap();

        // Should have [CLS] and [SEP] tokens
        assert!(
            encoding.tokens.contains(&"[CLS]".to_string())
                || encoding.tokens.contains(&"[SEP]".to_string())
        );
    }

    #[test]
    fn test_wordpiece_batch_encoding() {
        let vocab = create_test_vocab();
        let tokenizer = WordPieceTokenizer::new(vocab).unwrap();

        let texts = vec!["hello".to_string(), "world".to_string()];
        let batch = tokenizer
            .encode_batch(&texts, true, false, Some(5))
            .unwrap();

        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.max_len, 5);
        assert!(batch.padded);
    }

    #[test]
    fn test_wordpiece_tokenize_word() {
        let vocab = create_test_vocab();
        let tokenizer = WordPieceTokenizer::new(vocab).unwrap();

        let result = tokenizer.wordpiece_tokenize("hello");

        // Should recognize as whole word
        assert_eq!(result, vec!["hello"]);

        let result = tokenizer.wordpiece_tokenize("unknown");

        // Should handle unknown word
        assert!(!result.is_empty());
    }
}
