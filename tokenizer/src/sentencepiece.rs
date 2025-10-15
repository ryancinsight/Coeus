//! `SentencePiece` tokenizer implementation for unsupervised tokenization.

use crate::encoding::{BatchEncoding, Encoding};
use crate::error::{Result, TokenizerError};
use crate::post_processor::{PostProcessor, TemplatePostProcessor};
use crate::vocabulary::Vocabulary;
use crate::{BatchTokenizer, Tokenizer};
use std::collections::HashMap;

/// `SentencePiece` tokenizer for unsupervised subword tokenization.
///
/// Implements Google's `SentencePiece` algorithm used in multilingual models like `XLNet` and ALBERT.
/// Unlike BPE, `SentencePiece` treats text as a sequence of UTF-8 bytes and learns
/// subword units directly from raw text without explicit pre-tokenization.
#[derive(Debug, Clone)]
pub struct SentencePieceTokenizer {
    /// Token vocabulary.
    vocab: Vocabulary,
    /// BPE merge rules for `SentencePiece` (byte-level).
    merges: HashMap<(String, String), String>,
    /// Post-processor for special tokens.
    post_processor: Option<TemplatePostProcessor>,
    /// Whether to encode text as bytes first.
    byte_fallback: bool,
    /// Unknown token for out-of-vocabulary sequences.
    unk_token: Option<String>,
}

impl SentencePieceTokenizer {
    /// Create a new `SentencePiece` tokenizer.
    ///
    /// # Arguments
    /// * `vocab` - Token vocabulary mapping tokens to IDs
    /// * `merges` - BPE merge rules as (pair1, pair2) -> `merged_token`
    ///
    /// # Errors
    /// Returns error if vocabulary is invalid
    pub fn new(vocab: Vocabulary, merges: Vec<(String, String)>) -> Result<Self> {
        let mut merge_map = HashMap::new();
        for (pair1, pair2) in merges {
            let merged_token = format!("{pair1}{pair2}");
            merge_map.insert((pair1, pair2), merged_token);
        }

        Ok(Self {
            vocab,
            merges: merge_map,
            post_processor: None,
            byte_fallback: true,
            unk_token: Some("<unk>".to_string()),
        })
    }

    /// Create tokenizer with post-processor.
    #[must_use]
    pub fn with_post_processor(mut self, post_processor: TemplatePostProcessor) -> Self {
        self.post_processor = Some(post_processor);
        self
    }

    /// Set byte fallback behavior.
    #[must_use]
    pub fn with_byte_fallback(mut self, byte_fallback: bool) -> Self {
        self.byte_fallback = byte_fallback;
        self
    }

    /// Set unknown token.
    #[must_use]
    pub fn with_unk_token(mut self, unk_token: String) -> Self {
        self.unk_token = Some(unk_token);
        self
    }

    /// Encode text to UTF-8 bytes and split into characters.
    fn text_to_chars(text: &str) -> Vec<String> {
        text.chars().map(|c| c.to_string()).collect()
    }

    /// Apply `SentencePiece` BPE algorithm to character sequence.
    ///
    /// # Arguments
    /// * `chars` - Sequence of character strings to tokenize
    ///
    /// # Returns
    /// Vector of subword tokens
    fn apply_sentencepiece_bpe(&self, chars: &[String]) -> Vec<String> {
        if chars.is_empty() {
            return Vec::new();
        }

        // Start with character-level tokenization
        let mut tokens: Vec<String> = chars.to_vec();

        // Apply merge rules in order (SentencePiece uses similar BPE logic)
        for ((pair1, pair2), merged) in &self.merges {
            let mut i = 0;
            while i < tokens.len() - 1 {
                if tokens[i] == *pair1 && tokens[i + 1] == *pair2 {
                    // Merge the pair
                    tokens[i].clone_from(merged);
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        tokens
    }

    /// Handle unknown token sequences.
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

    /// Try to encode unknown character sequences as bytes.
    fn encode_as_bytes(&self, chars: &[String]) -> Vec<String> {
        if !self.byte_fallback {
            return chars.to_vec();
        }

        // Convert unknown characters to byte sequences
        let mut result = Vec::new();
        for char_str in chars {
            if self.vocab.token_to_id(char_str).is_some() {
                result.push(char_str.clone());
            } else {
                // Convert to UTF-8 bytes
                let bytes = char_str.as_bytes();
                for &byte in bytes {
                    result.push(format!("<0x{byte:02X}>"));
                }
            }
        }
        result
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str) -> Result<Encoding> {
        // Convert text to characters
        let chars = Self::text_to_chars(text);

        // Encode unknown characters as bytes if needed
        let processed_chars = self.encode_as_bytes(&chars);

        // Apply SentencePiece BPE
        let subwords = self.apply_sentencepiece_bpe(&processed_chars);

        let mut all_ids = Vec::new();
        let mut all_tokens = Vec::new();
        let mut all_offsets = Vec::new();
        let mut offset = 0;

        // Convert tokens to IDs
        for subword in subwords {
            let token_id = self
                .vocab
                .token_to_id(&subword)
                .unwrap_or_else(|| self.handle_unknown(&subword).unwrap_or(0)); // Fallback to 0

            all_ids.push(token_id);
            all_tokens.push(subword.clone());

            // Calculate character offsets (approximate for SentencePiece)
            let token_len = subword.chars().count();
            all_offsets.push((offset, offset + token_len));
            offset += token_len;
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

        // Join tokens, handling special tokens and byte sequences
        let mut result = String::new();
        for token in &tokens {
            if self.vocab.is_special_token(token) {
                // Special tokens are kept as-is
                result.push_str(token);
            } else if token.starts_with("<0x") && token.ends_with('>') {
                // Byte sequence: convert back to bytes
                if let Some(hex_str) = token.strip_prefix("<0x").and_then(|s| s.strip_suffix('>')) {
                    if let Ok(byte) = u8::from_str_radix(hex_str, 16) {
                        if let Some(ch) = char::from_u32(u32::from(byte)) {
                            result.push(ch);
                        }
                    }
                }
            } else {
                // Regular tokens: join without spaces (SentencePiece style)
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

impl BatchTokenizer for SentencePieceTokenizer {
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
        vocab.add_token("▁".to_string(), 0).unwrap(); // SentencePiece uses ▁ for word boundaries
        vocab.add_token("h".to_string(), 1).unwrap();
        vocab.add_token("e".to_string(), 2).unwrap();
        vocab.add_token("l".to_string(), 3).unwrap();
        vocab.add_token("o".to_string(), 4).unwrap();
        vocab.add_token("w".to_string(), 5).unwrap();
        vocab.add_token("r".to_string(), 6).unwrap();
        vocab.add_token("d".to_string(), 7).unwrap();
        vocab.add_token("he".to_string(), 8).unwrap();
        vocab.add_token("ll".to_string(), 9).unwrap();
        vocab.add_token("lo".to_string(), 10).unwrap();
        vocab.add_token("<unk>".to_string(), 11).unwrap();
        vocab
    }

    fn create_test_merges() -> Vec<(String, String)> {
        vec![
            ("h".to_string(), "e".to_string()),    // he
            ("l".to_string(), "l".to_string()),    // ll
            ("l".to_string(), "o".to_string()),    // lo
            ("w".to_string(), "o".to_string()),    // wo
            ("wo".to_string(), "r".to_string()),   // wor
            ("wor".to_string(), "l".to_string()),  // worl
            ("worl".to_string(), "d".to_string()), // world
        ]
    }

    #[test]
    fn test_sentencepiece_basic_encoding() {
        let vocab = create_test_vocab();
        let merges = create_test_merges();
        let tokenizer = SentencePieceTokenizer::new(vocab, merges).unwrap();

        let encoding = tokenizer.encode("hello").unwrap();

        // Should tokenize into characters and apply merges
        assert!(!encoding.ids.is_empty());
        assert_eq!(encoding.tokens.len(), encoding.ids.len());
    }

    #[test]
    fn test_sentencepiece_decode() {
        let vocab = create_test_vocab();
        let merges = create_test_merges();
        let tokenizer = SentencePieceTokenizer::new(vocab, merges).unwrap();

        // Test round-trip
        let original = "hello";
        let encoding = tokenizer.encode(original).unwrap();
        let decoded = tokenizer.decode(&encoding.ids).unwrap();

        // Should reconstruct original text
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_sentencepiece_with_special_tokens() {
        let mut vocab = create_test_vocab();
        vocab.add_special_token("[CLS]".to_string(), 12).unwrap();
        vocab.add_special_token("[SEP]".to_string(), 13).unwrap();

        let merges = create_test_merges();
        let post_processor = TemplatePostProcessor::bert();
        let tokenizer = SentencePieceTokenizer::new(vocab, merges)
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
    fn test_sentencepiece_batch_encoding() {
        let vocab = create_test_vocab();
        let merges = create_test_merges();
        let tokenizer = SentencePieceTokenizer::new(vocab, merges).unwrap();

        let texts = vec!["hello".to_string(), "world".to_string()];
        let batch = tokenizer
            .encode_batch(&texts, true, false, Some(5))
            .unwrap();

        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.max_len, 5);
        assert!(batch.padded);
    }

    #[test]
    fn test_sentencepiece_text_to_chars() {
        let vocab = create_test_vocab();
        let merges = create_test_merges();
        let _tokenizer = SentencePieceTokenizer::new(vocab, merges).unwrap();

        let chars = SentencePieceTokenizer::text_to_chars("hello");

        // Should split into individual characters
        assert_eq!(chars, vec!["h", "e", "l", "l", "o"]);
    }

    #[test]
    fn test_sentencepiece_byte_fallback() {
        let vocab = create_test_vocab();
        let merges = create_test_merges();
        let tokenizer = SentencePieceTokenizer::new(vocab, merges)
            .unwrap()
            .with_byte_fallback(true);

        let chars = vec!["❌".to_string()]; // Unknown character
        let encoded = tokenizer.encode_as_bytes(&chars);

        // Should encode as byte sequences
        assert!(!encoded.is_empty());
        // Check that byte encoding worked
        assert!(encoded.iter().any(|s| s.starts_with("<0x")));
    }
}
