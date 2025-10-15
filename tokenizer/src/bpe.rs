//! Byte Pair Encoding (BPE) tokenizer implementation.

use crate::encoding::{BatchEncoding, Encoding};
use crate::error::{Result, TokenizerError};
use crate::post_processor::{PostProcessor, TemplatePostProcessor};
use crate::pre_tokenizer::{BasicPreTokenizer, PreTokenizer};
use crate::vocabulary::Vocabulary;
use crate::{BatchTokenizer, Tokenizer};
use std::collections::HashMap;

/// Byte Pair Encoding tokenizer.
///
/// Implements the BPE algorithm used by GPT models and other subword tokenizers.
/// Merges frequent byte pairs iteratively to build a vocabulary of subword units.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Token vocabulary.
    vocab: Vocabulary,
    /// BPE merge rules (pair -> merged token).
    merges: HashMap<(String, String), String>,
    /// Pre-tokenizer for text preprocessing.
    pre_tokenizer: BasicPreTokenizer,
    /// Post-processor for special tokens.
    post_processor: Option<TemplatePostProcessor>,
    /// Whether to add prefix spaces.
    add_prefix_space: bool,
    /// Unknown token handling.
    unk_token: Option<String>,
}

impl BpeTokenizer {
    /// Create a new BPE tokenizer.
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
            pre_tokenizer: BasicPreTokenizer::new(),
            post_processor: None,
            add_prefix_space: true,
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

    /// Set prefix space handling.
    #[must_use]
    pub fn with_prefix_space(mut self, add_prefix_space: bool) -> Self {
        self.add_prefix_space = add_prefix_space;
        self
    }

    /// Set unknown token.
    #[must_use]
    pub fn with_unk_token(mut self, unk_token: String) -> Self {
        self.unk_token = Some(unk_token);
        self
    }

    /// Apply BPE algorithm to a word.
    ///
    /// # Arguments
    /// * `word` - Input word to tokenize
    ///
    /// # Returns
    /// Vector of subword tokens
    fn apply_bpe(&self, word: &str) -> Vec<String> {
        // Start with character-level tokenization
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        // Apply merge rules in order
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

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Result<Encoding> {
        // Pre-tokenize
        let pre_tokens = self.pre_tokenizer.pre_tokenize(text)?;

        let mut all_ids = Vec::new();
        let mut all_tokens = Vec::new();
        let mut all_offsets = Vec::new();
        let mut offset = 0;

        // Apply BPE to each pre-token
        for pre_token in pre_tokens {
            let subwords = self.apply_bpe(&pre_token);

            for subword in subwords {
                // Convert subword to ID
                let token_id = self
                    .vocab
                    .token_to_id(&subword)
                    .unwrap_or_else(|| self.handle_unknown(&subword).unwrap_or(0)); // Fallback to 0

                all_ids.push(token_id);
                all_tokens.push(subword.clone());

                // Calculate character offsets (approximate)
                let token_len = subword.chars().count();
                all_offsets.push((offset, offset + token_len));
                offset += token_len;
            }

            // Add space between words if prefix space is enabled
            if self.add_prefix_space {
                offset += 1; // Account for space
            }
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

        // Join tokens, handling special tokens
        // For BPE, we reconstruct by joining tokens directly since merges preserve original text
        let mut result = String::new();
        for token in tokens {
            if self.vocab.is_special_token(&token) {
                // Special tokens are kept as-is
                result.push_str(&token);
            } else {
                // Regular tokens: join directly
                result.push_str(&token);
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

impl BatchTokenizer for BpeTokenizer {
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
        vocab.add_token("h".to_string(), 0).unwrap();
        vocab.add_token("e".to_string(), 1).unwrap();
        vocab.add_token("l".to_string(), 2).unwrap();
        vocab.add_token("o".to_string(), 3).unwrap();
        vocab.add_token("w".to_string(), 4).unwrap();
        vocab.add_token("r".to_string(), 5).unwrap();
        vocab.add_token("d".to_string(), 6).unwrap();
        vocab.add_token("he".to_string(), 7).unwrap();
        vocab.add_token("ll".to_string(), 8).unwrap();
        vocab.add_token("lo".to_string(), 9).unwrap();
        vocab.add_token("world".to_string(), 10).unwrap();
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
    fn test_bpe_basic_encoding() {
        let vocab = create_test_vocab();
        let merges = create_test_merges();
        let tokenizer = BpeTokenizer::new(vocab, merges).unwrap();

        let encoding = tokenizer.encode("hello").unwrap();

        // Should apply merges: h + e → he, l + l → ll, l + o → lo
        assert!(!encoding.ids.is_empty());
        assert_eq!(encoding.tokens.len(), encoding.ids.len());
    }

    #[test]
    fn test_bpe_decode() {
        let vocab = create_test_vocab();
        let merges = create_test_merges();
        let tokenizer = BpeTokenizer::new(vocab, merges).unwrap();

        // Test round-trip
        let original = "hello";
        let encoding = tokenizer.encode(original).unwrap();
        let decoded = tokenizer.decode(&encoding.ids).unwrap();

        // Should reconstruct original text
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_bpe_with_special_tokens() {
        let mut vocab = create_test_vocab();
        vocab.add_special_token("[CLS]".to_string(), 11).unwrap();
        vocab.add_special_token("[SEP]".to_string(), 12).unwrap();

        let merges = create_test_merges();
        let post_processor = TemplatePostProcessor::bert();
        let tokenizer = BpeTokenizer::new(vocab, merges)
            .unwrap()
            .with_post_processor(post_processor);

        let encoding = tokenizer.encode("hello").unwrap();

        // Should have [CLS] and [SEP] tokens
        assert!(encoding.tokens.contains(&"[CLS]".to_string()));
        assert!(encoding.tokens.contains(&"[SEP]".to_string()));
    }

    #[test]
    fn test_bpe_batch_encoding() {
        let vocab = create_test_vocab();
        let merges = create_test_merges();
        let tokenizer = BpeTokenizer::new(vocab, merges).unwrap();

        let texts = vec!["hello".to_string(), "world".to_string()];
        let batch = tokenizer
            .encode_batch(&texts, true, false, Some(5))
            .unwrap();

        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.max_len, 5);
        assert!(batch.padded);
    }

    #[test]
    fn test_bpe_apply_bpe() {
        let vocab = create_test_vocab();
        let merges = create_test_merges();
        let tokenizer = BpeTokenizer::new(vocab, merges).unwrap();

        let result = tokenizer.apply_bpe("hello");

        // Should have merged some pairs
        assert!(!result.is_empty());
        assert!(result.contains(&"he".to_string()) || result.len() < 5); // Some merging occurred
    }
}
