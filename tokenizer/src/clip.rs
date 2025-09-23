//! CLIP tokenizer implementation with BPE algorithm and pre-trained vocabulary support

use crate::bpe::BpeTokenizer;
use crate::downloader::VocabDownloader;
use crate::error::{Result, TokenizerError};
use crate::tokenizer::{DecodeOptions, TokenizationResult, TokenizeOptions, Tokenizer};
use crate::vocabulary::Vocabulary;

/// CLIP tokenizer with BPE algorithm and pre-trained vocabulary
pub struct CLIPTokenizer {
    /// Underlying BPE tokenizer
    bpe_tokenizer: BpeTokenizer,
}

impl CLIPTokenizer {
    /// Create a new CLIP tokenizer with downloaded vocabulary
    ///
    /// # Errors
    /// Returns `TokenizerError::IoError` if vocabulary download fails
    /// Returns `TokenizerError::VocabularyError` if vocabulary parsing fails
    pub fn new() -> Result<Self> {
        let downloader = VocabDownloader::new();

        // Download CLIP vocabulary and merges
        let vocab_data = downloader.download_clip_vocab()?;

        // Create vocabulary from downloaded data
        let vocab = downloader.create_vocab_from_clip_data(&vocab_data)?;

        // Create BPE tokenizer with CLIP merges
        let mut bpe_tokenizer = BpeTokenizer::new("clip".to_string());
        bpe_tokenizer.load_vocab_and_merges(vocab.token_to_id_map().clone(), vocab_data.merges)?;

        // Add special tokens
        bpe_tokenizer
            .add_special_tokens(&["<|startoftext|>".to_string(), "<|endoftext|>".to_string()]);

        Ok(Self { bpe_tokenizer })
    }

    /// Create CLIP tokenizer with custom vocabulary
    ///
    /// # Errors
    /// Returns `TokenizerError` if vocabulary setup fails
    pub fn with_vocabulary(vocab: Vocabulary) -> Result<Self> {
        let mut bpe_tokenizer = BpeTokenizer::new("clip".to_string());
        *bpe_tokenizer.vocabulary_mut() = vocab;

        // Add special tokens
        bpe_tokenizer
            .add_special_tokens(&["<|startoftext|>".to_string(), "<|endoftext|>".to_string()]);

        Ok(Self { bpe_tokenizer })
    }

    /// Get the underlying BPE tokenizer
    #[must_use]
    pub const fn bpe_tokenizer(&self) -> &BpeTokenizer {
        &self.bpe_tokenizer
    }

    /// Get vocabulary size
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.bpe_tokenizer.vocabulary().size()
    }

    /// Get model name
    #[must_use]
    pub const fn name(&self) -> &'static str {
        "clip"
    }

    /// Get BOS token ID (CLIP uses startoftext as BOS)
    #[must_use]
    pub fn bos_token_id(&self) -> Option<usize> {
        self.bpe_tokenizer
            .vocabulary()
            .get_special_token_id("<|startoftext|>")
    }

    /// Get EOS token ID (CLIP uses endoftext as EOS)
    #[must_use]
    pub fn eos_token_id(&self) -> Option<usize> {
        self.bpe_tokenizer
            .vocabulary()
            .get_special_token_id("<|endoftext|>")
    }

    /// Get PAD token ID (CLIP doesn't have padding token)
    #[must_use]
    pub const fn pad_token_id(&self) -> Option<usize> {
        None
    }

    /// Get UNK token ID (CLIP doesn't have unknown token)
    #[must_use]
    pub const fn unk_token_id(&self) -> Option<usize> {
        None
    }

    /// Get CLS token ID (CLIP doesn't have CLS token)
    #[must_use]
    pub const fn cls_token_id(&self) -> Option<usize> {
        None
    }

    /// Get SEP token ID (CLIP doesn't have SEP token)
    #[must_use]
    pub const fn sep_token_id(&self) -> Option<usize> {
        None
    }

    /// Get MASK token ID (CLIP doesn't have MASK token)
    #[must_use]
    pub const fn mask_token_id(&self) -> Option<usize> {
        None
    }
}

impl Tokenizer for CLIPTokenizer {
    fn name(&self) -> &str {
        self.name()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<usize>> {
        let token_strings = self.bpe_tokenizer.encode_bpe(text)?;
        let mut tokens = Vec::new();

        // Convert token strings to IDs
        for token in token_strings {
            if let Some(id) = self.bpe_tokenizer.vocabulary().get_token_id(&token) {
                tokens.push(id);
            } else {
                return Err(TokenizerError::unknown_token(token));
            }
        }

        if add_special_tokens {
            let mut result = Vec::new();
            // Add BOS
            if let Some(bos_id) = self.bos_token_id() {
                result.push(bos_id);
            }
            result.extend(tokens);
            // Add EOS
            if let Some(eos_id) = self.eos_token_id() {
                result.push(eos_id);
            }
            Ok(result)
        } else {
            Ok(tokens)
        }
    }

    fn encode_with_options(&self, text: &str, options: &TokenizeOptions) -> Result<Vec<usize>> {
        self.encode(text, options.add_special_tokens)
    }

    fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Result<Vec<Vec<usize>>> {
        texts
            .iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }

    fn encode_batch_with_options(
        &self,
        texts: &[&str],
        options: &TokenizeOptions,
    ) -> Result<Vec<Vec<usize>>> {
        texts
            .iter()
            .map(|text| self.encode_with_options(text, options))
            .collect()
    }

    fn decode(&self, token_ids: &[usize], skip_special_tokens: bool) -> Result<String> {
        let tokens = Self::ids_to_tokens(token_ids, self.bpe_tokenizer.vocabulary());

        if skip_special_tokens {
            // Filter out special tokens
            let filtered_tokens: Vec<String> = tokens
                .into_iter()
                .filter(|token| !self.bpe_tokenizer.vocabulary().is_special_token(token))
                .collect();
            self.bpe_tokenizer.decode_bpe(&filtered_tokens)
        } else {
            self.bpe_tokenizer.decode_bpe(&tokens)
        }
    }

    fn decode_with_options(&self, token_ids: &[usize], options: &DecodeOptions) -> Result<String> {
        self.decode(token_ids, options.skip_special_tokens)
    }

    fn decode_batch(
        &self,
        batch_token_ids: &[&[usize]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        batch_token_ids
            .iter()
            .map(|token_ids| self.decode(token_ids, skip_special_tokens))
            .collect()
    }

    fn decode_batch_with_options(
        &self,
        batch_token_ids: &[&[usize]],
        options: &DecodeOptions,
    ) -> Result<Vec<String>> {
        batch_token_ids
            .iter()
            .map(|token_ids| self.decode_with_options(token_ids, options))
            .collect()
    }

    fn convert_ids_to_tokens(&self, token_ids: &[usize]) -> Result<Vec<String>> {
        token_ids
            .iter()
            .map(|&id| {
                self.bpe_tokenizer
                    .vocabulary()
                    .get_token(id)
                    .ok_or_else(|| TokenizerError::invalid_token_id(id))
                    .map(str::to_string)
            })
            .collect()
    }

    fn convert_tokens_to_ids(&self, tokens: &[&str]) -> Result<Vec<usize>> {
        tokens
            .iter()
            .map(|token| {
                self.bpe_tokenizer
                    .vocabulary()
                    .get_token_id(token)
                    .ok_or_else(|| TokenizerError::unknown_token((*token).to_string()))
            })
            .collect()
    }

    fn special_tokens(&self) -> &std::collections::HashMap<String, usize> {
        self.bpe_tokenizer.vocabulary().special_tokens()
    }

    fn is_special_token(&self, token: &str) -> bool {
        self.bpe_tokenizer.vocabulary().is_special_token(token)
    }

    fn bos_token_id(&self) -> Option<usize> {
        self.bos_token_id()
    }

    fn eos_token_id(&self) -> Option<usize> {
        self.eos_token_id()
    }

    fn pad_token_id(&self) -> Option<usize> {
        self.pad_token_id()
    }

    fn unk_token_id(&self) -> Option<usize> {
        self.unk_token_id()
    }

    fn cls_token_id(&self) -> Option<usize> {
        self.cls_token_id()
    }

    fn sep_token_id(&self) -> Option<usize> {
        self.sep_token_id()
    }

    fn mask_token_id(&self) -> Option<usize> {
        self.mask_token_id()
    }

    fn vocabulary(&self) -> &Vocabulary {
        self.bpe_tokenizer.vocabulary()
    }

    fn vocabulary_mut(&mut self) -> &mut Vocabulary {
        self.bpe_tokenizer.vocabulary_mut()
    }

    fn encode_to_result(
        &self,
        text: &str,
        options: &TokenizeOptions,
    ) -> Result<TokenizationResult> {
        let token_ids = self.encode_with_options(text, options)?;

        let attention_mask = vec![1; token_ids.len()];
        let special_tokens_mask = Some(
            token_ids
                .iter()
                .map(|&id| {
                    self.vocabulary()
                        .get_token(id)
                        .map_or(0, |token| usize::from(self.is_special_token(token)))
                })
                .collect(),
        );

        Ok(TokenizationResult {
            token_ids,
            attention_mask,
            token_type_ids: None,
            offsets: None,
            special_tokens_mask,
        })
    }

    fn validate(&self) -> Result<()> {
        self.bpe_tokenizer.vocabulary().validate()
    }
}

impl CLIPTokenizer {
    /// Helper function to convert token IDs to token strings
    fn ids_to_tokens(ids: &[usize], vocab: &Vocabulary) -> Vec<String> {
        ids.iter()
            .filter_map(|&id| vocab.get_token(id).map(str::to_string))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clip_tokenizer_creation() -> Result<()> {
        // Skip test if network is unavailable
        let result = CLIPTokenizer::new();
        match result {
            Ok(tokenizer) => {
                assert_eq!(tokenizer.name(), "clip");
                assert!(tokenizer.vocab_size() > 0);
                assert!(tokenizer.bos_token_id().is_some());
                assert!(tokenizer.eos_token_id().is_some());
                Ok(())
            }
            Err(TokenizerError::IoError { .. }) => {
                // Expected when no internet connection
                println!("Skipping CLIP test due to network unavailability");
                Ok(())
            }
            Err(e) => Err(TokenizerError::vocabulary_error(format!(
                "Failed to load CLIP vocabulary: {e}"
            ))),
        }
    }

    #[test]
    fn test_clip_special_tokens() {
        // Test with mock vocabulary since we can't download in tests
        let mut vocab = Vocabulary::new();
        vocab.add_special_token("<|startoftext|>".to_string());
        vocab.add_special_token("<|endoftext|>".to_string());

        let tokenizer = CLIPTokenizer::with_vocabulary(vocab).unwrap();
        assert!(tokenizer.bos_token_id().is_some());
        assert!(tokenizer.eos_token_id().is_some());
        assert!(tokenizer.pad_token_id().is_none());
    }
}
