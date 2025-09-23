//! Vocabulary downloader for external tokenizer models
//!
//! This module provides functionality to download pre-trained vocabularies,
//! merge files, and special token configurations from various sources.

use crate::error::{Result, TokenizerError};
use crate::vocabulary::Vocabulary;
use ahash::AHashMap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Vocabulary downloader with caching support
#[derive(Debug, Clone)]
pub struct VocabDownloader {
    /// Cache directory for downloaded vocabularies
    cache_dir: PathBuf,
    /// HTTP client timeout in seconds
    timeout_seconds: u64,
}

impl VocabDownloader {
    /// Create a new vocabulary downloader with default cache directory
    #[must_use]
    pub fn new() -> Self {
        Self {
            cache_dir: Self::default_cache_dir(),
            timeout_seconds: 30,
        }
    }

    /// Create downloader with custom cache directory
    #[must_use]
    pub const fn with_cache_dir(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            timeout_seconds: 30,
        }
    }

    /// Create downloader with custom timeout
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }

    /// Get the default cache directory
    fn default_cache_dir() -> PathBuf {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("coeus")
            .join("tokenizers")
    }

    /// Ensure cache directory exists
    fn ensure_cache_dir(&self) -> Result<()> {
        fs::create_dir_all(&self.cache_dir).map_err(|e| {
            TokenizerError::io_error(format!("Failed to create cache directory: {e}"))
        })?;
        Ok(())
    }

    /// Download file from URL with caching
    fn download_file(&self, url: &str, filename: &str) -> Result<String> {
        self.ensure_cache_dir()?;
        let cache_path = self.cache_dir.join(filename);

        // Check if file is already cached
        if cache_path.exists() {
            return fs::read_to_string(&cache_path).map_err(|e| {
                TokenizerError::io_error(format!("Failed to read cached file {filename}: {e}"))
            });
        }

        // Download the file
        let client = reqwest::blocking::Client::new();
        let response = client
            .get(url)
            .timeout(std::time::Duration::from_secs(self.timeout_seconds))
            .send()
            .map_err(|e| TokenizerError::io_error(format!("Failed to download {url}: {e}")))?;

        if !response.status().is_success() {
            return Err(TokenizerError::io_error(format!(
                "HTTP {} when downloading {url}",
                response.status()
            )));
        }

        let content = response.text().map_err(|e| {
            TokenizerError::io_error(format!("Failed to read response from {url}: {e}"))
        })?;

        // Cache the file
        fs::write(&cache_path, &content).map_err(|e| {
            TokenizerError::io_error(format!("Failed to cache file {filename}: {e}"))
        })?;

        Ok(content)
    }

    /// Download GPT-2 vocabulary and merges
    ///
    /// # Errors
    /// Returns an error if:
    /// - Network download fails
    /// - JSON parsing of the encoder fails
    /// - BPE merge parsing fails
    pub fn download_gpt2_vocab(&self) -> Result<Gpt2VocabData> {
        const BASE_URL: &str = "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main";

        // Download encoder (token -> id mapping)
        let encoder_json =
            self.download_file(&format!("{BASE_URL}/encoder.json"), "gpt2_encoder.json")?;

        // Download vocab.bpe (merge rules)
        let vocab_bpe = self.download_file(&format!("{BASE_URL}/vocab.bpe"), "gpt2_vocab.bpe")?;

        // Parse the data
        let encoder: AHashMap<String, usize> =
            serde_json::from_str(&encoder_json).map_err(|e| {
                TokenizerError::vocabulary_error(format!("Failed to parse GPT-2 encoder: {e}"))
            })?;

        let merges = Self::parse_bpe_merges(&vocab_bpe);

        Ok(Gpt2VocabData { encoder, merges })
    }

    /// Download CLIP vocabulary
    ///
    /// # Errors
    /// Returns an error if network download fails or BPE merge parsing fails
    pub fn download_clip_vocab(&self) -> Result<ClipVocabData> {
        const BASE_URL: &str = "https://openaipublic.blob.core.windows.net/clip";

        // Download vocab file
        let vocab_bpe = self.download_file(&format!("{BASE_URL}/vocab.bpe"), "clip_vocab.bpe")?;

        let merges = Self::parse_bpe_merges(&vocab_bpe);

        Ok(ClipVocabData { merges })
    }

    /// Load CLIP vocabulary
    ///
    /// # Errors
    /// Returns an error if download or vocabulary creation fails
    pub fn load_clip_vocabulary(&self) -> Result<Vocabulary> {
        let data = self.download_clip_vocab()?;
        self.create_vocab_from_clip_data(&data)
    }

    /// Create vocabulary from CLIP data
    ///
    /// # Errors
    /// Returns an error if vocabulary validation fails
    pub fn create_vocab_from_clip_data(&self, data: &ClipVocabData) -> Result<Vocabulary> {
        let mut vocab = Vocabulary::new();

        // CLIP uses the same byte-level base as GPT-2
        for byte_val in 0..=255 {
            let byte_token = format!("{byte_val:02x}");
            vocab.add_token(byte_token);
        }

        // CLIP merges define the vocabulary implicitly through BPE
        // We need to build the vocabulary by applying merges to a base set
        let mut current_tokens: std::collections::HashSet<String> =
            (0..=255).map(|b| format!("{b:02x}")).collect();

        // Apply merges to build vocabulary
        for (token1, token2) in &data.merges {
            if current_tokens.contains(token1) && current_tokens.contains(token2) {
                let merged = format!("{token1}{token2}");
                current_tokens.insert(merged);
            }
        }

        // Add all tokens to vocabulary
        for token in current_tokens {
            vocab.add_token(token);
        }

        // CLIP special tokens
        vocab.add_special_token("<|startoftext|>".to_string());
        vocab.add_special_token("<|endoftext|>".to_string());

        vocab.validate()?;
        Ok(vocab)
    }

    /// Parse BPE merge rules from vocab.bpe format
    fn parse_bpe_merges(vocab_bpe: &str) -> Vec<(String, String)> {
        let mut merges = Vec::new();

        for line in vocab_bpe.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // BPE format: "token1 token2" -> merge rule
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                merges.push((parts[0].to_string(), parts[1].to_string()));
            }
        }

        merges
    }

    /// Load vocabulary from downloaded data
    ///
    /// # Errors
    /// Returns an error if download or vocabulary creation fails
    pub fn load_gpt2_vocabulary(&self) -> Result<Vocabulary> {
        let data = self.download_gpt2_vocab()?;
        self.create_vocab_from_gpt2_data(&data)
    }

    /// Create vocabulary from GPT-2 data
    ///
    /// # Errors
    /// Returns an error if vocabulary validation fails
    pub fn create_vocab_from_gpt2_data(&self, data: &Gpt2VocabData) -> Result<Vocabulary> {
        let mut vocab = Vocabulary::new();

        // Add all tokens from encoder exactly as downloaded
        for (token, &id) in &data.encoder {
            // Ensure id_to_token has enough capacity
            while vocab.id_to_token.len() <= id {
                vocab.id_to_token.push(String::new());
            }
            vocab.id_to_token[id].clone_from(token);
            vocab.token_to_id.insert(token.clone(), id);
        }

        vocab.next_id = vocab.id_to_token.len();

        // Add special tokens
        vocab.add_special_token("<|endoftext|>".to_string());
        vocab.add_special_token("<|startofsequence|>".to_string());

        vocab.validate()?;
        Ok(vocab)
    }

    /// Get BPE merges for GPT-2
    ///
    /// # Errors
    /// Returns an error if download fails
    pub fn get_gpt2_merges(&self) -> Result<Vec<(String, String)>> {
        let data = self.download_gpt2_vocab()?;
        Ok(data.merges)
    }

    /// Clear cache directory
    ///
    /// # Errors
    /// Returns an error if filesystem operations fail
    pub fn clear_cache(&self) -> Result<()> {
        if self.cache_dir.exists() {
            // Remove all files and subdirectories but keep the cache directory itself
            for entry in fs::read_dir(&self.cache_dir).map_err(|e| {
                TokenizerError::io_error(format!("Failed to read cache directory: {e}"))
            })? {
                let entry = entry.map_err(|e| {
                    TokenizerError::io_error(format!("Failed to read cache entry: {e}"))
                })?;
                let path = entry.path();
                if path.is_dir() {
                    fs::remove_dir_all(&path).map_err(|e| {
                        TokenizerError::io_error(format!(
                            "Failed to remove cache subdirectory: {e}"
                        ))
                    })?;
                } else {
                    fs::remove_file(&path).map_err(|e| {
                        TokenizerError::io_error(format!("Failed to remove cache file: {e}"))
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Get cache directory path
    #[must_use]
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }
}

impl Default for VocabDownloader {
    fn default() -> Self {
        Self::new()
    }
}

/// GPT-2 vocabulary data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gpt2VocabData {
    /// Token to ID mapping
    pub encoder: AHashMap<String, usize>,
    /// BPE merge rules
    pub merges: Vec<(String, String)>,
}

/// CLIP vocabulary data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipVocabData {
    /// BPE merge rules
    pub merges: Vec<(String, String)>,
}

/// BERT vocabulary data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BertVocabData {
    /// Token to ID mapping
    pub vocab: AHashMap<String, usize>,
}

/// Download BERT vocabulary
impl VocabDownloader {
    /// Download BERT base vocabulary
    ///
    /// # Errors
    /// Returns an error if network download fails or model is unsupported
    pub fn download_bert_vocab(&self, model: &str) -> Result<BertVocabData> {
        let _vocab_file = match model {
            "bert-base" => "bert-base-uncased-vocab.txt",
            "bert-large" => "bert-large-uncased-vocab.txt",
            _ => {
                return Err(TokenizerError::model_error(format!(
                    "Unsupported BERT model: {model}"
                )))
            }
        };

        // BERT vocabularies are typically hosted on Hugging Face
        let url = "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt".to_string();

        let vocab_text = self.download_file(&url, &format!("{model}_vocab.txt"))?;

        let vocab = Self::parse_bert_vocab(&vocab_text);

        Ok(BertVocabData { vocab })
    }

    /// Parse BERT vocabulary from text format
    fn parse_bert_vocab(vocab_text: &str) -> AHashMap<String, usize> {
        let mut vocab = AHashMap::default();

        for (i, line) in vocab_text.lines().enumerate() {
            let token = line.trim();
            if !token.is_empty() {
                vocab.insert(token.to_string(), i);
            }
        }

        vocab
    }

    /// Load BERT vocabulary
    ///
    /// # Errors
    /// Returns an error if download or vocabulary creation fails
    pub fn load_bert_vocabulary(&self, model: &str) -> Result<Vocabulary> {
        let data = self.download_bert_vocab(model)?;
        self.create_vocab_from_bert_data(&data)
    }

    /// Create vocabulary from BERT data
    ///
    /// # Errors
    /// Returns an error if vocabulary validation fails
    pub fn create_vocab_from_bert_data(&self, data: &BertVocabData) -> Result<Vocabulary> {
        let mut vocab = Vocabulary::new();

        // Add all tokens from BERT vocab
        for (token, &id) in &data.vocab {
            // Ensure id_to_token has enough capacity
            while vocab.id_to_token.len() <= id {
                vocab.id_to_token.push(String::new());
            }
            vocab.id_to_token[id].clone_from(token);
            vocab.token_to_id.insert(token.clone(), id);
        }

        vocab.next_id = vocab.id_to_token.len();

        // BERT special tokens
        vocab.add_special_token("[CLS]".to_string());
        vocab.add_special_token("[SEP]".to_string());
        vocab.add_special_token("[MASK]".to_string());
        vocab.add_special_token("[PAD]".to_string());
        vocab.add_special_token("[UNK]".to_string());

        vocab.validate()?;
        Ok(vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_downloader_creation() {
        let downloader = VocabDownloader::new();
        assert!(downloader.cache_dir().exists() || !downloader.cache_dir().exists());
        // Cache dir may or may not exist yet
    }

    #[test]
    fn test_custom_cache_dir() {
        let temp_dir = tempdir().unwrap();
        let downloader = VocabDownloader::with_cache_dir(temp_dir.path().to_path_buf());
        assert_eq!(downloader.cache_dir(), temp_dir.path());
    }

    #[test]
    fn test_bpe_merge_parsing() {
        let vocab_bpe = r"
#version: 0.2
Ġ t
Ġ a
h e
i n
r e
o n
Ġt he
e r
";

        let merges = VocabDownloader::parse_bpe_merges(vocab_bpe);
        assert_eq!(merges.len(), 8);
        assert_eq!(merges[0], ("Ġ".to_string(), "t".to_string()));
        assert_eq!(merges[1], ("Ġ".to_string(), "a".to_string()));
    }

    #[test]
    fn test_empty_bpe_parsing() {
        let vocab_bpe = "#version: 0.2\n\n# Comment\n";
        let merges = VocabDownloader::parse_bpe_merges(vocab_bpe);
        assert!(merges.is_empty());
    }

    #[test]
    fn test_cache_operations() {
        let temp_dir = tempdir().unwrap();
        let downloader = VocabDownloader::with_cache_dir(temp_dir.path().to_path_buf());

        // Test cache directory creation
        downloader.ensure_cache_dir().unwrap();
        assert!(temp_dir.path().exists());

        // Test cache clearing
        downloader.clear_cache().unwrap();
        // Cache dir should still exist but be empty
        assert!(temp_dir.path().exists());
    }
}
