//! # Coeus Tokenizer
//!
//! A complete, safe Rust implementation of tokenization algorithms for natural language processing,
//! providing PyTorch-compatible APIs for seamless integration with deep learning workflows.
//!
//! ## Features
//!
//! - **Multiple Algorithms**: BPE, `WordPiece`, and `SentencePiece` tokenization
//! - **`PyTorch` Compatibility**: Drop-in replacement for `HuggingFace` transformers
//! - **Memory Safety**: Zero unsafe code with Miri validation
//! - **Unicode Support**: Proper Unicode normalization and segmentation
//! - **Batch Processing**: Efficient batch tokenization with padding/truncation
//! - **Python Bindings**: `PyO3` integration for Python ecosystem compatibility
//!
//! ## Architecture
//!
//! The tokenizer crate follows Coeus's design principles:
//!
//! - **Trait-Based Design**: Zero-cost abstractions via trait polymorphism
//! - **Iterator Processing**: Lazy evaluation for memory-efficient pipelines
//! - **Typed Errors**: Comprehensive error handling with `thiserror`
//! - **Clean Architecture**: Separation of algorithms, preprocessing, and postprocessing
//!
//! ## Example
//!
//! ```rust
//! use tokenizer::{Tokenizer, Vocabulary, Encoding};
//!
//! // Create a simple vocabulary
//! let mut vocab = Vocabulary::new();
//! vocab.add_token("hello".to_string(), 0)?;
//! vocab.add_token("world".to_string(), 1)?;
//!
//! // Example encoding (would use actual tokenizer implementation)
//! let ids = vec![0, 1];
//! let tokens = vec!["hello".to_string(), "world".to_string()];
//! let encoding = Encoding::new(
//!     ids,
//!     tokens,
//!     vec![(0, 5), (6, 11)], // character offsets
//!     vec![1, 1], // attention mask
//!     vec![0, 0], // token type ids
//!     vec![0, 0], // special tokens mask
//!     11, // original length
//! );
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::cargo)]

pub mod encoding;
pub mod error;
pub mod post_processor;
pub mod pre_tokenizer;
pub mod vocabulary;

#[cfg(feature = "bpe")]
pub mod bpe;
#[cfg(feature = "sentencepiece")]
pub mod sentencepiece;
#[cfg(feature = "wordpiece")]
pub mod wordpiece;

/// Core tokenizer trait providing unified interface for all tokenization algorithms.
pub trait Tokenizer {
    /// Encode text into tokens and IDs.
    ///
    /// # Arguments
    /// * `text` - Input text to tokenize
    ///
    /// # Returns
    /// `Encoding` containing token IDs, tokens, and metadata
    ///
    /// # Errors
    /// Returns `TokenizerError` if encoding fails
    fn encode(&self, text: &str) -> Result<Encoding, error::TokenizerError>;

    /// Decode token IDs back to text.
    ///
    /// # Arguments
    /// * `ids` - Token IDs to decode
    ///
    /// # Returns
    /// Decoded text string
    ///
    /// # Errors
    /// Returns `TokenizerError` if decoding fails
    fn decode(&self, ids: &[u32]) -> Result<String, error::TokenizerError>;

    /// Get the vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Get the tokenizer's vocabulary.
    fn vocabulary(&self) -> &Vocabulary;

    /// Convert tokens to IDs.
    ///
    /// # Arguments
    /// * `tokens` - Token strings to convert
    ///
    /// # Returns
    /// Vector of token IDs
    ///
    /// # Errors
    /// Returns `TokenizerError` for unknown tokens
    fn convert_tokens_to_ids(&self, tokens: &[String]) -> Result<Vec<u32>, error::TokenizerError> {
        let mut ids = Vec::with_capacity(tokens.len());
        for token in tokens {
            let id = self
                .vocabulary()
                .token_to_id(token)
                .ok_or_else(|| error::TokenizerError::UnknownToken(token.clone()))?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Convert IDs to tokens.
    ///
    /// # Arguments
    /// * `ids` - Token IDs to convert
    ///
    /// # Returns
    /// Vector of token strings
    ///
    /// # Errors
    /// Returns `TokenizerError` for invalid IDs
    fn convert_ids_to_tokens(&self, ids: &[u32]) -> Result<Vec<String>, error::TokenizerError> {
        let mut tokens = Vec::with_capacity(ids.len());
        for &id in ids {
            let token = self
                .vocabulary()
                .id_to_token(id)
                .ok_or(error::TokenizerError::InvalidTokenId(id))?;
            tokens.push(token);
        }
        Ok(tokens)
    }
}

/// Batch tokenizer trait for efficient batch processing.
pub trait BatchTokenizer: Tokenizer {
    /// Encode a batch of texts.
    ///
    /// # Arguments
    /// * `texts` - Batch of input texts
    /// * `padding` - Whether to pad sequences to same length
    /// * `truncation` - Whether to truncate sequences to `max_length`
    /// * `max_length` - Maximum sequence length
    ///
    /// # Returns
    /// Batch encoding with padding/truncation applied
    ///
    /// # Errors
    /// Returns `TokenizerError` if batch encoding fails
    fn encode_batch(
        &self,
        texts: &[String],
        padding: bool,
        truncation: bool,
        max_length: Option<usize>,
    ) -> Result<BatchEncoding, error::TokenizerError>;
}

// Re-exports
pub use encoding::{BatchEncoding, Encoding};
pub use post_processor::PostProcessor;
pub use pre_tokenizer::PreTokenizer;
pub use vocabulary::Vocabulary;

// Feature-gated re-exports
#[cfg(feature = "bpe")]
pub use bpe::BpeTokenizer;
#[cfg(feature = "sentencepiece")]
pub use sentencepiece::SentencePieceTokenizer;
#[cfg(feature = "wordpiece")]
pub use wordpiece::WordPieceTokenizer;

/// PyTorch-compatible tokenizer interface.
///
/// Provides methods that match `HuggingFace` transformers API for seamless integration.
pub trait PyTorchTokenizer: Tokenizer + BatchTokenizer {
    /// Encode text with PyTorch-style options.
    ///
    /// # Arguments
    /// * `text` - Input text to tokenize
    /// * `add_special_tokens` - Whether to add special tokens (\[CLS\], \[SEP\], etc.)
    ///
    /// # Returns
    /// Token IDs as vector
    ///
    /// # Errors
    /// Returns `TokenizerError` if encoding fails
    fn encode_pytorch(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, error::TokenizerError> {
        let mut encoding = self.encode(text)?;

        if add_special_tokens {
            // Add special tokens according to tokenizer type
            // This is a simplified implementation - real PyTorch tokenizers
            // have more complex special token handling
            if let Some(cls_id) = self.vocabulary().special_token_id("[CLS]") {
                encoding.ids.insert(0, cls_id);
            }
            if let Some(sep_id) = self.vocabulary().special_token_id("[SEP]") {
                encoding.ids.push(sep_id);
            }
        }

        Ok(encoding.ids)
    }

    /// Batch encode with PyTorch-style API.
    ///
    /// # Arguments
    /// * `texts` - Batch of input texts
    /// * `padding` - Padding strategy ("longest", "`max_length`", or false)
    /// * `truncation` - Whether to truncate sequences
    /// * `max_length` - Maximum sequence length
    /// * `return_tensors` - Ignored (always returns vectors)
    ///
    /// # Returns
    /// Dictionary-like structure with `input_ids`, `attention_mask`, etc.
    ///
    /// # Errors
    /// Returns `TokenizerError` if batch encoding fails
    fn batch_encode_pytorch(
        &self,
        texts: &[String],
        padding: Option<&str>,
        truncation: bool,
        max_length: Option<usize>,
        _return_tensors: Option<&str>,
    ) -> Result<PyTorchBatchEncoding, error::TokenizerError> {
        let padding_enabled = matches!(padding, Some("longest" | "max_length"));
        let batch = self.encode_batch(texts, padding_enabled, truncation, max_length)?;

        Ok(PyTorchBatchEncoding {
            input_ids: batch.input_ids(),
            attention_mask: batch.attention_mask(),
            token_type_ids: Some(batch.token_type_ids()),
        })
    }
}

/// PyTorch-style batch encoding result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyTorchBatchEncoding {
    /// Token IDs for each sequence in batch.
    pub input_ids: Vec<Vec<u32>>,
    /// Attention mask for each sequence.
    pub attention_mask: Vec<Vec<u32>>,
    /// Token type IDs for each sequence (optional).
    pub token_type_ids: Option<Vec<Vec<u32>>>,
}

impl PyTorchBatchEncoding {
    /// Get input IDs.
    #[must_use]
    pub fn input_ids(&self) -> &[Vec<u32>] {
        &self.input_ids
    }

    /// Get attention mask.
    #[must_use]
    pub fn attention_mask(&self) -> &[Vec<u32>] {
        &self.attention_mask
    }

    /// Get token type IDs.
    #[must_use]
    pub fn token_type_ids(&self) -> Option<&[Vec<u32>]> {
        self.token_type_ids.as_deref()
    }
}

// Implement PyTorch trait for all tokenizers
#[cfg(feature = "bpe")]
impl PyTorchTokenizer for BpeTokenizer {}
#[cfg(feature = "wordpiece")]
impl PyTorchTokenizer for WordPieceTokenizer {}
#[cfg(feature = "sentencepiece")]
impl PyTorchTokenizer for SentencePieceTokenizer {}
