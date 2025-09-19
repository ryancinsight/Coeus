//! # Coeus Tokenizer
//!
//! A high-performance, memory-safe tokenizer implementation for the Coeus ML framework,
//! providing tiktoken-compatible functionality with native Rust performance.
//!
//! ## Features
//!
//! - **Byte-Pair Encoding (BPE)**: Core BPE algorithm with vocabulary management
//! - **Popular Models**: GPT-2, GPT-3/4, CLIP, BERT tokenizer variants
//! - **Special Tokens**: Comprehensive special token handling
//! - **Batch Processing**: Efficient batch encoding/decoding operations
//! - **Memory Safety**: Zero unsafe code with Rust's ownership guarantees
//! - **Performance**: Competitive with tiktoken, optimized for ML workflows
//!
//! ## Quick Start
//!
//! ```rust
//! use coeus_tokenizer::{Encoding};
//!
//! // Create a GPT-2 tokenizer (will be available with features)
//! // let tokenizer = Encoding::new("gpt2")?;
//!
//! // For now, create a basic tokenizer example
//! // let tokens = tokenizer.encode("Hello, world!")?;
//! // let text = tokenizer.decode(&tokens)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(missing_docs, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![deny(unsafe_code)]

pub mod bpe;
pub mod encoding;
pub mod error;
pub mod tensor;
pub mod tokenizer;
pub mod vocabulary;

#[cfg(feature = "gpt2")]
pub mod gpt2;

#[cfg(feature = "gpt3")]
pub mod gpt3;

#[cfg(feature = "clip")]
pub mod clip;

#[cfg(feature = "bert")]
pub mod bert;

// Re-export main types
pub use encoding::Encoding;
pub use error::{Result, TokenizerError};
pub use tensor::TensorTokenizer;
pub use tokenizer::{DecodeOptions, TokenizeOptions, Tokenizer};
pub use vocabulary::{VocabEntry, Vocabulary};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default vocabulary size for BPE models
pub const DEFAULT_VOCAB_SIZE: usize = 50257;

/// Maximum sequence length for tokenization
pub const MAX_SEQUENCE_LENGTH: usize = 8192;

/// Common special tokens used across models
pub mod special_tokens {
    /// End of text token
    pub const END_OF_TEXT: &str = "<|endoftext|>";

    /// Start of sequence token
    pub const START_OF_SEQUENCE: &str = "<|startofsequence|>";

    /// Padding token
    pub const PAD: &str = "<|pad|>";

    /// Unknown token
    pub const UNK: &str = "<|unk|>";

    /// Beginning of sentence token (BERT-style)
    pub const BOS: &str = "[BOS]";

    /// End of sentence token (BERT-style)
    pub const EOS: &str = "[EOS]";

    /// Classification token (BERT-style)
    pub const CLS: &str = "[CLS]";

    /// Separation token (BERT-style)
    pub const SEP: &str = "[SEP]";

    /// Mask token (BERT-style)
    pub const MASK: &str = "[MASK]";
}

/// Common model names supported by the tokenizer
pub mod models {
    /// GPT-2 model
    pub const GPT2: &str = "gpt2";

    /// GPT-3.5 Turbo model
    pub const GPT3_5_TURBO: &str = "gpt-3.5-turbo";

    /// GPT-4 model
    pub const GPT4: &str = "gpt-4";

    /// CLIP model
    pub const CLIP: &str = "clip";

    /// BERT base model
    pub const BERT_BASE: &str = "bert-base";

    /// BERT large model
    pub const BERT_LARGE: &str = "bert-large";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        // VERSION is a const string, so this check is always true at compile time
        // We still test it to ensure the constant is properly defined
        let version = VERSION;
        assert!(!version.is_empty(), "VERSION constant should not be empty");
    }

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_VOCAB_SIZE, 50257);
        assert_eq!(MAX_SEQUENCE_LENGTH, 8192);
    }
}
