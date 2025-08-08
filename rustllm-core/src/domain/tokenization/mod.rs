//! Tokenization domain - Text processing and token management.
//!
//! This domain handles all aspects of text tokenization following DDD principles.
//! It provides a rich domain model for tokenization with clear boundaries.

use crate::core::traits::Identity;
use crate::foundation::{error::Result, types::VocabSize};
use core::fmt::Debug;

// ============================================================================
// Value Objects - Immutable domain concepts
// ============================================================================

/// Token as a value object in the tokenization domain.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    id: usize,
    text: String,
    score: Option<f32>,
}

impl Token {
    /// Creates a new token.
    pub const fn new(id: usize, text: String) -> Self {
        Self {
            id,
            text,
            score: None,
        }
    }

    /// Creates a token with score.
    pub const fn with_score(id: usize, text: String, score: f32) -> Self {
        Self {
            id,
            text,
            score: Some(score),
        }
    }

    /// Returns the token ID.
    pub const fn id(&self) -> usize {
        self.id
    }

    /// Returns the token text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the token score if available.
    pub const fn score(&self) -> Option<f32> {
        self.score
    }
}

/// Vocabulary as a value object.
#[derive(Debug, Clone)]
pub struct Vocabulary {
    tokens: Vec<String>,
    token_to_id: std::collections::HashMap<String, usize>,
}

impl Vocabulary {
    /// Creates a new vocabulary.
    pub fn new(tokens: Vec<String>) -> Self {
        let token_to_id = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id))
            .collect();

        Self {
            tokens,
            token_to_id,
        }
    }

    /// Returns the vocabulary size.
    pub fn size(&self) -> VocabSize {
        VocabSize::try_from(self.tokens.len()).expect("vocab size fits")
    }

    /// Looks up a token by text.
    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }

    /// Looks up a token by ID.
    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.tokens.get(id).map(String::as_str)
    }
}

// ============================================================================
// Entities - Domain objects with identity
// ============================================================================

/// Tokenizer configuration entity.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    id: String,
    name: String,
    vocab_size: VocabSize,
    special_tokens: Vec<String>,
    lowercase: bool,
    strip_accents: bool,
}

impl Identity for TokenizerConfig {
    fn id(&self) -> &str {
        &self.id
    }
}

// ============================================================================
// Aggregates - Consistency boundaries
// ============================================================================

/// Tokenizer aggregate - maintains consistency for tokenization operations.
pub struct TokenizerAggregate {
    config: TokenizerConfig,
    vocabulary: Vocabulary,
    state: TokenizerState,
}

#[derive(Debug, Clone)]
enum TokenizerState {
    Initialized,
    Ready,
    Processing,
    Error(String),
}

impl TokenizerAggregate {
    /// Creates a new tokenizer aggregate.
    pub const fn new(config: TokenizerConfig, vocabulary: Vocabulary) -> Self {
        Self {
            config,
            vocabulary,
            state: TokenizerState::Initialized,
        }
    }

    /// Tokenizes text maintaining consistency.
    pub fn tokenize(&mut self, text: &str) -> Result<Vec<Token>> {
        self.state = TokenizerState::Processing;

        // Domain logic for tokenization
        let tokens = text
            .split_whitespace()
            .filter_map(|word| {
                let word = if self.config.lowercase {
                    word.to_lowercase()
                } else {
                    word.to_string()
                };

                self.vocabulary
                    .token_to_id(&word)
                    .map(|id| Token::new(id, word))
            })
            .collect();

        self.state = TokenizerState::Ready;
        Ok(tokens)
    }

    /// Returns the current state.
    pub const fn state(&self) -> &TokenizerState {
        &self.state
    }
}

// ============================================================================
// Domain Services - Stateless operations
// ============================================================================

/// Tokenization service for complex domain operations.
pub struct TokenizationService;

impl TokenizationService {
    /// Merges multiple vocabularies.
    pub fn merge_vocabularies(vocabs: &[Vocabulary]) -> Vocabulary {
        let mut merged_tokens = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for vocab in vocabs {
            for token in &vocab.tokens {
                if seen.insert(token.clone()) {
                    merged_tokens.push(token.clone());
                }
            }
        }

        Vocabulary::new(merged_tokens)
    }

    /// Validates tokenizer configuration.
    pub fn validate_config(config: &TokenizerConfig) -> Result<()> {
        if config.vocab_size == 0 {
            return Err(crate::foundation::error::Error::Config(
                crate::foundation::error::ConfigError::Invalid {
                    key: "vocab_size".to_string(),
                    value: config.vocab_size.to_string(),
                    error: "Vocabulary size must be greater than 0".to_string(),
                },
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Domain Events - Communication between bounded contexts
// ============================================================================

/// Events that occur in the tokenization domain.
#[derive(Debug, Clone)]
pub enum TokenizationEvent {
    /// Tokenization started.
    Started {
        tokenizer_id: String,
        text_length: usize,
    },

    /// Tokenization completed.
    Completed {
        tokenizer_id: String,
        token_count: usize,
    },

    /// Tokenization failed.
    Failed { tokenizer_id: String, error: String },

    /// Vocabulary updated.
    VocabularyUpdated {
        tokenizer_id: String,
        new_size: VocabSize,
    },
}

// ============================================================================
// Repository - Abstract persistence
// ============================================================================

/// Repository trait for tokenizer persistence.
pub trait TokenizerRepository: Send + Sync {
    /// Saves a tokenizer configuration.
    fn save(&self, config: &TokenizerConfig) -> Result<()>;

    /// Loads a tokenizer configuration by ID.
    fn load(&self, id: &str) -> Result<TokenizerConfig>;

    /// Lists all tokenizer configurations.
    fn list(&self) -> Result<Vec<String>>;

    /// Deletes a tokenizer configuration.
    fn delete(&self, id: &str) -> Result<()>;
}

// ============================================================================
// Application Service - Orchestrates use cases
// ============================================================================

/// Application service for tokenization use cases.
pub struct TokenizationApplicationService<R: TokenizerRepository> {
    repository: R,
    domain_service: TokenizationService,
}

impl<R: TokenizerRepository> TokenizationApplicationService<R> {
    /// Creates a new application service.
    pub fn new(repository: R) -> Self {
        Self {
            repository,
            domain_service: TokenizationService,
        }
    }

    /// Creates and saves a new tokenizer.
    pub fn create_tokenizer(
        &self,
        config: TokenizerConfig,
        vocabulary: Vocabulary,
    ) -> Result<TokenizerAggregate> {
        // Validate configuration
        TokenizationService::validate_config(&config)?;

        // Save configuration
        self.repository.save(&config)?;

        // Create aggregate
        Ok(TokenizerAggregate::new(config, vocabulary))
    }

    /// Loads an existing tokenizer.
    pub fn load_tokenizer(&self, id: &str, vocabulary: Vocabulary) -> Result<TokenizerAggregate> {
        let config = self.repository.load(id)?;
        Ok(TokenizerAggregate::new(config, vocabulary))
    }
}
