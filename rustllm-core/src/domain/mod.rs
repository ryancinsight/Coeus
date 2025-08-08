//! Domain-driven design module organization.
//!
//! This module organizes the codebase by business domains rather than technical layers,
//! following Domain-Driven Design (DDD) principles by Eric Evans.
//!
//! ## Domain Structure
//!
//! - **Tokenization**: Text processing and token management domain
//! - **Modeling**: Neural network architecture and model building domain
//! - **Inference**: Model execution and prediction domain
//! - **Training**: Model optimization and learning domain
//! - **Persistence**: Model serialization and storage domain
//!
//! ## Design Principles
//!
//! - **Ubiquitous Language**: Each domain uses terminology from its problem space
//! - **Bounded Contexts**: Clear boundaries between domains with explicit interfaces
//! - **Aggregates**: Domain objects that maintain consistency boundaries
//! - **Value Objects**: Immutable domain concepts
//! - **Domain Events**: Communication between domains through events
//! - **Repository Pattern**: Abstract persistence from domain logic

pub mod inference;
pub mod modeling;
pub mod persistence;
pub mod tokenization;
pub mod training;

// Re-export commonly used domain types
pub mod prelude {
    pub use super::tokenization::{
        Token as DomainToken, TokenizationEvent, TokenizationService, TokenizerAggregate,
        TokenizerRepository,
    };

    pub use super::modeling::{
        Architecture, ModelAggregate, ModelRepository, ModelingEvent, ModelingService,
    };

    pub use super::inference::{
        InferenceEvent, InferenceRequest, InferenceResult, InferenceService, InferenceSession,
    };

    pub use super::training::{
        Optimizer, TrainingConfig, TrainingEvent, TrainingService, TrainingSession,
    };

    pub use super::persistence::{
        Checkpoint, PersistenceEvent, PersistenceService, SerializationFormat, StorageBackend,
    };
}
