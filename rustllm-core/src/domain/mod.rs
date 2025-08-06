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

pub mod tokenization;
pub mod modeling;
pub mod inference;
pub mod training;
pub mod persistence;

// Re-export commonly used domain types
pub mod prelude {
    pub use super::tokenization::{
        TokenizationService, TokenizerAggregate, Token as DomainToken,
        TokenizationEvent, TokenizerRepository,
    };
    
    pub use super::modeling::{
        ModelingService, ModelAggregate, Architecture,
        ModelingEvent, ModelRepository,
    };
    
    pub use super::inference::{
        InferenceService, InferenceSession, InferenceRequest,
        InferenceEvent, InferenceResult,
    };
    
    pub use super::training::{
        TrainingService, TrainingSession, TrainingConfig,
        TrainingEvent, Optimizer,
    };
    
    pub use super::persistence::{
        PersistenceService, StorageBackend, SerializationFormat,
        PersistenceEvent, Checkpoint,
    };
}