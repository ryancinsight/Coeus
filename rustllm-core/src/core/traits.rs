//! Core traits defining the fundamental abstractions of RustLLM Core.
//!
//! This module contains the essential traits that all components must implement,
//! following the Interface Segregation Principle (ISP) and Single Responsibility
//! Principle (SRP) from SOLID.
//!
//! ## Design Principles
//!
//! - **Single Responsibility**: Each trait has one clear purpose
//! - **Open/Closed**: Traits are open for extension through composition
//! - **Liskov Substitution**: All implementations are interchangeable
//! - **Interface Segregation**: Small, focused traits that can be combined
//! - **Dependency Inversion**: Depend on abstractions, not concretions
//! - **Composable**: Traits can be combined to create complex behaviors
//! - **Unix Philosophy**: Do one thing well
//! - **Predictable**: Clear contracts and behavior
//! - **Idiomatic**: Follow Rust patterns and conventions
//! - **Domain-based**: Traits reflect domain concepts
//!
//! ## Organization
//!
//! The traits are organized by domain following DDD principles:
//!
//! 1. **Identity** - Component identification and metadata
//! 2. **Lifecycle** - Component lifecycle management
//! 3. **Processing** - Data transformation and computation
//! 4. **Persistence** - State management and serialization
//! 5. **Monitoring** - Observability and diagnostics

use crate::foundation::error::Result;
use crate::foundation::types::Version;
use core::fmt::Debug;

// ============================================================================
// Identity Domain - Component identification following SRP
// ============================================================================

/// Core identity trait - every component must be identifiable.
/// This is the foundation of our type system following ISP.
pub trait Identity: Debug + Send + Sync {
    /// Returns the unique identifier for this component.
    fn id(&self) -> &str;
    
    /// Returns the component type name for runtime type information.
    fn type_name(&self) -> &'static str {
        core::any::type_name::<Self>()
    }
}

/// Versioning trait for components that evolve over time.
/// Separated from Identity following ISP - not all components need versioning.
pub trait Versioned {
    /// Returns the component's semantic version.
    fn version(&self) -> Version;
}

/// Metadata trait for components that provide additional information.
/// This follows the Open/Closed principle - extend through composition.
pub trait Metadata {
    /// Returns human-readable description.
    fn description(&self) -> &str;
    
    /// Returns component tags for categorization.
    fn tags(&self) -> &[&str] {
        &[]
    }
}

// ============================================================================
// Lifecycle Domain - Component lifecycle following State pattern
// ============================================================================

/// Core lifecycle trait defining component state transitions.
/// This follows the State pattern and enables ACID properties.
pub trait Lifecycle: Identity {
    /// Component state enumeration.
    type State: Debug + Clone + Send + Sync;
    
    /// Returns the current state.
    fn state(&self) -> Self::State;
    
    /// Initializes the component.
    fn initialize(&mut self) -> Result<()>;
    
    /// Shuts down the component gracefully.
    fn shutdown(&mut self) -> Result<()>;
    
    /// Validates component state for consistency (ACID).
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

    /// Async lifecycle for components that require async operations.
    /// This extends Lifecycle following the Open/Closed principle.
    /// Note: Using explicit Future return types to avoid async trait limitations.
    #[cfg(feature = "async")]
    pub trait AsyncLifecycle: Lifecycle {
        /// The future type for initialization.
        type InitFuture: core::future::Future<Output = Result<()>> + Send;
        
        /// The future type for shutdown.
        type ShutdownFuture: core::future::Future<Output = Result<()>> + Send;
        
        /// Async initialization returning a future.
        fn initialize_async(&mut self) -> Self::InitFuture;
        
        /// Async shutdown returning a future.
        fn shutdown_async(&mut self) -> Self::ShutdownFuture;
    }

// ============================================================================
// Processing Domain - Data transformation following functional principles
// ============================================================================

/// Core processing trait for transforming data.
/// This follows functional programming principles and enables composition.
pub trait Process: Identity {
    /// Input type for processing.
    type Input;
    
    /// Output type after processing.
    type Output;
    
    /// Processes input and produces output.
    /// This is a pure function following functional principles.
    fn process(&self, input: Self::Input) -> Result<Self::Output>;
}

/// Streaming processor for handling data streams.
/// This enables zero-copy processing through iterators.
pub trait StreamProcess: Process
where
    Self::Input: IntoIterator,
    Self::Output: IntoIterator,
{
    /// Processes a stream of inputs into a stream of outputs.
    /// Default implementation uses iterator combinators for efficiency.
    fn process_stream<I>(&self, inputs: I) -> impl Iterator<Item = Result<Self::Output>>
    where
        I: IntoIterator<Item = Self::Input>,
    {
        inputs.into_iter().map(move |input| self.process(input))
    }
}

/// Batch processor for efficient bulk operations.
/// This follows the Flyweight pattern for memory efficiency.
pub trait BatchProcess: Process
where
    Self::Input: Clone,
{
    /// Processes multiple inputs in a single operation.
    /// This enables SIMD and other optimizations.
    fn process_batch(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>> {
        inputs.iter().map(|input| self.process(input.clone())).collect()
    }
}

// ============================================================================
// Persistence Domain - State management following Repository pattern
// ============================================================================

/// Serialization trait for components that can be persisted.
/// This follows the Repository pattern and enables ACID durability.
pub trait Serialize: Identity {
    /// Serializes the component to bytes.
    fn serialize(&self) -> Result<Vec<u8>>;
    
    /// Deserializes from bytes.
    fn deserialize(data: &[u8]) -> Result<Self>
    where
        Self: Sized;
}

/// Checkpoint trait for components that support state snapshots.
/// This enables transactional semantics and recovery.
pub trait Checkpoint: Serialize {
    /// Creates a checkpoint of current state.
    fn checkpoint(&self) -> Result<Vec<u8>> {
        self.serialize()
    }
    
    /// Restores from a checkpoint.
    fn restore(&mut self, checkpoint: &[u8]) -> Result<()>;
}

// ============================================================================
// Monitoring Domain - Observability following Observer pattern
// ============================================================================

/// Health check trait for component monitoring.
/// This follows the Observer pattern for system observability.
pub trait HealthCheck: Identity {
    /// Health status enumeration.
    type Status: Debug + Send + Sync;
    
    /// Returns current health status.
    fn health(&self) -> Self::Status;
    
    /// Performs a health check.
    fn check_health(&self) -> Result<Self::Status> {
        Ok(self.health())
    }
}

/// Metrics trait for performance monitoring.
/// This enables profiling and optimization.
pub trait Metrics: Identity {
    /// Metric type for this component.
    type Metric: Debug + Send + Sync;
    
    /// Returns current metrics.
    fn metrics(&self) -> Vec<Self::Metric>;
    
    /// Resets metrics to initial state.
    fn reset_metrics(&mut self);
}

// ============================================================================
// Composite Traits - Combining domains for complex components
// ============================================================================

/// Full-featured component combining all domains.
/// This demonstrates trait composition following CUPID principles.
pub trait Component: 
    Identity + 
    Versioned + 
    Metadata + 
    Lifecycle + 
    HealthCheck + 
    Send + 
    Sync 
{
    /// Helper method to get full component info.
    fn info(&self) -> ComponentInfo {
        ComponentInfo {
            id: self.id().to_string(),
            type_name: self.type_name(),
            version: self.version(),
            description: self.description().to_string(),
            state: format!("{:?}", self.state()),
            health: format!("{:?}", self.health()),
        }
    }
}

/// Component information structure.
#[derive(Debug, Clone)]
pub struct ComponentInfo {
    /// Component ID.
    pub id: String,
    /// Type name.
    pub type_name: &'static str,
    /// Version.
    pub version: Version,
    /// Description.
    pub description: String,
    /// Current state.
    pub state: String,
    /// Health status.
    pub health: String,
}

// ============================================================================
// Extension Traits - Advanced patterns for specific use cases
// ============================================================================

/// Pipeline trait for composing processors.
/// This follows the Chain of Responsibility pattern.
pub trait Pipeline: Process {
    /// Chains this processor with another.
    fn pipe<P>(self, next: P) -> Piped<Self, P>
    where
        Self: Sized,
        P: Process<Input = Self::Output>,
    {
        Piped::new(self, next)
    }
}

/// Piped processor implementation.
#[derive(Debug, Clone)]
pub struct Piped<T, P> {
    first: T,
    second: P,
}

impl<T, P> Piped<T, P> {
    /// Creates a new piped processor.
    pub const fn new(first: T, second: P) -> Self {
        Self { first, second }
    }
}

impl<T, P> Identity for Piped<T, P>
where
    T: Identity,
    P: Identity,
{
    fn id(&self) -> &str {
        self.first.id()
    }
}

impl<T, P> Process for Piped<T, P>
where
    T: Process,
    P: Process<Input = T::Output>,
{
    type Input = T::Input;
    type Output = P::Output;
    
    fn process(&self, input: Self::Input) -> Result<Self::Output> {
        let intermediate = self.first.process(input)?;
        self.second.process(intermediate)
    }
}

// Blanket implementation for all processors
impl<T: Process> Pipeline for T {}

// ============================================================================
// Factory Traits - Creation patterns following Factory pattern
// ============================================================================

/// Factory trait for creating components.
/// This follows the Abstract Factory pattern.
pub trait Factory: Identity {
    /// The type of component this factory creates.
    type Product: Identity;
    
    /// Configuration type for creation.
    type Config: Debug + Send + Sync;
    
    /// Creates a new instance.
    fn create(&self, config: Self::Config) -> Result<Self::Product>;
}

/// Builder trait for complex component construction.
/// This follows the Builder pattern for step-by-step construction.
pub trait Builder: Identity {
    /// The type being built.
    type Product: Identity;
    
    /// Builds the final product.
    fn build(self) -> Result<Self::Product>;
}

// ============================================================================
// Adapter Traits - Integration patterns following Adapter pattern
// ============================================================================

/// Adapter trait for converting between types.
/// This follows the Adapter pattern for integration.
pub trait Adapter<T>: Identity {
    /// Adapts from the source type.
    fn adapt_from(source: T) -> Result<Self>
    where
        Self: Sized;
    
    /// Adapts to the target type.
    fn adapt_to(self) -> Result<T>;
}

// ============================================================================
// Strategy Traits - Behavioral patterns following Strategy pattern
// ============================================================================

/// Strategy trait for interchangeable algorithms.
/// This follows the Strategy pattern for runtime polymorphism.
pub trait Strategy: Identity {
    /// Context type for the strategy.
    type Context;
    
    /// Result type of the strategy.
    type Result;
    
    /// Executes the strategy.
    fn execute(&self, context: &Self::Context) -> Result<Self::Result>;
}

// ============================================================================
// Iterator Extension Traits - Advanced iteration patterns
// ============================================================================

/// Trait for components that produce iterators.
/// This enables zero-copy streaming and lazy evaluation.
pub trait IntoIterator: Identity {
    /// The item type produced by the iterator.
    type Item;
    
    /// The iterator type.
    type IntoIter: Iterator<Item = Self::Item>;
    
    /// Converts into an iterator.
    fn into_iter(self) -> Self::IntoIter;
}

/// Trait for components that can be created from iterators.
/// This enables efficient construction from streams.
pub trait FromIterator<T>: Identity + Sized {
    /// Creates from an iterator.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Result<Self>;
}

// ============================================================================
// Validation Traits - Data integrity following DDD principles
// ============================================================================

/// Validation trait for ensuring data integrity.
/// This follows Domain-Driven Design principles.
pub trait Validate {
    /// Validation error type.
    type Error: Debug + Send + Sync;
    
    /// Validates the data according to domain rules.
    fn validate(&self) -> core::result::Result<(), Self::Error>;
    
    /// Checks if the data is valid.
    fn is_valid(&self) -> bool {
        self.validate().is_ok()
    }
}

/// Constraint trait for defining validation rules.
/// This enables declarative validation following CUPID.
pub trait Constraint<T> {
    /// Checks if the value satisfies the constraint.
    fn check(&self, value: &T) -> bool;
    
    /// Returns a description of the constraint.
    fn description(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Example implementation for testing
    #[derive(Debug)]
    struct TestComponent {
        id: String,
        version: Version,
    }
    
    impl Identity for TestComponent {
        fn id(&self) -> &str {
            &self.id
        }
    }
    
    impl Versioned for TestComponent {
        fn version(&self) -> Version {
            self.version
        }
    }
    
    #[test]
    fn test_identity_trait() {
        let component = TestComponent {
            id: "test-1".to_string(),
            version: Version::new(1, 0, 0),
        };
        
        assert_eq!(component.id(), "test-1");
        assert_eq!(component.type_name(), "rustllm_core::core::traits::tests::TestComponent");
    }
    
    #[test]
    fn test_pipeline_composition() {
        struct AddOne;
        struct MultiplyTwo;
        
        impl Identity for AddOne {
            fn id(&self) -> &str { "add-one" }
        }
        
        impl Process for AddOne {
            type Input = i32;
            type Output = i32;
            
            fn process(&self, input: Self::Input) -> Result<Self::Output> {
                Ok(input + 1)
            }
        }
        
        impl Identity for MultiplyTwo {
            fn id(&self) -> &str { "multiply-two" }
        }
        
        impl Process for MultiplyTwo {
            type Input = i32;
            type Output = i32;
            
            fn process(&self, input: Self::Input) -> Result<Self::Output> {
                Ok(input * 2)
            }
        }
        
        let pipeline = AddOne.pipe(MultiplyTwo);
        let result = pipeline.process(5).unwrap();
        assert_eq!(result, 12); // (5 + 1) * 2 = 12
    }
}