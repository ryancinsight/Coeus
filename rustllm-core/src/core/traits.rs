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

use crate::foundation::error::Result;
use core::fmt::Debug;

// ============================================================================
// Foundational Traits (CUPID: Unix Philosophy - Do One Thing Well)
// ============================================================================

/// Trait for components that can be identified by name.
/// 
/// This is the most fundamental trait, following SRP by focusing solely
/// on identification. This enables the Unix philosophy of composability.
pub trait Named {
    /// Returns the component's name.
    fn name(&self) -> &str;
}

/// Trait for components with type information.
/// 
/// Separated from Named to follow ISP - not all named things need type info.
pub trait Typed {
    /// Returns the component's type identifier.
    fn type_name(&self) -> &'static str {
        core::any::type_name::<Self>()
    }
}

// ============================================================================
// Capability Traits (SOLID: Interface Segregation)
// ============================================================================

/// Marker trait for components that support cloning.
/// 
/// This allows runtime capability checking without forcing Clone on all types.
pub trait Cloneable: Send + Sync {
    /// Clones the component into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Cloneable>;
}

/// Trait for components that can be serialized.
/// 
/// Follows ISP by not forcing serialization on all components.
pub trait Serializable: Send + Sync {
    /// Serializes the component to bytes.
    fn serialize(&self) -> Result<Vec<u8>>;
    
    /// Deserializes from bytes.
    fn deserialize(data: &[u8]) -> Result<Self>
    where
        Self: Sized;
}

// ============================================================================
// Single Responsibility Traits
// ============================================================================

/// Trait for components that can be initialized.
/// 
/// This trait follows SRP by focusing solely on initialization concerns.
pub trait Initialize {
    /// Initializes the component.
    fn initialize(&mut self) -> Result<()>;
    
    /// Checks if the component is initialized.
    fn is_initialized(&self) -> bool;
}

/// Trait for components that can be reset to their initial state.
/// 
/// Separated from Initialize to follow ISP - not all initializable
/// components need to be resettable.
pub trait Reset {
    /// Resets the component to its initial state.
    fn reset(&mut self) -> Result<()>;
}

/// Trait for components that can be validated.
/// 
/// This trait enables defensive programming and follows SRP.
pub trait Validate {
    /// Validates the component's state.
    fn validate(&self) -> Result<()>;
}

// ============================================================================
// Data Processing Traits (Following ISP)
// ============================================================================

/// Trait for single-item processing.
/// 
/// This is the most basic processing trait, following ISP by not
/// forcing batch processing on components that don't need it.
pub trait Process: Send + Sync {
    /// The input type.
    type Input;
    
    /// The output type.
    type Output;
    
    /// Processes a single input.
    fn process(&self, input: Self::Input) -> Result<Self::Output>;
}

/// Extension trait for batch processing.
/// 
/// Separated from Process to follow ISP - components can implement
/// Process without being forced to implement batch processing.
pub trait BatchProcess: Process {
    /// Processes a batch of inputs efficiently.
    /// 
    /// Default implementation uses the single-item process method.
    fn process_batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        inputs.into_iter()
            .map(|input| self.process(input))
            .collect()
    }
    
    /// Returns the optimal batch size for this processor.
    fn optimal_batch_size(&self) -> usize {
        32
    }
}

/// Trait for streaming processing with state.
/// 
/// This trait is for processors that maintain state between calls,
/// following SRP by separating stateful from stateless processing.
pub trait StreamProcess: Send + Sync {
    /// The input type.
    type Input;
    
    /// The output type.
    type Output;
    
    /// The internal state type.
    type State: Default;
    
    /// Processes input with state.
    fn process_with_state(
        &self,
        input: Self::Input,
        state: &mut Self::State,
    ) -> Result<Self::Output>;
    
    /// Flushes any buffered data in the state.
    fn flush_state(&self, _state: &mut Self::State) -> Result<Option<Self::Output>> {
        Ok(None)
    }
}

// ============================================================================
// Iterator-Based Processing Traits
// ============================================================================

/// Trait for components that produce iterators.
/// 
/// This trait embraces Rust's iterator pattern for zero-cost abstractions.
pub trait IteratorSource: Send + Sync {
    /// The item type produced by the iterator.
    type Item;
    
    /// The iterator type.
    type Iter: Iterator<Item = Self::Item>;
    
    /// Creates an iterator from this source.
    fn iter(&self) -> Self::Iter;
}

/// Trait for components that can process iterators.
/// 
/// This trait allows for efficient stream processing with iterator chains.
pub trait IteratorProcessor: Send + Sync {
    /// The input item type.
    type Input;
    
    /// The output item type.
    type Output;
    
    /// Processes an iterator of inputs into an iterator of outputs.
    fn process_iter<I>(&self, iter: I) -> ProcessIterator<I, Self>
    where
        I: Iterator<Item = Self::Input>,
        Self: Sized,
    {
        ProcessIterator::new(iter, self)
    }
}

/// Iterator adapter for processing.
pub struct ProcessIterator<'a, I, P> {
    iter: I,
    processor: &'a P,
}

impl<'a, I, P> ProcessIterator<'a, I, P> {
    fn new(iter: I, processor: &'a P) -> Self {
        Self { iter, processor }
    }
}

impl<'a, I, P> Iterator for ProcessIterator<'a, I, P>
where
    I: Iterator,
    P: Process<Input = I::Item>,
{
    type Item = Result<P::Output>;
    
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| self.processor.process(item))
    }
}

// ============================================================================
// Functional Traits (CUPID: Composable)
// ============================================================================

/// Trait for pure transformations.
/// 
/// This trait represents pure functions with no side effects,
/// following functional programming principles.
pub trait Transform<T, U>: Send + Sync {
    /// Transforms input of type T to output of type U.
    fn transform(&self, input: T) -> U;
}

/// Trait for fallible transformations.
/// 
/// Separated from Transform to follow ISP.
pub trait TryTransform<T, U>: Send + Sync {
    /// Attempts to transform input of type T to output of type U.
    fn try_transform(&self, input: T) -> Result<U>;
}

/// Trait for transformations that consume the transformer.
/// 
/// This enables move semantics and zero-copy optimizations.
pub trait IntoTransform<T, U> {
    /// Transforms input consuming self.
    fn into_transform(self, input: T) -> U;
}

// ============================================================================
// Async Traits (CUPID: Predictable)
// ============================================================================

/// Trait for async initialization.
/// 
/// Separated from sync Initialize to follow ISP and support async/await.
#[cfg(feature = "std")]
pub trait AsyncInitialize: Send + Sync {
    /// Asynchronously initializes the component.
    async fn initialize_async(&mut self) -> Result<()>;
}

/// Trait for async processing.
/// 
/// This enables non-blocking I/O and concurrent processing.
#[cfg(feature = "std")]
pub trait AsyncProcess: Send + Sync {
    /// The input type.
    type Input: Send;
    
    /// The output type.
    type Output: Send;
    
    /// Asynchronously processes input.
    async fn process_async(&self, input: Self::Input) -> Result<Self::Output>;
}

// ============================================================================
// Configuration Traits (Following DIP)
// ============================================================================

/// Trait for components with configuration.
/// 
/// This trait follows Dependency Inversion Principle by depending
/// on the abstract Config type rather than concrete types.
pub trait Configurable {
    /// The configuration type.
    type Config: ConfigBounds;
    
    /// Applies configuration to the component.
    fn configure(&mut self, config: Self::Config) -> Result<()>;
    
    /// Returns the current configuration.
    fn config(&self) -> &Self::Config;
}

/// Bounds for configuration types.
/// 
/// This trait defines the requirements for configuration types,
/// following ISP by keeping the requirements minimal.
pub trait ConfigBounds: Debug + Clone + Validate {}

// Blanket implementation for types that meet the bounds
impl<T> ConfigBounds for T where T: Debug + Clone + Validate {}

// ============================================================================
// Metadata Traits (Following SRP)
// ============================================================================

/// Trait for components with basic metadata.
/// 
/// Refactored to be even more minimal following ISP.
/// The Named trait has been moved to foundational traits.

/// Trait for components with version information.
pub trait Versioned {
    /// Returns the component's version.
    fn version(&self) -> crate::foundation::types::Version;
}

/// Trait for components with description.
pub trait Described {
    /// Returns the component's description.
    fn description(&self) -> &str;
}

/// Trait for components with author information.
/// 
/// Added to follow ISP - not all components need author info.
pub trait Authored {
    /// Returns the component's author.
    fn author(&self) -> &str;
}

/// Trait for components with license information.
pub trait Licensed {
    /// Returns the component's license.
    fn license(&self) -> &str;
}

/// Combined metadata trait for convenience.
/// 
/// This follows ISP by allowing components to implement only
/// the metadata traits they need.
pub trait Metadata: Named + Versioned + Described {}

/// Extended metadata trait.
pub trait ExtendedMetadata: Metadata + Authored + Licensed {}

// Blanket implementations
impl<T> Metadata for T where T: Named + Versioned + Described {}
impl<T> ExtendedMetadata for T where T: Metadata + Authored + Licensed {}

// ============================================================================
// Event-Driven Traits (CUPID: Domain-based)
// ============================================================================

/// Trait for event producers.
/// 
/// This enables event-driven architectures following domain patterns.
pub trait EventProducer: Send + Sync {
    /// The event type produced.
    type Event: Send + Sync;
    
    /// Subscribes a handler to events.
    fn subscribe(&mut self, handler: Box<dyn EventHandler<Event = Self::Event>>);
    
    /// Publishes an event to all subscribers.
    fn publish(&self, event: Self::Event);
}

/// Trait for event handlers.
pub trait EventHandler: Send + Sync {
    /// The event type handled.
    type Event;
    
    /// Handles an event.
    fn handle(&mut self, event: Self::Event);
}

/// Trait for event filters.
/// 
/// Follows SRP by separating filtering from handling.
pub trait EventFilter: Send + Sync {
    /// The event type filtered.
    type Event;
    
    /// Determines if an event should be processed.
    fn should_handle(&self, event: &Self::Event) -> bool;
}

// ============================================================================
// Chain of Responsibility Pattern (SOLID: Open/Closed)
// ============================================================================

/// Trait for handlers in a chain of responsibility.
/// 
/// This pattern allows extending behavior without modifying existing code.
pub trait ChainHandler: Send + Sync {
    /// The request type.
    type Request;
    
    /// The response type.
    type Response;
    
    /// Handles a request or passes it to the next handler.
    fn handle_or_pass(
        &self,
        request: Self::Request,
        next: Option<&dyn ChainHandler<Request = Self::Request, Response = Self::Response>>,
    ) -> Result<Self::Response>;
    
    /// Checks if this handler can handle the request.
    fn can_handle(&self, request: &Self::Request) -> bool;
}

// ============================================================================
// Strategy Pattern Traits (SOLID: Dependency Inversion)
// ============================================================================

/// Trait for strategy selection.
/// 
/// This enables runtime strategy selection following DIP.
pub trait Strategy: Send + Sync {
    /// The context type for the strategy.
    type Context;
    
    /// The result type.
    type Result;
    
    /// Executes the strategy.
    fn execute(&self, context: &Self::Context) -> Self::Result;
}

/// Trait for strategy selectors.
pub trait StrategySelector<S: Strategy>: Send + Sync {
    /// Selects a strategy based on context.
    fn select(&self, context: &S::Context) -> &dyn Strategy<Context = S::Context, Result = S::Result>;
}

// ============================================================================
// Visitor Pattern Traits (CUPID: Predictable)
// ============================================================================

/// Trait for visitable elements.
/// 
/// This enables the visitor pattern for extensible operations.
pub trait Visitable {
    /// Accepts a visitor.
    fn accept<V: Visitor>(&self, visitor: &mut V) -> Result<()>;
}

/// Trait for visitors.
pub trait Visitor: Send + Sync {
    /// Called when visiting starts.
    fn begin_visit(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Called when visiting ends.
    fn end_visit(&mut self) -> Result<()> {
        Ok(())
    }
}

// ============================================================================
// Observer Pattern Traits (CUPID: Composable)
// ============================================================================

/// Trait for observable subjects.
/// 
/// This enables loose coupling between components.
pub trait Observable: Send + Sync {
    /// The state type being observed.
    type State: Clone + Send + Sync;
    
    /// Attaches an observer.
    fn attach(&mut self, observer: Box<dyn Observer<State = Self::State>>);
    
    /// Detaches an observer.
    fn detach(&mut self, id: usize);
    
    /// Notifies all observers of state change.
    fn notify(&self, state: &Self::State);
}

/// Trait for observers.
pub trait Observer: Send + Sync {
    /// The state type being observed.
    type State;
    
    /// Called when the observed state changes.
    fn update(&mut self, state: &Self::State);
    
    /// Returns the observer's unique ID.
    fn id(&self) -> usize;
}

// ============================================================================
// Monitoring Traits (Following SRP and ISP)
// ============================================================================

/// Trait for components that provide metrics.
pub trait Metrics {
    /// Returns current metrics.
    fn metrics(&self) -> MetricsSnapshot;
}

/// Trait for components that provide health status.
pub trait HealthCheck {
    /// Returns current health status.
    fn health(&self) -> HealthStatus;
}

/// Metrics snapshot.
#[derive(Debug, Clone, Default)]
pub struct MetricsSnapshot {
    /// Total operations performed.
    pub operations: u64,
    
    /// Total processing time in microseconds.
    pub total_time_us: u64,
    
    /// Number of errors encountered.
    pub errors: u64,
    
    /// Current memory usage in bytes.
    pub memory_bytes: usize,
}

impl MetricsSnapshot {
    /// Creates a new metrics snapshot.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Calculates average processing time per operation.
    pub fn avg_time_us(&self) -> f64 {
        if self.operations == 0 {
            0.0
        } else {
            self.total_time_us as f64 / self.operations as f64
        }
    }
    
    /// Calculates error rate.
    pub fn error_rate(&self) -> f64 {
        if self.operations == 0 {
            0.0
        } else {
            self.errors as f64 / self.operations as f64
        }
    }
}

/// Health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Component is healthy and operating normally.
    Healthy,
    
    /// Component is operational but degraded.
    Degraded,
    
    /// Component is not operational.
    Unhealthy,
}

// ============================================================================
// Lifecycle Traits (Following ISP)
// ============================================================================

/// Trait for components with start/stop lifecycle.
/// 
/// This trait is separate from Initialize/Reset to follow ISP.
pub trait Lifecycle: Send + Sync {
    /// Starts the component.
    fn start(&mut self) -> Result<()>;
    
    /// Stops the component.
    fn stop(&mut self) -> Result<()>;
}

/// Trait for components that support pause/resume.
/// 
/// Separated from Lifecycle to follow ISP - not all components
/// need pause/resume functionality.
pub trait Pausable: Lifecycle {
    /// Pauses the component.
    fn pause(&mut self) -> Result<()>;
    
    /// Resumes the component.
    fn resume(&mut self) -> Result<()>;
}

// ============================================================================
// Composition Traits (Following OCP and DIP)
// ============================================================================

/// Trait for components that can be composed.
/// 
/// This trait follows Open/Closed Principle by allowing extension
/// through composition without modifying existing code.
pub trait Compose: Sized {
    /// The output type when composed.
    type Output;
    
    /// Composes this component with another.
    fn compose<T>(self, other: T) -> Composed<Self, T>
    where
        T: Compose,
    {
        Composed::new(self, other)
    }
}

/// Trait for components that can be piped.
/// 
/// This enables Unix-style pipelines following CUPID principles.
pub trait Pipe: Sized {
    /// Pipes the output of this component to another.
    fn pipe<T, F>(self, f: F) -> Piped<Self, F>
    where
        F: FnOnce(Self) -> T,
    {
        Piped::new(self, f)
    }
}

/// A piped component.
pub struct Piped<T, F> {
    inner: T,
    func: F,
}

impl<T, F> Piped<T, F> {
    /// Creates a new piped component.
    pub fn new(inner: T, func: F) -> Self {
        Self { inner, func }
    }
}

// ============================================================================
// Pipeline Pattern Traits (CUPID: Unix Philosophy)
// ============================================================================

/// Trait for pipeline stages.
/// 
/// This enables building complex processing pipelines from simple stages.
pub trait PipelineStage: Send + Sync {
    /// The input type.
    type Input: Send;
    
    /// The output type.
    type Output: Send;
    
    /// Processes input through this stage.
    fn process_stage(&self, input: Self::Input) -> Result<Self::Output>;
}

/// Trait for complete pipelines.
pub trait Pipeline: Send + Sync {
    /// The input type.
    type Input: Send;
    
    /// The output type.
    type Output: Send;
    
    /// Executes the entire pipeline.
    fn execute(&self, input: Self::Input) -> Result<Self::Output>;
    
    /// Returns the number of stages in the pipeline.
    fn stage_count(&self) -> usize;
}

/// Builder for pipelines.
pub struct PipelineBuilder<I, O> {
    stages: Vec<Box<dyn PipelineStage<Input = I, Output = O>>>,
}

impl<I: Send + 'static, O: Send + 'static> PipelineBuilder<I, O> {
    /// Creates a new pipeline builder.
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
        }
    }
    
    /// Adds a stage to the pipeline.
    pub fn add_stage<S>(mut self, stage: S) -> Self
    where
        S: PipelineStage<Input = I, Output = O> + 'static,
    {
        self.stages.push(Box::new(stage));
        self
    }
}

// ============================================================================
// Adapter Pattern Traits (SOLID: Liskov Substitution)
// ============================================================================

/// Trait for adapters that convert between types.
/// 
/// This enables adapting incompatible interfaces following LSP.
pub trait Adapter<From, To>: Send + Sync {
    /// Adapts from one type to another.
    fn adapt(&self, from: From) -> Result<To>;
}

/// Trait for bidirectional adapters.
pub trait BidirectionalAdapter<A, B>: Adapter<A, B> + Adapter<B, A> {
    /// Adapts from A to B (alias for adapt).
    fn forward(&self, a: A) -> Result<B> {
        self.adapt(a)
    }
    
    /// Adapts from B to A.
    fn backward(&self, b: B) -> Result<A> {
        <Self as Adapter<B, A>>::adapt(self, b)
    }
}

// ============================================================================
// Functional Composition Traits (CUPID: Composable)
// ============================================================================

/// Function composition helper.
/// 
/// This enables point-free style programming.
pub fn compose<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> C
where
    F: Fn(A) -> B,
    G: Fn(B) -> C,
{
    move |a| g(f(a))
}

/// Trait for partial application.
/// 
/// This enables partial application following functional patterns.
pub trait PartialApply<A, B, C>: Sized {
    /// Partially applies the first argument.
    fn partial_apply(self, a: A) -> Box<dyn Fn(B) -> C>;
}

impl<F, A, B, C> PartialApply<A, B, C> for F
where
    F: Fn(A, B) -> C + 'static,
    A: Clone + 'static,
    B: 'static,
    C: 'static,
{
    fn partial_apply(self, a: A) -> Box<dyn Fn(B) -> C> {
        Box::new(move |b| self(a.clone(), b))
    }
}

// ============================================================================
// Monad-like Traits (CUPID: Predictable)
// ============================================================================

/// Trait for types that can be mapped over.
/// 
/// This is similar to Functor in functional programming.
pub trait Mappable<T> {
    /// The mapped type.
    type Mapped<U>;
    
    /// Maps a function over the contained value.
    fn map<U, F>(self, f: F) -> Self::Mapped<U>
    where
        F: FnOnce(T) -> U;
}

/// Trait for types that can be flat-mapped.
/// 
/// This is similar to Monad in functional programming.
pub trait FlatMappable<T>: Mappable<T> {
    /// Flat-maps a function over the contained value.
    fn flat_map<U, F>(self, f: F) -> Self::Mapped<U>
    where
        F: FnOnce(T) -> Self::Mapped<U>;
}

// ============================================================================
// Resource Management Traits (SOLID: Single Responsibility)
// ============================================================================

/// Trait for components that manage resources.
/// 
/// This follows SRP by focusing solely on resource lifecycle.
pub trait Resource: Send + Sync {
    /// Acquires the resource.
    fn acquire(&mut self) -> Result<()>;
    
    /// Releases the resource.
    fn release(&mut self) -> Result<()>;
    
    /// Checks if the resource is acquired.
    fn is_acquired(&self) -> bool;
}

/// Trait for components that can be disposed.
/// 
/// This is similar to IDisposable in other languages.
pub trait Disposable {
    /// Disposes of the component, releasing all resources.
    fn dispose(self);
}

/// A composed component.
#[derive(Debug, Clone)]
pub struct Composed<A, B> {
    first: A,
    second: B,
}

impl<A, B> Composed<A, B> {
    /// Creates a new composed component.
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
    
    /// Decomposes into the constituent components.
    pub fn decompose(self) -> (A, B) {
        (self.first, self.second)
    }
}

impl<A, B> Process for Composed<A, B>
where
    A: Process,
    B: Process<Input = A::Output>,
{
    type Input = A::Input;
    type Output = B::Output;
    
    fn process(&self, input: Self::Input) -> Result<Self::Output> {
        let intermediate = self.first.process(input)?;
        self.second.process(intermediate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_snapshot() {
        let mut metrics = MetricsSnapshot::new();
        metrics.operations = 100;
        metrics.total_time_us = 5000;
        metrics.errors = 5;
        
        assert_eq!(metrics.avg_time_us(), 50.0);
        assert_eq!(metrics.error_rate(), 0.05);
    }
    
    #[test]
    fn test_health_status() {
        assert_eq!(HealthStatus::Healthy, HealthStatus::Healthy);
        assert_ne!(HealthStatus::Healthy, HealthStatus::Degraded);
    }
    
    // Test implementations for new traits
    struct DoubleTransform;
    
    impl Transform<i32, i32> for DoubleTransform {
        fn transform(&self, input: i32) -> i32 {
            input * 2
        }
    }
    
    struct StringifyTransform;
    
    impl Transform<i32, String> for StringifyTransform {
        fn transform(&self, input: i32) -> String {
            input.to_string()
        }
    }
    
    #[test]
    fn test_transform_trait() {
        let doubler = DoubleTransform;
        assert_eq!(doubler.transform(5), 10);
        
        let stringifier = StringifyTransform;
        assert_eq!(stringifier.transform(42), "42");
    }
    
    struct DivideTransform;
    
    impl TryTransform<i32, i32> for DivideTransform {
        fn try_transform(&self, input: i32) -> Result<i32> {
            if input == 0 {
                Err(crate::foundation::error::invalid_input("Cannot divide by zero"))
            } else {
                Ok(100 / input)
            }
        }
    }
    
    #[test]
    fn test_try_transform_trait() {
        let divider = DivideTransform;
        assert_eq!(divider.try_transform(10).unwrap(), 10);
        assert!(divider.try_transform(0).is_err());
    }
    
    struct AddOne;
    struct MultiplyTwo;
    
    impl Process for AddOne {
        type Input = i32;
        type Output = i32;
        
        fn process(&self, input: Self::Input) -> Result<Self::Output> {
            Ok(input + 1)
        }
    }
    
    impl Process for MultiplyTwo {
        type Input = i32;
        type Output = i32;
        
        fn process(&self, input: Self::Input) -> Result<Self::Output> {
            Ok(input * 2)
        }
    }
    
    #[test]
    fn test_composed_process() {
        let add = AddOne;
        let mul = MultiplyTwo;
        let composed = Composed::new(add, mul);
        
        // (5 + 1) * 2 = 12
        assert_eq!(composed.process(5).unwrap(), 12);
    }
    
    struct IntToStringAdapter;
    
    impl Adapter<i32, String> for IntToStringAdapter {
        fn adapt(&self, from: i32) -> Result<String> {
            Ok(from.to_string())
        }
    }
    
    #[test]
    fn test_adapter_trait() {
        let adapter = IntToStringAdapter;
        assert_eq!(adapter.adapt(42).unwrap(), "42");
    }
    
    #[test]
    fn test_typed_trait() {
        struct MyComponent;
        
        impl Typed for MyComponent {}
        
        let component = MyComponent;
        assert!(component.type_name().contains("MyComponent"));
    }
}