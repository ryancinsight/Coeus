//! Core traits defining the fundamental abstractions of RustLLM Core.
//!
//! This module contains the essential traits that all components must implement,
//! following the Interface Segregation Principle (ISP) and Single Responsibility
//! Principle (SRP) from SOLID.

use crate::foundation::error::Result;
use core::fmt::Debug;

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
    fn flush_state(&self, state: &mut Self::State) -> Result<Option<Self::Output>> {
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
/// Separated into minimal required metadata following ISP.
pub trait Named {
    /// Returns the component's name.
    fn name(&self) -> &str;
}

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

/// Combined metadata trait for convenience.
/// 
/// This follows ISP by allowing components to implement only
/// the metadata traits they need.
pub trait Metadata: Named + Versioned + Described {}

// Blanket implementation
impl<T> Metadata for T where T: Named + Versioned + Described {}

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
}