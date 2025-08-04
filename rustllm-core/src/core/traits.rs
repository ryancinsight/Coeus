//! Core traits defining the fundamental abstractions of RustLLM Core.
//!
//! This module contains the essential traits that all components must implement,
//! following the Interface Segregation Principle (ISP) from SOLID.

use crate::foundation::error::Result;
use core::fmt::Debug;

/// Trait for components that can be initialized.
pub trait Initialize {
    /// Initializes the component.
    fn initialize(&mut self) -> Result<()>;
    
    /// Checks if the component is initialized.
    fn is_initialized(&self) -> bool;
}

/// Trait for components that can be reset to their initial state.
pub trait Reset {
    /// Resets the component to its initial state.
    fn reset(&mut self) -> Result<()>;
}

/// Trait for components that can be validated.
pub trait Validate {
    /// Validates the component's state.
    fn validate(&self) -> Result<()>;
}

/// Trait for components that can be serialized.
pub trait Serialize {
    /// Serializes the component to bytes.
    fn serialize(&self) -> Result<Vec<u8>>;
    
    /// Deserializes the component from bytes.
    fn deserialize(data: &[u8]) -> Result<Self>
    where
        Self: Sized;
}

/// Trait for components with configuration.
pub trait Configurable {
    /// The configuration type.
    type Config: Debug + Clone;
    
    /// Applies configuration to the component.
    fn configure(&mut self, config: Self::Config) -> Result<()>;
    
    /// Returns the current configuration.
    fn config(&self) -> &Self::Config;
}

/// Trait for components that can be cloned with a custom implementation.
pub trait CloneBox: Send + Sync {
    /// Clones the component into a boxed trait object.
    fn clone_box(&self) -> Box<dyn CloneBox>;
}

/// Trait for components with metadata.
pub trait Metadata {
    /// Returns the component's name.
    fn name(&self) -> &str;
    
    /// Returns the component's description.
    fn description(&self) -> &str;
    
    /// Returns the component's version.
    fn version(&self) -> crate::foundation::types::Version;
}

/// Trait for components that can process data in batches.
pub trait BatchProcessor: Send + Sync {
    /// The input type.
    type Input;
    
    /// The output type.
    type Output;
    
    /// Processes a batch of inputs.
    fn process_batch(&self, inputs: &[Self::Input]) -> Result<Vec<Self::Output>>;
    
    /// Returns the optimal batch size for processing.
    fn optimal_batch_size(&self) -> usize {
        32
    }
}

/// Trait for components that can process data in a streaming fashion.
pub trait StreamProcessor: Send + Sync {
    /// The input type.
    type Input;
    
    /// The output type.
    type Output;
    
    /// Processes a single input.
    fn process(&self, input: Self::Input) -> Result<Self::Output>;
    
    /// Flushes any buffered data.
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Trait for components that support async operations.
#[cfg(feature = "async")]
pub trait AsyncProcessor: Send + Sync {
    /// The input type.
    type Input: Send;
    
    /// The output type.
    type Output: Send;
    
    /// Processes input asynchronously.
    async fn process_async(&self, input: Self::Input) -> Result<Self::Output>;
}

/// Trait for components that can be monitored.
pub trait Monitor {
    /// Returns performance metrics.
    fn metrics(&self) -> Metrics;
    
    /// Returns health status.
    fn health(&self) -> Health;
}

/// Performance metrics for monitoring.
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    /// Total number of operations.
    pub operations: u64,
    
    /// Total processing time in microseconds.
    pub total_time_us: u64,
    
    /// Number of errors.
    pub errors: u64,
    
    /// Current memory usage in bytes.
    pub memory_bytes: usize,
}

impl Metrics {
    /// Creates new empty metrics.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Returns the average processing time per operation.
    pub fn avg_time_us(&self) -> f64 {
        if self.operations == 0 {
            0.0
        } else {
            self.total_time_us as f64 / self.operations as f64
        }
    }
    
    /// Returns the error rate.
    pub fn error_rate(&self) -> f64 {
        if self.operations == 0 {
            0.0
        } else {
            self.errors as f64 / self.operations as f64
        }
    }
}

/// Health status for monitoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Health {
    /// Component is healthy.
    Healthy,
    
    /// Component is degraded but functional.
    Degraded,
    
    /// Component is unhealthy.
    Unhealthy,
}

/// Trait for components that support caching.
pub trait Cacheable {
    /// The key type for cache lookups.
    type Key: Eq + core::hash::Hash;
    
    /// The value type stored in cache.
    type Value: Clone;
    
    /// Looks up a value in the cache.
    fn cache_get(&self, key: &Self::Key) -> Option<Self::Value>;
    
    /// Stores a value in the cache.
    fn cache_put(&mut self, key: Self::Key, value: Self::Value);
    
    /// Clears the cache.
    fn cache_clear(&mut self);
}

/// Trait for components that support lifecycle management.
pub trait Lifecycle: Initialize + Reset + Send + Sync {
    /// Starts the component.
    fn start(&mut self) -> Result<()> {
        self.initialize()
    }
    
    /// Stops the component.
    fn stop(&mut self) -> Result<()> {
        self.reset()
    }
    
    /// Pauses the component.
    fn pause(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// Resumes the component.
    fn resume(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Trait for components that can be composed.
pub trait Composable: Send + Sync {
    /// The output type of this component.
    type Output;
    
    /// Composes this component with another.
    fn compose<T>(self, other: T) -> ComposedComponent<Self, T>
    where
        Self: Sized,
        T: Composable,
    {
        ComposedComponent::new(self, other)
    }
}

/// A composed component.
pub struct ComposedComponent<A, B> {
    first: A,
    second: B,
}

impl<A, B> ComposedComponent<A, B> {
    /// Creates a new composed component.
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
    
    /// Returns a reference to the first component.
    pub fn first(&self) -> &A {
        &self.first
    }
    
    /// Returns a reference to the second component.
    pub fn second(&self) -> &B {
        &self.second
    }
    
    /// Returns mutable references to both components.
    pub fn components_mut(&mut self) -> (&mut A, &mut B) {
        (&mut self.first, &mut self.second)
    }
}

impl<A, B> Composable for ComposedComponent<A, B>
where
    A: Composable,
    B: Composable,
{
    type Output = B::Output;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics() {
        let mut metrics = Metrics::new();
        metrics.operations = 100;
        metrics.total_time_us = 5000;
        metrics.errors = 5;
        
        assert_eq!(metrics.avg_time_us(), 50.0);
        assert_eq!(metrics.error_rate(), 0.05);
    }
    
    #[test]
    fn test_health() {
        assert_eq!(Health::Healthy, Health::Healthy);
        assert_ne!(Health::Healthy, Health::Degraded);
    }
}