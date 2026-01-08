//! Research Agent Registry
//!
//! This module provides a factory registry for creating and managing research agents.
//! It allows dynamic registration and instantiation of different agent types.

use std::collections::HashMap;
use std::sync::RwLock;

use crate::core::error::{NNError, Result};

use super::{ResearchAgent, ResearchAgentFactory};

/// Thread-safe registry for research agents
#[derive(Default, Debug)]
pub struct ResearchAgentRegistry {
    /// Registered agent factories
    factories: RwLock<HashMap<String, Box<dyn ResearchAgentFactory>>>,
}

impl ResearchAgentRegistry {
    /// Create new agent registry
    pub fn new() -> Self {
        Self {
            factories: RwLock::new(HashMap::new()),
        }
    }

    /// Register an agent factory
    pub fn register<F: ResearchAgentFactory + 'static>(&self, name: &str) -> Result<()> {
        let factory = F::create_factory();
        self.factories
            .write()
            .unwrap()
            .insert(name.to_string(), factory);
        Ok(())
    }

    /// Create an agent instance
    pub fn create_agent(
        &self,
        name: &str,
        config: serde_json::Value,
    ) -> Result<Box<dyn ResearchAgent>> {
        let factories = self.factories.read().unwrap();
        let factory = factories
            .get(name)
            .ok_or_else(|| NNError::InvalidConfiguration {
                message: format!("Agent '{}' not registered", name),
            })?;

        factory.create(config)
    }

    /// List all registered agent types
    pub fn list_agents(&self) -> Vec<String> {
        self.factories.read().unwrap().keys().cloned().collect()
    }

    /// Check if agent type is registered
    pub fn has_agent(&self, name: &str) -> bool {
        self.factories.read().unwrap().contains_key(name)
    }
}
