//! Plugin manager for loading and managing plugins.
//!
//! This module provides the main interface for plugin management,
//! including loading, unloading, and querying plugins.
//!
//! Note: The plugin manager requires the `std` feature for thread-safe operations.

#![cfg(feature = "std")]

use crate::core::plugin::{Plugin, PluginEntry, PluginState};
use crate::core::traits::Versioned;
use crate::foundation::{
    error::{Error, PluginError, Result, ProcessingError},
    types::{PluginName, Version},
};
use crate::plugins::registry::PluginRegistry;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

/// The main plugin manager.
/// 
/// This manager follows the Repository pattern for managing plugins,
/// providing a centralized access point for all plugin operations.
pub struct PluginManager {
    registry: Arc<RwLock<PluginRegistry>>,
    plugins: Arc<RwLock<HashMap<PluginName, PluginEntry>>>,
}

impl PluginManager {
    /// Creates a new plugin manager.
    pub fn new() -> Self {
        Self {
            registry: Arc::new(RwLock::new(PluginRegistry::new())),
            plugins: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Helper to create a processing error for lock failures
    fn lock_error(component: &'static str) -> Error {
        Error::Processing(ProcessingError::Failed {
            component,
            reason: "Failed to acquire lock".to_string(),
        })
    }
    
    /// Registers a plugin type with the manager.
    pub fn register<P: Plugin + Default + 'static>(&self) -> Result<()> {
        let mut registry = self.registry.write()
            .map_err(|_| Self::lock_error("registry"))?;
        
        registry.register::<P>()
    }
    
    /// Loads a plugin by name.
    pub fn load(&self, name: &str) -> Result<()> {
        let plugin_name = PluginName::from(name);
        
        // Check if already loaded
        {
            let plugins = self.plugins.read()
                .map_err(|_| Self::lock_error("plugins"))?;
            
            if plugins.contains_key(&plugin_name) {
                return Err(Error::Plugin(PluginError::Lifecycle { 
                    name: name.to_string(),
                    current_state: "loaded".to_string(),
                    operation: "load",
                }));
            }
        }
        
        // Create plugin from registry
        let plugin = {
            let registry = self.registry.read()
                .map_err(|_| Self::lock_error("registry"))?;
            registry.create(&plugin_name)?
        };
        
        // Create plugin entry
        let mut entry = PluginEntry::new(plugin);
        
        // Initialize if the plugin supports it
        let capabilities = entry.plugin().capabilities();
        if capabilities.lifecycle {
            // Plugin supports lifecycle, but we need the concrete type
            // to call initialize. This is a limitation of the trait object approach.
            // For now, we'll just transition to ready state.
        }
        
        // Transition to ready state
        entry.transition_to(PluginState::Ready)?;
        
        // Store plugin
        let mut plugins = self.plugins.write()
            .map_err(|_| Self::lock_error("plugins"))?;
        plugins.insert(plugin_name, entry);
        
        Ok(())
    }
    
    /// Unloads a plugin by name.
    pub fn unload(&self, name: &str) -> Result<()> {
        let plugin_name = PluginName::from(name);
        
        let mut plugins = self.plugins.write()
            .map_err(|_| Self::lock_error("plugins"))?;
        
        if let Some(mut entry) = plugins.remove(&plugin_name) {
            // Stop if running
            if entry.state() == PluginState::Running {
                entry.transition_to(PluginState::Stopped)?;
            }
            
            // Call on_unload
            entry.plugin_mut().on_unload()?;
            
            Ok(())
        } else {
            Err(Error::Plugin(PluginError::NotFound { 
                name: name.to_string(),
                available: self.list_loaded(),
            }))
        }
    }
    
    /// Starts a plugin.
    pub fn start(&self, name: &str) -> Result<()> {
        let plugin_name = PluginName::from(name);
        
        let mut plugins = self.plugins.write()
            .map_err(|_| Self::lock_error("plugins"))?;
        
        if let Some(entry) = plugins.get_mut(&plugin_name) {
            if !entry.state().can_start() {
                return Err(Error::Plugin(PluginError::Lifecycle {
                    name: name.to_string(),
                    current_state: format!("{:?}", entry.state()),
                    operation: "start",
                }));
            }
            
            entry.transition_to(PluginState::Running)?;
            Ok(())
        } else {
            Err(Error::Plugin(PluginError::NotFound { 
                name: name.to_string(),
                available: self.list_loaded(),
            }))
        }
    }
    
    /// Stops a plugin.
    pub fn stop(&self, name: &str) -> Result<()> {
        let plugin_name = PluginName::from(name);
        
        let mut plugins = self.plugins.write()
            .map_err(|_| Self::lock_error("plugins"))?;
        
        if let Some(entry) = plugins.get_mut(&plugin_name) {
            if !entry.state().can_stop() {
                return Err(Error::Plugin(PluginError::Lifecycle {
                    name: name.to_string(),
                    current_state: format!("{:?}", entry.state()),
                    operation: "stop",
                }));
            }
            
            entry.transition_to(PluginState::Stopped)?;
            Ok(())
        } else {
            Err(Error::Plugin(PluginError::NotFound { 
                name: name.to_string(),
                available: self.list_loaded(),
            }))
        }
    }
    
    /// Executes a function with access to a plugin.
    pub fn with_plugin<F, R>(&self, name: &str, f: F) -> Result<R>
    where
        F: FnOnce(&dyn Plugin) -> R,
    {
        let plugin_name = PluginName::from(name);
        
        let plugins = self.plugins.read()
            .map_err(|_| Self::lock_error("plugins"))?;
        
        if let Some(entry) = plugins.get(&plugin_name) {
            Ok(f(entry.plugin()))
        } else {
            Err(Error::Plugin(PluginError::NotFound { 
                name: name.to_string(),
                available: self.list_loaded(),
            }))
        }
    }
    
    /// Lists all registered plugins.
    pub fn list_registered(&self) -> Vec<String> {
        self.registry.read()
            .map(|registry| registry.list())
            .unwrap_or_default()
    }
    
    /// Lists all loaded plugins.
    pub fn list_loaded(&self) -> Vec<String> {
        self.plugins.read()
            .map(|plugins| {
                plugins.keys()
                    .map(|name| name.as_str().to_string())
                    .collect()
            })
            .unwrap_or_default()
    }
    
    /// Gets plugin information.
    pub fn info(&self, name: &str) -> Result<PluginInfo> {
        let plugin_name = PluginName::from(name);
        
        let plugins = self.plugins.read()
            .map_err(|_| Self::lock_error("plugins"))?;
        
        if let Some(entry) = plugins.get(&plugin_name) {
            Ok(PluginInfo {
                name: entry.name().to_string(),
                version: entry.version(),
                state: entry.state(),
                capabilities: entry.plugin().capabilities(),
                dependencies: entry.dependencies().to_vec(),
            })
        } else {
            Err(Error::Plugin(PluginError::NotFound { 
                name: name.to_string(),
                available: self.list_loaded(),
            }))
        }
    }
    
    /// Clears all plugins.
    pub fn clear(&self) -> Result<()> {
        let mut plugins = self.plugins.write()
            .map_err(|_| Self::lock_error("plugins"))?;
        
        // Unload all plugins
        for (_, mut entry) in plugins.drain() {
            let _ = entry.plugin_mut().on_unload();
        }
        
        Ok(())
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Plugin information.
#[derive(Debug, Clone)]
pub struct PluginInfo {
    /// Plugin name.
    pub name: String,
    
    /// Plugin version.
    pub version: Version,
    
    /// Current state.
    pub state: PluginState,
    
    /// Plugin capabilities.
    pub capabilities: crate::core::plugin::PluginCapabilities,
    
    /// Dependencies.
    pub dependencies: Vec<PluginName>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::plugin::PluginCapabilities;
    use crate::core::traits::{foundation::Named, identity::Versioned};
    
    #[derive(Debug, Default)]
    struct TestPlugin;
    
    impl Identity for TestPlugin {
        fn id(&self) -> &str {
            "test"
        }
    }
    
    impl crate::core::traits::Versioned for TestPlugin {
        fn version(&self) -> crate::foundation::types::Version {
            crate::foundation::types::Version::new(1, 0, 0)
        }
    }
    
    impl Plugin for TestPlugin {
        fn capabilities(&self) -> PluginCapabilities {
            PluginCapabilities::none()
        }
    }
    
    #[test]
    fn test_plugin_manager() {
        let manager = PluginManager::new();
        
        // Register plugin type
        manager.register::<TestPlugin>().unwrap();
        
        // Load plugin
        manager.load("test").unwrap();
        
        // Check loaded plugins
        let loaded = manager.list_loaded();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0], "test");
        
        // Get plugin info
        let info = manager.info("test").unwrap();
        assert_eq!(info.name, "test");
        assert_eq!(info.version, Version::new(1, 0, 0));
        assert_eq!(info.state, PluginState::Ready);
        
        // Start plugin
        manager.start("test").unwrap();
        let info = manager.info("test").unwrap();
        assert_eq!(info.state, PluginState::Running);
        
        // Stop plugin
        manager.stop("test").unwrap();
        let info = manager.info("test").unwrap();
        assert_eq!(info.state, PluginState::Stopped);
        
        // Unload plugin
        manager.unload("test").unwrap();
        assert!(manager.info("test").is_err());
    }
}