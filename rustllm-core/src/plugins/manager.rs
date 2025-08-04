//! Plugin manager for loading and managing plugins.
//!
//! This module provides the main interface for plugin management,
//! including loading, unloading, and querying plugins.
//!
//! Note: The plugin manager requires the `std` feature for thread-safe operations.

#![cfg(feature = "std")]

use crate::core::plugin::{Plugin, PluginEntry, PluginState, PluginMetadata, PluginCapabilities};
use crate::foundation::{
    error::{Error, PluginError, Result},
    types::{PluginName, Version},
};
use crate::plugins::registry::PluginRegistry;
use core::any::Any;
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

/// The main plugin manager.
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
    
    /// Registers a plugin type with the manager.
    pub fn register<P: Plugin + Default + 'static>(&self) -> Result<()> {
        let mut registry = self.registry.write()
            .map_err(|_| Error::Other("Failed to acquire registry lock".to_string()))?;
        
        registry.register::<P>()
    }
    
    /// Loads a plugin by name.
    /// 
    /// Note: This method is deprecated. Use `load_plugin` instead for loading plugins.
    /// Direct type conversion is not supported due to the dynamic nature of plugins.
    pub fn load<T: 'static>(&self, _name: &str) -> Result<Arc<T>>
    where
        T: Any,
    {
        // This method cannot work as intended because we can't convert
        // Arc<dyn Plugin> to Arc<T> for arbitrary T
        Err(Error::Other(
            "Direct type loading is not supported. Use load_plugin() and then downcast or use specific plugin traits.".to_string()
        ))
    }
    
    /// Loads a plugin and returns it as a Plugin trait object.
    pub fn load_plugin(&self, name: &str) -> Result<Arc<dyn Plugin>> {
        let plugin_name = PluginName::from(name);
        
        // Double-checked locking pattern
        // First check with read lock
        {
            let plugins = self.plugins.read()
                .map_err(|_| Error::Other("Failed to acquire plugins lock".to_string()))?;
            
            if let Some(entry) = plugins.get(&plugin_name) {
                return Ok(entry.plugin.clone());
            }
        }
        
        // Plugin not loaded, acquire write lock
        let mut plugins = self.plugins.write()
            .map_err(|_| Error::Other("Failed to acquire plugins lock".to_string()))?;
        
        // Check again in case another thread loaded it
        if let Some(entry) = plugins.get(&plugin_name) {
            return Ok(entry.plugin.clone());
        }
        
        // Create plugin from registry
        let registry = self.registry.read()
            .map_err(|_| Error::Other("Failed to acquire registry lock".to_string()))?;
        
        let mut plugin = registry.create(&plugin_name)?;
        
        // Initialize plugin
        plugin.initialize()?;
        
        // Get metadata
        let metadata = PluginMetadata {
            name: plugin_name.clone(),
            version: plugin.version(),
            author: String::new(),
            description: plugin.description().to_string(),
            license: String::new(),
            homepage: None,
            repository: None,
        };
        
        let capabilities = PluginCapabilities::default();
        
        // Wrap in Arc
        let plugin_arc: Arc<dyn Plugin> = Arc::from(plugin);
        
        let entry = PluginEntry {
            plugin: plugin_arc.clone(),
            metadata,
            capabilities,
            state: PluginState::Ready,
        };
        
        // Store plugin
        plugins.insert(plugin_name, entry);
        
        Ok(plugin_arc)
    }
    
    /// Unloads a plugin by name.
    pub fn unload(&self, name: &str) -> Result<()> {
        let plugin_name = PluginName::from(name);
        
        let mut plugins = self.plugins.write()
            .map_err(|_| Error::Other("Failed to acquire plugins lock".to_string()))?;
        
        if let Some(mut entry) = plugins.remove(&plugin_name) {
            entry.state = PluginState::ShuttingDown;
            // We can't call shutdown on Arc<dyn Plugin> because it requires &mut self
            // This is a design limitation - we need to rethink plugin lifecycle
            entry.state = PluginState::Shutdown;
            Ok(())
        } else {
            Err(Error::Plugin(PluginError::NotFound(name.to_string())))
        }
    }
    
    /// Lists all loaded plugins.
    pub fn list_loaded(&self) -> Result<Vec<String>> {
        let plugins = self.plugins.read()
            .map_err(|_| Error::Other("Failed to acquire plugins lock".to_string()))?;
        
        Ok(plugins.keys().map(|name| name.as_str().to_string()).collect())
    }
    
    /// Lists all available plugins in the registry.
    pub fn list_available(&self) -> Result<Vec<String>> {
        let registry = self.registry.read()
            .map_err(|_| Error::Other("Failed to acquire registry lock".to_string()))?;
        
        Ok(registry.list())
    }
    
    /// Gets plugin information.
    pub fn info(&self, name: &str) -> Result<PluginInfo> {
        let plugin_name = PluginName::from(name);
        
        let plugins = self.plugins.read()
            .map_err(|_| Error::Other("Failed to acquire plugins lock".to_string()))?;
        
        if let Some(entry) = plugins.get(&plugin_name) {
            Ok(PluginInfo {
                name: entry.metadata.name.clone(),
                version: entry.metadata.version.clone(),
                author: entry.metadata.author.clone(),
                description: entry.metadata.description.clone(),
                state: entry.state,
                capabilities: entry.capabilities.clone(),
            })
        } else {
            Err(Error::Plugin(PluginError::NotFound(name.to_string())))
        }
    }
    
    /// Checks if a plugin is loaded.
    pub fn is_loaded(&self, name: &str) -> bool {
        let plugin_name = PluginName::from(name);
        
        self.plugins.read()
            .map(|plugins| plugins.contains_key(&plugin_name))
            .unwrap_or(false)
    }
    
    /// Reloads a plugin.
    pub fn reload(&self, name: &str) -> Result<()> {
        self.unload(name)?;
        // In a real implementation, we would re-scan for the plugin
        // For now, just return an error
        Err(Error::NotSupported("Plugin reloading not implemented".to_string()))
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
    pub name: PluginName,
    
    /// Plugin version.
    pub version: Version,
    
    /// Plugin author.
    pub author: String,
    
    /// Plugin description.
    pub description: String,
    
    /// Plugin state.
    pub state: PluginState,
    
    /// Plugin capabilities.
    pub capabilities: PluginCapabilities,
}

/// Global plugin manager instance.
static PLUGIN_MANAGER: std::sync::OnceLock<PluginManager> = std::sync::OnceLock::new();

/// Gets the global plugin manager instance.
pub fn plugin_manager() -> &'static PluginManager {
    PLUGIN_MANAGER.get_or_init(PluginManager::new)
}

/// Convenience function to load a plugin.
pub fn plugin_load(name: &str) -> Result<Arc<dyn Plugin>> {
    plugin_manager().load_plugin(name)
}

/// Convenience function to unload a plugin.
pub fn unload_plugin(name: &str) -> Result<()> {
    plugin_manager().unload(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_plugin_manager_creation() {
        let manager = PluginManager::new();
        assert_eq!(manager.list_loaded().unwrap().len(), 0);
    }
    
    #[test]
    fn test_plugin_manager_global() {
        let manager = plugin_manager();
        assert_eq!(manager.list_loaded().unwrap().len(), 0);
    }
}