//! Plugin manager for loading and managing plugins.
//!
//! This module provides the main interface for plugin management,
//! including loading, unloading, and querying plugins.
//!
//! Note: The plugin manager requires the `std` feature for thread-safe operations.

#![cfg(feature = "std")]

use crate::core::plugin::{Plugin, PluginEntry, PluginState, PluginCapabilities};
use crate::foundation::{
    error::{Error, PluginError, Result, internal_error, ProcessingError},
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
    plugins: Arc<RwLock<HashMap<PluginName, Arc<RwLock<PluginEntry>>>>>,
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
            .map_err(|_| internal_error("Failed to acquire registry lock"))?;
        
        registry.register::<P>()
    }
    
    /// Loads a plugin and returns it as a Plugin trait object.
    /// Loads a plugin and returns it as a shared [`Arc<dyn Plugin>`].
    ///
    /// This will reuse a loaded plugin if one exists and is usable. If not,
    /// it loads, initializes, and registers the plugin, returning a shared reference.
    pub fn load_plugin(&self, name: &str) -> Result<Arc<dyn Plugin>> {
        let plugin_name = PluginName::from(name);

        // Double-checked locking pattern for thread safety
        {
            let plugins = self.plugins.read()
                .map_err(|_| internal_error("Failed to acquire plugins lock"))?;

            if let Some(entry_arc) = plugins.get(&plugin_name) {
                let entry = entry_arc.read()
                    .map_err(|_| internal_error("Failed to acquire plugin entry lock"))?;

                // Check if plugin is in a usable state
                if entry.state().is_usable() {
                    // Return a cloned Arc to the plugin
                    return Ok(entry.plugin_arc().clone());
                }
            }
        }

        // Plugin not loaded, acquire write lock
        let mut plugins = self.plugins.write()
            .map_err(|_| internal_error("Failed to acquire plugins lock"))?;

        // Check again in case another thread loaded it
        if plugins.contains_key(&plugin_name) {
            return Err(Error::Plugin(PluginError::AlreadyLoaded {
                name: name.to_string()
            }));
        }

        // Create plugin from registry
        let registry = self.registry.read()
            .map_err(|_| internal_error("Failed to acquire registry lock"))?;

        let plugin = registry.create(&plugin_name)?;

        // Create plugin entry
        let mut entry = PluginEntry::new(plugin);

        // Transition to initializing state
        entry.transition_to(PluginState::Initializing)?;

        // Initialize if the plugin supports it
        let capabilities = entry.plugin().capabilities();
        if capabilities.initializable && !entry.plugin().is_initialized() {
            entry.plugin_mut().initialize()?;
        }

        // Transition to ready state
        entry.transition_to(PluginState::Ready)?;

        // Wrap in Arc<RwLock> for thread-safe access
        let entry_arc = Arc::new(RwLock::new(entry));

        // Store plugin
        plugins.insert(plugin_name.clone(), entry_arc.clone());

        // Return a reference to the plugin
        let plugin_arc = {
            let entry = entry_arc.read()
                .map_err(|_| internal_error("Failed to acquire plugin entry lock"))?;
            // Clone the Arc<dyn Plugin> from the entry
            entry.plugin_arc().clone()
        };
        Ok(plugin_arc)
    }
    
    /// Unloads a plugin by name.
    pub fn unload(&self, name: &str) -> Result<()> {
        let plugin_name = PluginName::from(name);
        
        let mut plugins = self.plugins.write()
            .map_err(|_| internal_error("Failed to acquire plugins lock"))?;
        
        if let Some(entry_arc) = plugins.remove(&plugin_name) {
            let mut entry = entry_arc.write()
                .map_err(|_| internal_error("Failed to acquire plugin entry lock"))?;

            // Transition through proper states
            if entry.state().can_stop() {
                entry.transition_to(PluginState::Stopping)?;

                // Attempt to call on_unload, but tolerate external Arc refs.
                if let Some(plugin_mut) = entry.try_plugin_mut() {
                    plugin_mut.on_unload()?;
                } else {
                    // Defer clean-up until last Arc drops; on_unload will not be called now.
                    // This is safe: plugin resource clean-up is deferred to last Arc drop.
                }

                entry.transition_to(PluginState::Stopped)?;
            }

            Ok(())
        } else {
            Err(Error::Plugin(PluginError::NotFound {
                name: name.to_string()
            }))
        }
    }
    
    /// Lists all loaded plugins.
    pub fn list_loaded(&self) -> Result<Vec<String>> {
        let plugins = self.plugins.read()
            .map_err(|_| internal_error("Failed to acquire plugins lock"))?;
        
        Ok(plugins.keys().map(|name| name.as_str().to_string()).collect())
    }
    
    /// Lists all available plugins in the registry.
    pub fn list_available(&self) -> Result<Vec<String>> {
        let registry = self.registry.read()
            .map_err(|_| internal_error("Failed to acquire registry lock"))?;
        
        Ok(registry.list())
    }
    
    /// Gets plugin information.
    pub fn info(&self, name: &str) -> Result<PluginInfo> {
        let plugin_name = PluginName::from(name);
        
        let plugins = self.plugins.read()
            .map_err(|_| internal_error("Failed to acquire plugins lock"))?;
        
        if let Some(entry_arc) = plugins.get(&plugin_name) {
            let entry = entry_arc.read()
                .map_err(|_| internal_error("Failed to acquire plugin entry lock"))?;
            
            let plugin = entry.plugin();
            
            Ok(PluginInfo {
                name: plugin_name,
                version: plugin.version(),
                state: entry.state(),
                capabilities: plugin.capabilities(),
            })
        } else {
            Err(Error::Plugin(PluginError::NotFound { 
                name: name.to_string() 
            }))
        }
    }
    
    /// Checks if a plugin is loaded.
    pub fn is_loaded(&self, name: &str) -> bool {
        let plugin_name = PluginName::from(name);
        
        self.plugins.read()
            .map(|plugins| plugins.contains_key(&plugin_name))
            .unwrap_or(false)
    }
    
    /// Executes an operation on a plugin.
    /// 
    /// This is the preferred way to interact with plugins, as it handles
    /// all the locking and state management.
    pub fn with_plugin<F, R>(&self, name: &str, f: F) -> Result<R>
    where
        F: FnOnce(&dyn Plugin) -> Result<R>,
    {
        let plugin_name = PluginName::from(name);
        
        let plugins = self.plugins.read()
            .map_err(|_| internal_error("Failed to acquire plugins lock"))?;
        
        if let Some(entry_arc) = plugins.get(&plugin_name) {
            let entry = entry_arc.read()
                .map_err(|_| internal_error("Failed to acquire plugin entry lock"))?;
            
            // Check if plugin is usable
            if !entry.state().is_usable() {
                return Err(Error::Plugin(PluginError::InvalidState {
                    plugin: name.to_string(),
                    expected: "Ready, Running, or Paused",
                    actual: "Not usable",
                }));
            }
            
            f(entry.plugin())
        } else {
            Err(Error::Plugin(PluginError::NotFound { 
                name: name.to_string() 
            }))
        }
    }
    
    /// Executes a mutable operation on a plugin.
    pub fn with_plugin_mut<F, R>(&self, name: &str, f: F) -> Result<R>
    where
        F: FnOnce(&mut dyn Plugin) -> Result<R>,
    {
        let plugin_name = PluginName::from(name);
        
        let plugins = self.plugins.read()
            .map_err(|_| internal_error("Failed to acquire plugins lock"))?;
        
        if let Some(entry_arc) = plugins.get(&plugin_name) {
            let mut entry = entry_arc.write()
                .map_err(|_| internal_error("Failed to acquire plugin entry lock"))?;
            
            // Check if plugin is usable
            if !entry.state().is_usable() {
                return Err(Error::Plugin(PluginError::InvalidState {
                    plugin: name.to_string(),
                    expected: "Ready, Running, or Paused",
                    actual: "Not usable",
                }));
            }
            
            f(entry.plugin_mut())
        } else {
            Err(Error::Plugin(PluginError::NotFound { 
                name: name.to_string() 
            }))
        }
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

/// Convenience function to execute an operation on a plugin.
pub fn with_plugin<F, R>(name: &str, f: F) -> Result<R>
where
    F: FnOnce(&dyn Plugin) -> Result<R>,
{
    plugin_manager().with_plugin(name, f)
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