//! Plugin system traits and types.
//!
//! This module defines the plugin architecture for extending RustLLM Core
//! with custom tokenizers, models, and loaders.

use crate::foundation::{
    error::Result,
    types::{PluginName, Version},
};
use core::any::Any;
use core::fmt::Debug;

#[cfg(feature = "std")]
use std::sync::Arc;

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;

/// Main plugin trait that all plugins must implement.
pub trait Plugin: Send + Sync + Any + Debug {
    /// Returns the plugin name.
    fn name(&self) -> &str;
    
    /// Returns the plugin version.
    fn version(&self) -> Version;
    
    /// Returns the plugin description.
    fn description(&self) -> &str {
        "No description provided"
    }
    
    /// Initializes the plugin.
    fn initialize(&mut self) -> Result<()>;
    
    /// Shuts down the plugin.
    fn shutdown(&mut self) -> Result<()>;
    
    /// Returns the plugin's dependencies.
    fn dependencies(&self) -> Vec<PluginDependency> {
        Vec::new()
    }
    
    /// Casts the plugin to Any for downcasting.
    fn as_any(&self) -> &dyn Any;
    
    /// Casts the plugin to mutable Any for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Represents a plugin dependency.
#[derive(Debug, Clone)]
pub struct PluginDependency {
    /// The name of the required plugin.
    pub name: PluginName,
    
    /// The minimum required version.
    pub min_version: Version,
    
    /// Whether the dependency is optional.
    pub optional: bool,
}

impl PluginDependency {
    /// Creates a new required dependency.
    pub fn required(name: impl Into<PluginName>, min_version: Version) -> Self {
        Self {
            name: name.into(),
            min_version,
            optional: false,
        }
    }
    
    /// Creates a new optional dependency.
    pub fn optional(name: impl Into<PluginName>, min_version: Version) -> Self {
        Self {
            name: name.into(),
            min_version,
            optional: true,
        }
    }
}

/// Trait for plugins that provide tokenization functionality.
pub trait TokenizerPlugin: Plugin {
    /// The tokenizer type provided by this plugin.
    type Tokenizer: crate::core::tokenizer::Tokenizer;
    
    /// Creates a new tokenizer instance.
    fn create_tokenizer(&self) -> Result<Self::Tokenizer>;
}

/// Trait for plugins that provide model building functionality.
pub trait ModelBuilderPlugin: Plugin {
    /// The model builder type provided by this plugin.
    type Builder: crate::core::model::ModelBuilder;
    
    /// Creates a new model builder instance.
    fn create_builder(&self) -> Result<Self::Builder>;
}

/// Trait for plugins that provide model loading functionality.
pub trait ModelLoaderPlugin: Plugin {
    /// Supported file formats.
    fn supported_formats(&self) -> Vec<&'static str>;
    
    /// Loads a model from the given path.
    fn load_model(&self, path: &str) -> Result<Box<dyn crate::core::model::Model<
        Input = Vec<f32>,
        Output = Vec<f32>,
        Config = crate::core::model::BasicModelConfig,
    >>>;
}

/// Plugin lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginState {
    /// Plugin is created but not initialized.
    Created,
    
    /// Plugin is initializing.
    Initializing,
    
    /// Plugin is initialized and ready.
    Ready,
    
    /// Plugin is shutting down.
    ShuttingDown,
    
    /// Plugin is shut down.
    Shutdown,
    
    /// Plugin encountered an error.
    Error,
}

/// Plugin metadata.
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    /// Plugin name.
    pub name: PluginName,
    
    /// Plugin version.
    pub version: Version,
    
    /// Plugin author.
    pub author: String,
    
    /// Plugin description.
    pub description: String,
    
    /// Plugin license.
    pub license: String,
    
    /// Plugin homepage.
    pub homepage: Option<String>,
    
    /// Plugin repository.
    pub repository: Option<String>,
}

/// Plugin capabilities.
#[derive(Debug, Clone)]
pub struct PluginCapabilities {
    /// Whether the plugin provides tokenization.
    pub tokenization: bool,
    
    /// Whether the plugin provides model building.
    pub model_building: bool,
    
    /// Whether the plugin provides model loading.
    pub model_loading: bool,
    
    /// Custom capabilities.
    pub custom: Vec<String>,
}

impl Default for PluginCapabilities {
    fn default() -> Self {
        Self {
            tokenization: false,
            model_building: false,
            model_loading: false,
            custom: Vec::new(),
        }
    }
}

/// Trait for plugins that support configuration.
pub trait ConfigurablePlugin: Plugin {
    /// Configuration type for the plugin.
    type Config: Debug + Clone + Send + Sync;
    
    /// Configures the plugin.
    fn configure(&mut self, config: Self::Config) -> Result<()>;
    
    /// Returns the current configuration.
    fn config(&self) -> &Self::Config;
}

/// Trait for plugins that support hot reloading.
pub trait HotReloadablePlugin: Plugin {
    /// Prepares the plugin for reloading.
    fn prepare_reload(&mut self) -> Result<()>;
    
    /// Completes the reload process.
    fn complete_reload(&mut self) -> Result<()>;
    
    /// Checks if the plugin can be safely reloaded.
    fn can_reload(&self) -> bool {
        true
    }
}

/// Plugin registry entry.
#[derive(Debug)]
pub struct PluginEntry {
    /// The plugin instance.
    pub plugin: Arc<dyn Plugin>,
    
    /// Plugin metadata.
    pub metadata: PluginMetadata,
    
    /// Plugin capabilities.
    pub capabilities: PluginCapabilities,
    
    /// Plugin state.
    pub state: PluginState,
}

/// Helper macro for implementing the Plugin trait.
#[macro_export]
macro_rules! impl_plugin {
    ($type:ty, $name:expr, $version:expr) => {
        impl Plugin for $type {
            fn name(&self) -> &str {
                $name
            }
            
            fn version(&self) -> $crate::foundation::types::Version {
                $version
            }
            
            fn initialize(&mut self) -> $crate::foundation::error::Result<()> {
                Ok(())
            }
            
            fn shutdown(&mut self) -> $crate::foundation::error::Result<()> {
                Ok(())
            }
            
            fn as_any(&self) -> &dyn core::any::Any {
                self
            }
            
            fn as_any_mut(&mut self) -> &mut dyn core::any::Any {
                self
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[derive(Debug)]
    struct TestPlugin {
        initialized: bool,
    }
    
    impl Plugin for TestPlugin {
        fn name(&self) -> &str {
            "test_plugin"
        }
        
        fn version(&self) -> Version {
            Version::new(1, 0, 0)
        }
        
        fn initialize(&mut self) -> Result<()> {
            self.initialized = true;
            Ok(())
        }
        
        fn shutdown(&mut self) -> Result<()> {
            self.initialized = false;
            Ok(())
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
        
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }
    
    #[test]
    fn test_plugin_lifecycle() {
        let mut plugin = TestPlugin { initialized: false };
        
        assert_eq!(plugin.name(), "test_plugin");
        assert_eq!(plugin.version(), Version::new(1, 0, 0));
        assert!(!plugin.initialized);
        
        plugin.initialize().unwrap();
        assert!(plugin.initialized);
        
        plugin.shutdown().unwrap();
        assert!(!plugin.initialized);
    }
    
    #[test]
    fn test_plugin_dependency() {
        let dep = PluginDependency::required("tokenizer", Version::new(1, 0, 0));
        assert_eq!(dep.name.as_str(), "tokenizer");
        assert_eq!(dep.min_version, Version::new(1, 0, 0));
        assert!(!dep.optional);
        
        let opt_dep = PluginDependency::optional("model", Version::new(2, 0, 0));
        assert!(opt_dep.optional);
    }
}