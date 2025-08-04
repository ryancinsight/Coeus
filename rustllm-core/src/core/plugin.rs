//! Plugin system for RustLLM Core.
//!
//! This module provides a flexible plugin architecture that follows
//! the Open/Closed Principle - the system is open for extension
//! through plugins but closed for modification.
//!
//! ## Design Principles
//!
//! - **Dependency Inversion**: Core depends on plugin abstractions, not concrete implementations
//! - **Interface Segregation**: Plugins implement only the interfaces they need
//! - **Single Responsibility**: Each plugin has one clear purpose
//! - **Liskov Substitution**: All plugins are interchangeable through traits

use crate::core::traits::{Initialize, Lifecycle, Metadata, Named, Versioned, Described};
use crate::foundation::{
    error::{Error, PluginError, Result},
    types::{PluginName, Version},
};
use core::fmt::Debug;

#[cfg(feature = "std")]
use std::sync::{Arc, RwLock, Weak};

#[cfg(feature = "std")]
use std::collections::HashMap;

// ============================================================================
// Core Plugin Trait
// ============================================================================

/// The base trait that all plugins must implement.
/// 
/// This trait is intentionally minimal, following the Interface
/// Segregation Principle. Additional functionality is provided
/// through extension traits.
pub trait Plugin: Send + Sync + Debug {
    /// Returns the unique name of this plugin.
    fn name(&self) -> &str;
    
    /// Returns the plugin version.
    fn version(&self) -> Version;
    
    /// Returns the plugin's capabilities.
    fn capabilities(&self) -> PluginCapabilities;
    
    /// Called when the plugin is being unloaded.
    /// 
    /// This gives the plugin a chance to clean up resources.
    fn on_unload(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Plugin capabilities descriptor.
/// 
/// This allows plugins to declare their capabilities without
/// implementing unnecessary interfaces.
#[derive(Debug, Clone, Default)]
pub struct PluginCapabilities {
    /// Whether the plugin supports initialization.
    pub initializable: bool,
    
    /// Whether the plugin supports lifecycle management.
    pub lifecycle: bool,
    
    /// Whether the plugin provides metadata.
    pub metadata: bool,
    
    /// Whether the plugin supports configuration.
    pub configurable: bool,
    
    /// Custom capabilities as feature flags.
    pub features: Vec<String>,
}

impl PluginCapabilities {
    /// Creates a new capabilities descriptor with all features disabled.
    pub fn none() -> Self {
        Self::default()
    }
    
    /// Creates a capabilities descriptor with standard features.
    pub fn standard() -> Self {
        Self {
            initializable: true,
            lifecycle: true,
            metadata: true,
            configurable: true,
            features: Vec::new(),
        }
    }
    
    /// Adds a custom feature flag.
    pub fn with_feature(mut self, feature: impl Into<String>) -> Self {
        self.features.push(feature.into());
        self
    }
    
    /// Checks if a feature is supported.
    pub fn has_feature(&self, feature: &str) -> bool {
        self.features.iter().any(|f| f == feature)
    }
}

// ============================================================================
// Plugin State Management
// ============================================================================

/// The state of a plugin in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginState {
    /// Plugin is registered but not initialized.
    Registered,
    
    /// Plugin is being initialized.
    Initializing,
    
    /// Plugin is initialized and ready.
    Ready,
    
    /// Plugin is running.
    Running,
    
    /// Plugin is paused.
    Paused,
    
    /// Plugin is being stopped.
    Stopping,
    
    /// Plugin is stopped.
    Stopped,
    
    /// Plugin is in an error state.
    Error,
}

impl PluginState {
    /// Returns whether the plugin is in a usable state.
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Ready | Self::Running | Self::Paused)
    }
    
    /// Returns whether the plugin can be started.
    pub fn can_start(&self) -> bool {
        matches!(self, Self::Ready | Self::Stopped)
    }
    
    /// Returns whether the plugin can be stopped.
    pub fn can_stop(&self) -> bool {
        matches!(self, Self::Running | Self::Paused)
    }
    
    /// Validates a state transition.
    pub fn can_transition_to(&self, next: Self) -> bool {
        use PluginState::*;
        
        match (*self, next) {
            // Registration transitions
            (Registered, Initializing) => true,
            
            // Initialization transitions
            (Initializing, Ready) => true,
            (Initializing, Error) => true,
            
            // Start transitions
            (Ready, Running) => true,
            (Stopped, Running) => true,
            
            // Pause transitions
            (Running, Paused) => true,
            (Paused, Running) => true,
            
            // Stop transitions
            (Running, Stopping) => true,
            (Paused, Stopping) => true,
            (Stopping, Stopped) => true,
            
            // Error transitions
            (_, Error) => true,
            (Error, Registered) => true, // Allow recovery
            
            // No change
            (state, next) if state == next => true,
            
            // All other transitions are invalid
            _ => false,
        }
    }
}

// ============================================================================
// Plugin Container
// ============================================================================

/// Container for a plugin instance with its metadata.
/// 
/// This struct encapsulates a plugin with its state and metadata,
/// following the principle of encapsulation.
#[cfg(feature = "std")]
pub struct PluginEntry {
    /// The plugin instance, reference counted for shared access.
    plugin: std::sync::Arc<dyn Plugin>,

    /// Current state of the plugin.
    state: PluginState,

    /// Plugin metadata.
    metadata: PluginMetadata,

    /// Dependencies on other plugins.
    dependencies: Vec<PluginName>,

    /// Dependents of this plugin.
    dependents: Vec<Weak<RwLock<PluginEntry>>>,
}

#[cfg(feature = "std")]
impl PluginEntry {
    /// Creates a new plugin entry from a boxed plugin.
    ///
    /// The plugin will be wrapped in an [`Arc`] for shared ownership.
    pub fn new(plugin: Box<dyn Plugin>) -> Self {
        let plugin_arc: std::sync::Arc<dyn Plugin> = plugin.into();
        let metadata = PluginMetadata {
            name: plugin_arc.name().into(),
            version: plugin_arc.version(),
            capabilities: plugin_arc.capabilities(),
        };

        Self {
            plugin: plugin_arc,
            state: PluginState::Registered,
            metadata,
            dependencies: Vec::new(),
            dependents: Vec::new(),
        }
    }

    /// Returns the plugin name.
    pub fn name(&self) -> &PluginName {
        &self.metadata.name
    }

    /// Returns the plugin version.
    pub fn version(&self) -> &Version {
        &self.metadata.version
    }

    /// Returns the current state.
    pub fn state(&self) -> PluginState {
        self.state
    }

    /// Transitions to a new state.
    pub fn transition_to(&mut self, new_state: PluginState) -> Result<()> {
        if !self.state.can_transition_to(new_state) {
            return Err(Error::Plugin(PluginError::InvalidState {
                plugin: self.name().to_string(),
                expected: "valid transition",
                actual: "invalid transition",
            }));
        }

        self.state = new_state;
        Ok(())
    }

    /// Returns a reference to the plugin's shared [`Arc`].
    ///
    /// This allows cheap cloning for thread-safe plugin access.
    pub fn plugin_arc(&self) -> &std::sync::Arc<dyn Plugin> {
        &self.plugin
    }

    /// Returns a reference to the plugin.
    pub fn plugin(&self) -> &dyn Plugin {
        self.plugin.as_ref()
    }

    /// Returns a mutable reference to the plugin.
    ///
    /// # Panics
    /// Panics if multiple references to the plugin exist.
    pub fn plugin_mut(&mut self) -> &mut dyn Plugin {
        std::sync::Arc::get_mut(&mut self.plugin)
            .expect("multiple refs to plugin")
            .as_mut()
    }

    /// Adds a dependency.
    pub fn add_dependency(&mut self, dep: PluginName) {
        if !self.dependencies.contains(&dep) {
            self.dependencies.push(dep);
        }
    }

    /// Returns the dependencies.
    pub fn dependencies(&self) -> &[PluginName] {
        &self.dependencies
    }

    /// Adds a dependent.
    pub fn add_dependent(&mut self, dependent: Weak<RwLock<PluginEntry>>) {
        self.dependents.push(dependent);
    }

    /// Removes stale dependents.
    pub fn clean_dependents(&mut self) {
        self.dependents.retain(|weak| weak.strong_count() > 0);
    }
}

/// Plugin metadata.
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    /// Plugin name.
    pub name: PluginName,
    
    /// Plugin version.
    pub version: Version,
    
    /// Plugin capabilities.
    pub capabilities: PluginCapabilities,
}

// ============================================================================
// Plugin Builders
// ============================================================================

/// Builder for creating plugins with a fluent API.
/// 
/// This follows the Builder pattern for complex object construction.
pub struct PluginBuilder<P> {
    plugin: P,
    dependencies: Vec<PluginName>,
}

impl<P: Plugin + 'static> PluginBuilder<P> {
    /// Creates a new plugin builder.
    pub fn new(plugin: P) -> Self {
        Self {
            plugin,
            dependencies: Vec::new(),
        }
    }
    
    /// Adds a dependency.
    pub fn depends_on(mut self, plugin: impl Into<PluginName>) -> Self {
        self.dependencies.push(plugin.into());
        self
    }
    
    /// Builds the plugin entry.
    #[cfg(feature = "std")]
    pub fn build(self) -> PluginEntry {
        let mut entry = PluginEntry::new(Box::new(self.plugin));
        for dep in self.dependencies {
            entry.add_dependency(dep);
        }
        entry
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

// ============================================================================
// Helper Macros
// ============================================================================

/// Helper macro for implementing the Plugin trait.
/// 
/// This macro provides a convenient way to implement the Plugin trait
/// with sensible defaults.
#[macro_export]
macro_rules! impl_plugin {
    ($type:ty, $name:expr, $version:expr) => {
        impl $crate::core::plugin::Plugin for $type {
            fn name(&self) -> &str {
                $name
            }
            
            fn version(&self) -> $crate::foundation::types::Version {
                $version
            }
            
            fn capabilities(&self) -> $crate::core::plugin::PluginCapabilities {
                $crate::core::plugin::PluginCapabilities::standard()
            }
        }
    };
    
    ($type:ty, $name:expr, $version:expr, $capabilities:expr) => {
        impl $crate::core::plugin::Plugin for $type {
            fn name(&self) -> &str {
                $name
            }
            
            fn version(&self) -> $crate::foundation::types::Version {
                $version
            }
            
            fn capabilities(&self) -> $crate::core::plugin::PluginCapabilities {
                $capabilities
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::types::Version;
    
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
        
        fn capabilities(&self) -> PluginCapabilities {
            PluginCapabilities::none()
        }
        
        fn on_unload(&mut self) -> Result<()> {
            self.initialized = false;
            Ok(())
        }
    }
    
    #[test]
    fn test_plugin_state_transitions() {
        let state = PluginState::Registered;
        assert!(state.can_transition_to(PluginState::Initializing));
        assert!(!state.can_transition_to(PluginState::Running));
        
        let state = PluginState::Ready;
        assert!(state.can_transition_to(PluginState::Running));
        assert!(!state.can_transition_to(PluginState::Initializing));
    }
    
    #[test]
    fn test_plugin_capabilities() {
        let caps = PluginCapabilities::standard();
        assert!(caps.initializable);
        assert!(caps.lifecycle);
        assert!(caps.metadata);
        assert!(caps.configurable);
        
        let caps = PluginCapabilities::none()
            .with_feature("custom_feature");
        assert!(caps.has_feature("custom_feature"));
        assert!(!caps.has_feature("other_feature"));
    }
}