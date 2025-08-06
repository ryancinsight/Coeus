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
//!
//! ## Architecture
//!
//! The plugin system uses a layered approach:
//! 1. Core traits define minimal contracts
//! 2. Extension traits add optional capabilities
//! 3. The plugin manager handles lifecycle

use crate::{
    foundation::{
        error::{Error, PluginError, Result},
        types::{PluginName, Version},
    },
    core::traits::{Identity, Versioned, Lifecycle, HealthCheck, Metrics},
};
use core::fmt::Debug;

// ============================================================================
// Core Plugin Trait (Interface Segregation Principle)
// ============================================================================

/// The base trait that all plugins must implement.
/// 
/// This trait is intentionally minimal, following the Interface
/// Segregation Principle. Additional functionality is provided
/// through extension traits.
pub trait Plugin: Send + Sync + Debug + Identity + Versioned {
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
    /// Whether the plugin supports lifecycle management.
    pub lifecycle: bool,
    
    /// Whether the plugin provides health monitoring.
    pub health: bool,
    
    /// Whether the plugin provides metrics.
    pub metrics: bool,
    
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
            lifecycle: true,
            health: true,
            metrics: true,
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
// Extension Traits (Interface Segregation Principle)
// ============================================================================

/// Lifecycle management for plugins.
pub trait PluginLifecycle: Plugin + Lifecycle {}

/// Health monitoring for plugins.
pub trait PluginHealth: Plugin + HealthCheck {}

/// Metrics collection for plugins.
pub trait PluginMetrics: Plugin + Metrics {}

/// Configuration interface for plugins.
pub trait PluginConfig: Debug + Send + Sync {
    /// Gets a configuration value by key.
    fn get(&self, key: &str) -> Option<&dyn core::any::Any>;
    
    /// Clones the configuration into a boxed trait object.
    fn clone_box(&self) -> Box<dyn PluginConfig>;
    
    /// Validates the configuration.
    fn validate(&self) -> Result<()> {
        Ok(())
    }
}

/// Factory trait for creating plugins (Abstract Factory Pattern).
pub trait PluginFactory: Send + Sync {
    /// The type of plugin this factory creates.
    type Plugin: Plugin;
    
    /// Creates a new plugin instance.
    fn create(&self, config: Option<&dyn PluginConfig>) -> Result<Self::Plugin>;
    
    /// Returns the plugin type identifier.
    fn plugin_type(&self) -> &str;
}

// ============================================================================
// Plugin State Management
// ============================================================================

/// The state of a plugin in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginState {
    /// Plugin is registered but not initialized.
    Registered,
    
    /// Plugin is initialized and ready.
    Ready,
    
    /// Plugin is running.
    Running,
    
    /// Plugin is stopped.
    Stopped,
    
    /// Plugin is in an error state.
    Error,
}

impl PluginState {
    /// Returns whether the plugin is in a usable state.
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Ready | Self::Running)
    }
    
    /// Returns whether the plugin can be started.
    pub fn can_start(&self) -> bool {
        matches!(self, Self::Ready | Self::Stopped)
    }
    
    /// Returns whether the plugin can be stopped.
    pub fn can_stop(&self) -> bool {
        matches!(self, Self::Running)
    }
    
    /// Validates a state transition.
    pub fn can_transition_to(&self, next: Self) -> bool {
        use PluginState::*;
        
        match (*self, next) {
            // Registration transitions
            (Registered, Ready) => true,
            
            // Start transitions
            (Ready, Running) => true,
            (Stopped, Running) => true,
            
            // Stop transitions
            (Running, Stopped) => true,
            
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
    /// The plugin instance.
    plugin: Box<dyn Plugin>,

    /// Current state of the plugin.
    state: PluginState,

    /// Dependencies on other plugins.
    dependencies: Vec<PluginName>,
}

#[cfg(feature = "std")]
impl PluginEntry {
    /// Creates a new plugin entry.
    pub fn new(plugin: Box<dyn Plugin>) -> Self {
        Self {
            plugin,
            state: PluginState::Registered,
            dependencies: Vec::new(),
        }
    }
    
    /// Returns the plugin name.
    pub fn name(&self) -> &str {
        self.plugin.id()
    }
    
    /// Returns the plugin version.
    pub fn version(&self) -> Version {
        self.plugin.version()
    }
    
    /// Returns the current state.
    pub fn state(&self) -> PluginState {
        self.state
    }
    
    /// Transitions to a new state.
    pub fn transition_to(&mut self, new_state: PluginState) -> Result<()> {
        if !self.state.can_transition_to(new_state) {
            return Err(Error::Plugin(PluginError::Lifecycle {
                name: self.name().to_string(),
                current_state: format!("{:?}", self.state),
                operation: "transition",
            }));
        }
        
        self.state = new_state;
        Ok(())
    }
    
    /// Returns a reference to the plugin.
    pub fn plugin(&self) -> &dyn Plugin {
        &*self.plugin
    }
    
    /// Returns a mutable reference to the plugin.
    pub fn plugin_mut(&mut self) -> &mut dyn Plugin {
        &mut *self.plugin
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

// ============================================================================
// Specialized Plugin Traits
// ============================================================================

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
    fn load_model(&self, path: &str) -> Result<Box<dyn crate::core::model::Model<Config = crate::core::model::BasicModelConfig>>>;
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
        impl $crate::core::traits::Identity for $type {
            fn id(&self) -> &str {
                $name
            }
        }
        
        impl $crate::core::traits::Versioned for $type {
            fn version(&self) -> $crate::foundation::types::Version {
                $version
            }
        }
        
        impl $crate::core::plugin::Plugin for $type {
            fn capabilities(&self) -> $crate::core::plugin::PluginCapabilities {
                $crate::core::plugin::PluginCapabilities::standard()
            }
        }
    };
    
    ($type:ty, $name:expr, $version:expr, $capabilities:expr) => {
        impl $crate::core::traits::Identity for $type {
            fn id(&self) -> &str {
                $name
            }
        }
        
        impl $crate::core::traits::Versioned for $type {
            fn version(&self) -> $crate::foundation::types::Version {
                $version
            }
        }
        
        impl $crate::core::plugin::Plugin for $type {
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
    
    impl Identity for TestPlugin {
        fn id(&self) -> &str {
            "test_plugin"
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
        
        fn on_unload(&mut self) -> Result<()> {
            self.initialized = false;
            Ok(())
        }
    }
    
    #[test]
    fn test_plugin_state_transitions() {
        let state = PluginState::Registered;
        assert!(state.can_transition_to(PluginState::Ready));
        assert!(!state.can_transition_to(PluginState::Running));
        
        let state = PluginState::Ready;
        assert!(state.can_transition_to(PluginState::Running));
        assert!(!state.can_transition_to(PluginState::Registered));
    }
    
    #[test]
    fn test_plugin_capabilities() {
        let caps = PluginCapabilities::standard();
        assert!(caps.lifecycle);
        assert!(caps.health);
        assert!(caps.metrics);
        assert!(caps.configurable);
        
        let caps = PluginCapabilities::none()
            .with_feature("custom_feature");
        assert!(caps.has_feature("custom_feature"));
        assert!(!caps.has_feature("other_feature"));
    }
}