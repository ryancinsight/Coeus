//! Plugin registry for storing and creating plugins.
//!
//! This module provides the registry that stores plugin factories
//! and creates plugin instances on demand.
//!
//! Note: The plugin registry requires the `std` feature.

#![cfg(feature = "std")]

use crate::core::plugin::Plugin;

use crate::foundation::{
    error::{Error, PluginError, Result},
    types::PluginName,
};
use std::any::TypeId;
use std::collections::HashMap;

/// Plugin factory function type.
type PluginFactory = Box<dyn Fn() -> Result<Box<dyn Plugin>> + Send + Sync>;

/// Plugin registry that stores plugin factories.
pub struct PluginRegistry {
    factories: HashMap<PluginName, PluginFactory>,
    type_map: HashMap<TypeId, PluginName>,
    deps_map: HashMap<PluginName, Vec<PluginName>>, // new: track dependencies
}

impl PluginRegistry {
    /// Creates a new empty registry.
    pub fn new() -> Self {
        Self {
            factories: HashMap::new(),
            type_map: HashMap::new(),
            deps_map: HashMap::new(),
        }
    }

    /// Registers a plugin type.
    pub fn register<P>(&mut self) -> Result<()>
    where
        P: Plugin + Default + 'static,
    {
        let plugin = P::default();
        let name = PluginName::from(plugin.id());
        let type_id = TypeId::of::<P>();

        // Check if already registered
        if self.factories.contains_key(&name) {
            return Err(Error::Plugin(PluginError::Lifecycle {
                name: name.as_str().to_string(),
                current_state: "Ready".to_string(),
                operation: "register",
            }));
        }

        // Create factory
        let factory: PluginFactory = Box::new(|| {
            let plugin = P::default();
            Ok(Box::new(plugin) as Box<dyn Plugin>)
        });

        self.factories.insert(name.clone(), factory);
        self.type_map.insert(type_id, name);

        Ok(())
    }

    /// Registers a plugin type with declared dependencies.
    pub fn register_with_deps<P>(&mut self, deps: &[PluginName]) -> Result<()>
    where
        P: Plugin + Default + 'static,
    {
        let plugin = P::default();
        let name = PluginName::from(plugin.id());
        self.register::<P>()?;
        self.deps_map.insert(name, deps.to_vec());
        Ok(())
    }

    /// Registers a plugin with a custom factory.
    pub fn register_with_factory<F>(
        &mut self,
        name: impl Into<PluginName>,
        factory: F,
    ) -> Result<()>
    where
        F: Fn() -> Result<Box<dyn Plugin>> + Send + Sync + 'static,
    {
        let name = name.into();

        // Check if already registered
        if self.factories.contains_key(&name) {
            return Err(Error::Plugin(PluginError::Lifecycle {
                name: name.as_str().to_string(),
                current_state: "Ready".to_string(),
                operation: "register",
            }));
        }

        self.factories.insert(name, Box::new(factory));

        Ok(())
    }

    /// Registers a plugin with a factory and declared dependencies.
    pub fn register_factory_with_deps<F>(
        &mut self,
        name: impl Into<PluginName>,
        factory: F,
        deps: &[PluginName],
    ) -> Result<()>
    where
        F: Fn() -> Result<Box<dyn Plugin>> + Send + Sync + 'static,
    {
        let name = name.into();
        self.register_with_factory(name.clone(), factory)?;
        self.deps_map.insert(name, deps.to_vec());
        Ok(())
    }

    /// Creates a plugin instance by name.
    pub fn create(&self, name: &PluginName) -> Result<Box<dyn Plugin>> {
        self.factories
            .get(name)
            .ok_or_else(|| {
                Error::Plugin(PluginError::NotFound {
                    name: name.as_str().to_string(),
                    available: self.list(),
                })
            })
            .and_then(|factory| factory())
    }

    /// Lists all registered plugins.
    pub fn list(&self) -> Vec<String> {
        self.factories
            .keys()
            .map(|name| name.as_str().to_string())
            .collect()
    }

    /// Checks if a plugin is registered.
    pub fn contains(&self, name: &PluginName) -> bool {
        self.factories.contains_key(name)
    }

    /// Gets the plugin name for a type.
    pub fn name_for_type<T: 'static>(&self) -> Option<&PluginName> {
        let type_id = TypeId::of::<T>();
        self.type_map.get(&type_id)
    }

    /// Returns declared dependencies for a plugin, if any.
    pub fn dependencies(&self, name: &PluginName) -> &[PluginName] {
        self.deps_map
            .get(name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Unregisters a plugin.
    pub fn unregister(&mut self, name: &PluginName) -> Result<()> {
        if self.factories.remove(name).is_some() {
            // Remove from type map
            self.type_map.retain(|_, v| v != name);
            self.deps_map.remove(name);
            Ok(())
        } else {
            Err(Error::Plugin(PluginError::NotFound {
                name: name.as_str().to_string(),
                available: self.list(),
            }))
        }
    }

    /// Clears all registered plugins.
    pub fn clear(&mut self) {
        self.factories.clear();
        self.type_map.clear();
        self.deps_map.clear();
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for plugin registry with common plugins pre-registered.
pub struct PluginRegistryBuilder {
    registry: PluginRegistry,
}

impl PluginRegistryBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            registry: PluginRegistry::new(),
        }
    }

    /// Registers a plugin type.
    pub fn with_plugin<P>(mut self) -> Self
    where
        P: Plugin + Default + 'static,
    {
        let _ = self.registry.register::<P>();
        self
    }

    /// Registers a plugin with a custom factory.
    pub fn with_factory<F>(mut self, name: impl Into<PluginName>, factory: F) -> Self
    where
        F: Fn() -> Result<Box<dyn Plugin>> + Send + Sync + 'static,
    {
        let _ = self.registry.register_with_factory(name, factory);
        self
    }

    /// Builds the registry.
    pub fn build(self) -> PluginRegistry {
        self.registry
    }
}

impl Default for PluginRegistryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::types::Version;
    use crate::prelude::Identity;

    #[derive(Debug, Default)]
    struct TestPlugin {
        value: i32,
    }

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
        fn capabilities(&self) -> crate::core::plugin::PluginCapabilities {
            crate::core::plugin::PluginCapabilities::none()
        }
    }

    #[test]
    fn test_registry_register() {
        let mut registry = PluginRegistry::new();
        assert!(registry.register::<TestPlugin>().is_ok());
        assert!(registry.contains(&PluginName::from("test")));
    }

    #[test]
    fn test_registry_create() {
        let mut registry = PluginRegistry::new();
        registry.register::<TestPlugin>().unwrap();

        let plugin = registry.create(&PluginName::from("test")).unwrap();
        assert_eq!(plugin.id(), "test");
        assert_eq!(plugin.version(), Version::new(1, 0, 0));
    }

    #[test]
    fn test_registry_duplicate() {
        let mut registry = PluginRegistry::new();
        assert!(registry.register::<TestPlugin>().is_ok());
        assert!(registry.register::<TestPlugin>().is_err());
    }

    #[test]
    fn test_registry_list() {
        let mut registry = PluginRegistry::new();
        registry.register::<TestPlugin>().unwrap();

        let list = registry.list();
        assert_eq!(list.len(), 1);
        assert!(list.contains(&"test".to_string()));
    }

    #[test]
    fn test_registry_builder() {
        let registry = PluginRegistryBuilder::new()
            .with_plugin::<TestPlugin>()
            .build();

        assert!(registry.contains(&PluginName::from("test")));
    }
}
