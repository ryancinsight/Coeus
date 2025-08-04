//! Configuration system for plugins and components.

use crate::foundation::{
    error::{Error, Result},
    types::PluginName,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// A configuration value that can be of various types.
#[derive(Debug, Clone)]
pub enum ConfigValue {
    /// Boolean value.
    Bool(bool),
    /// Integer value.
    Integer(i64),
    /// Floating point value.
    Float(f64),
    /// String value.
    String(String),
    /// List of values.
    List(Vec<ConfigValue>),
    /// Nested configuration.
    Map(HashMap<String, ConfigValue>),
}

impl ConfigValue {
    /// Tries to get the value as a boolean.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
    
    /// Tries to get the value as an integer.
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Self::Integer(v) => Some(*v),
            _ => None,
        }
    }
    
    /// Tries to get the value as a float.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Integer(v) => Some(*v as f64),
            _ => None,
        }
    }
    
    /// Tries to get the value as a string.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }
    
    /// Tries to get the value as a list.
    pub fn as_list(&self) -> Option<&Vec<ConfigValue>> {
        match self {
            Self::List(v) => Some(v),
            _ => None,
        }
    }
    
    /// Tries to get the value as a map.
    pub fn as_map(&self) -> Option<&HashMap<String, ConfigValue>> {
        match self {
            Self::Map(v) => Some(v),
            _ => None,
        }
    }
}

/// Builder for ConfigValue.
pub struct ConfigBuilder {
    value: ConfigValue,
}

impl ConfigBuilder {
    /// Creates a new map builder.
    pub fn map() -> Self {
        Self {
            value: ConfigValue::Map(HashMap::new()),
        }
    }
    
    /// Adds a key-value pair to a map.
    pub fn with(mut self, key: impl Into<String>, value: impl Into<ConfigValue>) -> Self {
        if let ConfigValue::Map(ref mut map) = self.value {
            map.insert(key.into(), value.into());
        }
        self
    }
    
    /// Builds the configuration value.
    pub fn build(self) -> ConfigValue {
        self.value
    }
}

// Implement From traits for common types
impl From<bool> for ConfigValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<i32> for ConfigValue {
    fn from(v: i32) -> Self {
        Self::Integer(v as i64)
    }
}

impl From<i64> for ConfigValue {
    fn from(v: i64) -> Self {
        Self::Integer(v)
    }
}

impl From<f32> for ConfigValue {
    fn from(v: f32) -> Self {
        Self::Float(v as f64)
    }
}

impl From<f64> for ConfigValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<String> for ConfigValue {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl From<&str> for ConfigValue {
    fn from(v: &str) -> Self {
        Self::String(v.to_string())
    }
}

impl<T: Into<ConfigValue>> From<Vec<T>> for ConfigValue {
    fn from(v: Vec<T>) -> Self {
        Self::List(v.into_iter().map(Into::into).collect())
    }
}

/// Configuration store for managing plugin and component configurations.
#[derive(Debug, Clone)]
pub struct ConfigStore {
    configs: Arc<RwLock<HashMap<String, ConfigValue>>>,
}

impl ConfigStore {
    /// Creates a new configuration store.
    pub fn new() -> Self {
        Self {
            configs: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Sets a configuration value.
    pub fn set(&self, key: impl Into<String>, value: impl Into<ConfigValue>) -> Result<()> {
        let mut configs = self.configs.write()
            .map_err(|_| Error::Other("Failed to acquire config lock".to_string()))?;
        configs.insert(key.into(), value.into());
        Ok(())
    }
    
    /// Gets a configuration value.
    pub fn get(&self, key: &str) -> Result<Option<ConfigValue>> {
        let configs = self.configs.read()
            .map_err(|_| Error::Other("Failed to acquire config lock".to_string()))?;
        Ok(configs.get(key).cloned())
    }
    
    /// Gets a configuration value or returns a default.
    pub fn get_or_default<T>(&self, key: &str, default: T) -> Result<ConfigValue>
    where
        T: Into<ConfigValue>,
    {
        self.get(key).map(|v| v.unwrap_or_else(|| default.into()))
    }
    
    /// Removes a configuration value.
    pub fn remove(&self, key: &str) -> Result<Option<ConfigValue>> {
        let mut configs = self.configs.write()
            .map_err(|_| Error::Other("Failed to acquire config lock".to_string()))?;
        Ok(configs.remove(key))
    }
    
    /// Clears all configuration values.
    pub fn clear(&self) -> Result<()> {
        let mut configs = self.configs.write()
            .map_err(|_| Error::Other("Failed to acquire config lock".to_string()))?;
        configs.clear();
        Ok(())
    }
    
    /// Lists all configuration keys.
    pub fn keys(&self) -> Result<Vec<String>> {
        let configs = self.configs.read()
            .map_err(|_| Error::Other("Failed to acquire config lock".to_string()))?;
        Ok(configs.keys().cloned().collect())
    }
    
    /// Merges another configuration store into this one.
    pub fn merge(&self, other: &ConfigStore) -> Result<()> {
        let other_configs = other.configs.read()
            .map_err(|_| Error::Other("Failed to acquire other config lock".to_string()))?;
        let mut configs = self.configs.write()
            .map_err(|_| Error::Other("Failed to acquire config lock".to_string()))?;
        
        for (key, value) in other_configs.iter() {
            configs.insert(key.clone(), value.clone());
        }
        
        Ok(())
    }
}

impl Default for ConfigStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for types that can be configured.
pub trait Configurable {
    /// Applies configuration from a ConfigValue.
    fn configure(&mut self, config: &ConfigValue) -> Result<()>;
    
    /// Gets the current configuration as a ConfigValue.
    fn get_config(&self) -> ConfigValue;
}

/// Plugin configuration manager.
pub struct PluginConfigManager {
    store: ConfigStore,
}

impl PluginConfigManager {
    /// Creates a new plugin configuration manager.
    pub fn new() -> Self {
        Self {
            store: ConfigStore::new(),
        }
    }
    
    /// Sets configuration for a plugin.
    pub fn set_plugin_config(
        &self,
        plugin: &PluginName,
        config: ConfigValue,
    ) -> Result<()> {
        self.store.set(format!("plugin.{}", plugin.as_str()), config)
    }
    
    /// Gets configuration for a plugin.
    pub fn get_plugin_config(&self, plugin: &PluginName) -> Result<Option<ConfigValue>> {
        self.store.get(&format!("plugin.{}", plugin.as_str()))
    }
    
    /// Sets a specific configuration key for a plugin.
    pub fn set_plugin_key(
        &self,
        plugin: &PluginName,
        key: &str,
        value: impl Into<ConfigValue>,
    ) -> Result<()> {
        let config_key = format!("plugin.{}.{}", plugin.as_str(), key);
        self.store.set(config_key, value)
    }
    
    /// Gets a specific configuration key for a plugin.
    pub fn get_plugin_key(
        &self,
        plugin: &PluginName,
        key: &str,
    ) -> Result<Option<ConfigValue>> {
        let config_key = format!("plugin.{}.{}", plugin.as_str(), key);
        self.store.get(&config_key)
    }
}

impl Default for PluginConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_value_conversions() {
        let bool_val = ConfigValue::from(true);
        assert_eq!(bool_val.as_bool(), Some(true));
        
        let int_val = ConfigValue::from(42i32);
        assert_eq!(int_val.as_integer(), Some(42));
        
        let float_val = ConfigValue::from(3.14);
        assert_eq!(float_val.as_float(), Some(3.14));
        
        let string_val = ConfigValue::from("hello");
        assert_eq!(string_val.as_string(), Some("hello"));
        
        let list_val = ConfigValue::from(vec![1, 2, 3]);
        assert!(list_val.as_list().is_some());
    }
    
    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::map()
            .with("name", "test")
            .with("version", 1)
            .with("enabled", true)
            .with("threshold", 0.5)
            .build();
        
        if let ConfigValue::Map(map) = config {
            assert_eq!(map.get("name").and_then(|v| v.as_string()), Some("test"));
            assert_eq!(map.get("version").and_then(|v| v.as_integer()), Some(1));
            assert_eq!(map.get("enabled").and_then(|v| v.as_bool()), Some(true));
            assert_eq!(map.get("threshold").and_then(|v| v.as_float()), Some(0.5));
        } else {
            panic!("Expected Map");
        }
    }
    
    #[test]
    fn test_config_store() {
        let store = ConfigStore::new();
        
        // Set and get
        store.set("test.key", "value").unwrap();
        let value = store.get("test.key").unwrap();
        assert!(value.is_some());
        assert_eq!(value.unwrap().as_string(), Some("value"));
        
        // Get with default
        let default_val = store.get_or_default("missing", "default").unwrap();
        assert_eq!(default_val.as_string(), Some("default"));
        
        // List keys
        let keys = store.keys().unwrap();
        assert!(keys.contains(&"test.key".to_string()));
        
        // Remove
        let removed = store.remove("test.key").unwrap();
        assert!(removed.is_some());
        assert!(store.get("test.key").unwrap().is_none());
    }
    
    #[test]
    fn test_plugin_config_manager() {
        let manager = PluginConfigManager::new();
        let plugin_name = PluginName::from("test_plugin");
        
        // Set plugin config
        let config = ConfigBuilder::map()
            .with("batch_size", 32)
            .with("learning_rate", 0.001)
            .build();
        
        manager.set_plugin_config(&plugin_name, config).unwrap();
        
        // Get plugin config
        let retrieved = manager.get_plugin_config(&plugin_name).unwrap();
        assert!(retrieved.is_some());
        
        // Set/get specific keys
        manager.set_plugin_key(&plugin_name, "epochs", 10).unwrap();
        let epochs = manager.get_plugin_key(&plugin_name, "epochs").unwrap();
        assert_eq!(epochs.and_then(|v| v.as_integer()), Some(10));
    }
}