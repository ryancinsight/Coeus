//! Core type definitions for RustLLM Core.
//!
//! This module provides fundamental type aliases and definitions used throughout
//! the library, ensuring consistency and type safety.

use core::fmt;
use core::str::FromStr;

/// Version information for plugins and compatibility checking.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    /// Major version number (breaking changes).
    pub major: u16,
    /// Minor version number (new features).
    pub minor: u16,
    /// Patch version number (bug fixes).
    pub patch: u16,
}

impl Version {
    /// Creates a new version.
    #[inline]
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }
    
    /// Checks if this version is compatible with another version.
    ///
    /// Compatibility rules:
    /// - Major versions must match exactly
    /// - Minor version must be >= the required minor version
    /// - Patch version is ignored for compatibility
    #[inline]
    pub fn is_compatible_with(&self, required: &Self) -> bool {
        self.major == required.major && self.minor >= required.minor
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl FromStr for Version {
    type Err = crate::foundation::error::Error;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(crate::foundation::error::Error::Validation(
                crate::foundation::error::ValidationError::PatternMismatch {
                    value: s.to_string(),
                    pattern: "major.minor.patch".to_string(),
                }
            ));
        }
        
        let major = parts[0].parse::<u16>()
            .map_err(|e| crate::foundation::error::Error::Validation(
                crate::foundation::error::ValidationError::PatternMismatch {
                    value: parts[0].to_string(),
                    pattern: format!("valid u16: {}", e),
                }
            ))?;
        let minor = parts[1].parse::<u16>()
            .map_err(|e| crate::foundation::error::Error::Validation(
                crate::foundation::error::ValidationError::PatternMismatch {
                    value: parts[1].to_string(),
                    pattern: format!("valid u16: {}", e),
                }
            ))?;
        let patch = parts[2].parse::<u16>()
            .map_err(|e| crate::foundation::error::Error::Validation(
                crate::foundation::error::ValidationError::PatternMismatch {
                    value: parts[2].to_string(),
                    pattern: format!("valid u16: {}", e),
                }
            ))?;
        
        Ok(Self::new(major, minor, patch))
    }
}

/// Token ID type for efficient token representation.
pub type TokenId = u32;

/// Vocabulary size type.
pub type VocabSize = u32;

/// Sequence length type.
pub type SequenceLength = usize;

/// Batch size type.
pub type BatchSize = usize;

/// Model dimension type.
pub type ModelDim = usize;

/// Attention head count type.
pub type HeadCount = usize;

/// Layer count type.
pub type LayerCount = usize;

/// Float type used for model computations.
pub type ModelFloat = f32;

/// Double precision float type.
pub type ModelDouble = f64;

/// Type alias for dynamic trait objects.
pub type DynError = dyn core::error::Error + Send + Sync;

/// Model identifier type.
pub type ModelId = String;

/// A type-safe wrapper for plugin names.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PluginName(String);

impl PluginName {
    /// Creates a new plugin name.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
    
    /// Returns the plugin name as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PluginName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for PluginName {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for PluginName {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Configuration key type for type-safe configuration access.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConfigKey(String);

impl ConfigKey {
    /// Creates a new configuration key.
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into())
    }
    
    /// Returns the key as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ConfigKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A type for representing tensor shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Creates a new shape from dimensions.
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }
    
    /// Returns the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }
    
    /// Returns the dimensions.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    
    /// Computes the total number of elements.
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
    
    /// Checks if this shape is compatible with another shape.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.dims.len() == other.dims.len()
            && self.dims.iter().zip(other.dims.iter())
                .all(|(a, b)| *a == *b || *a == 1 || *b == 1)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

/// Phantom type markers for compile-time type safety.
pub mod markers {
    /// Marker for initialized state.
    pub struct Initialized;
    
    /// Marker for uninitialized state.
    pub struct Uninitialized;
    
    /// Marker for mutable access.
    pub struct Mutable;
    
    /// Marker for immutable access.
    pub struct Immutable;
    
    /// Marker for owned data.
    pub struct Owned;
    
    /// Marker for borrowed data.
    pub struct Borrowed;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version_compatibility() {
        let v1 = Version::new(1, 2, 3);
        let v2 = Version::new(1, 2, 0);
        let v3 = Version::new(1, 3, 0);
        let v4 = Version::new(2, 0, 0);
        
        assert!(v1.is_compatible_with(&v2));
        assert!(v3.is_compatible_with(&v2));
        assert!(!v2.is_compatible_with(&v3));
        assert!(!v1.is_compatible_with(&v4));
    }
    
    #[test]
    fn test_version_parsing() {
        let version = "1.2.3".parse::<Version>().unwrap();
        assert_eq!(version, Version::new(1, 2, 3));
        assert_eq!(version.to_string(), "1.2.3");
        
        assert!("1.2".parse::<Version>().is_err());
        assert!("a.b.c".parse::<Version>().is_err());
    }
    
    #[test]
    fn test_shape_operations() {
        let shape1 = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape1.ndim(), 3);
        assert_eq!(shape1.numel(), 24);
        
        let shape2 = Shape::new(vec![2, 1, 4]);
        assert!(shape1.is_compatible_with(&shape2));
        
        let shape3 = Shape::new(vec![2, 3]);
        assert!(!shape1.is_compatible_with(&shape3));
    }
}