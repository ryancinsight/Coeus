//! Serialization and deserialization for models.
//!
//! This module provides efficient serialization for models with hundreds of millions
//! of parameters, using a custom binary format optimized for speed and size.

use crate::foundation::{
    error::{Error, Result, ValidationError, ProcessingError, internal_error},
    types::Version,
};
use std::io::{Read, Write, Seek};
use std::mem;

/// Magic bytes for model files.
const MAGIC: &[u8; 8] = b"RUSTLLM\0";

/// Current serialization format version.
const FORMAT_VERSION: u32 = 1;

// Helper function for validation errors
fn validation_error(msg: &str) -> Error {
    Error::Validation(ValidationError::PatternMismatch {
        value: "input".to_string(),
        pattern: msg.to_string(),
    })
}

// Helper function for I/O errors
fn io_error(msg: &str) -> Error {
    Error::Processing(ProcessingError::Internal {
        reason: msg.to_string(),
    })
}

/// Header for serialized model files.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ModelHeader {
    /// Magic bytes for file identification.
    pub magic: [u8; 8],
    /// Format version.
    pub format_version: u32,
    /// Model version.
    pub model_version: Version,
    /// Number of parameters.
    pub param_count: u64,
    /// Total size of parameter data in bytes.
    pub param_bytes: u64,
    /// Offset to parameter data.
    pub param_offset: u64,
    /// Offset to metadata.
    pub metadata_offset: u64,
    /// Size of metadata in bytes.
    pub metadata_size: u64,
    /// Checksum of parameter data.
    pub checksum: u64,
    /// Reserved for future use.
    pub reserved: [u64; 8],
}

impl ModelHeader {
    /// Creates a new model header.
    pub fn new(
        model_version: Version,
        param_count: u64,
        param_bytes: u64,
    ) -> Self {
        Self {
            magic: *MAGIC,
            format_version: FORMAT_VERSION,
            model_version,
            param_count,
            param_bytes,
            param_offset: mem::size_of::<Self>() as u64,
            metadata_offset: 0,
            metadata_size: 0,
            checksum: 0,
            reserved: [0; 8],
        }
    }
    
    /// Validates the header.
    pub fn validate(&self) -> Result<()> {
        if &self.magic != MAGIC {
            return Err(validation_error("Invalid magic bytes"));
        }
        
        if self.format_version != FORMAT_VERSION {
            return Err(validation_error(&format!(
                "Unsupported format version: {}",
                self.format_version
            )));
        }
        
        Ok(())
    }
}

/// Trait for types that can be serialized to model files.
pub trait ModelSerializable {
    /// Writes the model to a writer.
    fn write_to<W: Write + Seek>(&self, writer: &mut W) -> Result<()>;
    
    /// Reads the model from a reader.
    fn read_from<R: Read + Seek>(reader: &mut R) -> Result<Self>
    where
        Self: Sized;
}

/// Efficient parameter serializer for large models.
pub struct ParameterSerializer {
    /// Buffer size for chunked I/O (8MB for efficient disk I/O).
    buffer_size: usize,
}

impl ParameterSerializer {
    /// Creates a new parameter serializer.
    pub fn new() -> Self {
        Self {
            buffer_size: 8 * 1024 * 1024, // 8MB chunks
        }
    }
    
    /// Writes parameters to a writer with progress callback.
    pub fn write_parameters<W, F>(
        &self,
        writer: &mut W,
        params: &[f32],
        mut progress: F,
    ) -> Result<u64>
    where
        W: Write,
        F: FnMut(usize, usize),
    {
        let total_bytes = params.len() * mem::size_of::<f32>();
        let mut written = 0;
        
        // Write in chunks for better performance
        for chunk in params.chunks(self.buffer_size / mem::size_of::<f32>()) {
            // Serialize each f32 in little-endian order
            let mut bytes = Vec::with_capacity(chunk.len() * mem::size_of::<f32>());
            for &val in chunk {
                bytes.extend_from_slice(&val.to_le_bytes());
            }
            
            writer.write_all(&bytes)
                .map_err(|e| io_error(&format!("Failed to write parameters: {}", e)))?;
            
            written += bytes.len();
            progress(written, total_bytes);
        }
        
        Ok(written as u64)
    }
    
    /// Reads parameters from a reader with progress callback.
    pub fn read_parameters<R, F>(
        &self,
        reader: &mut R,
        param_count: usize,
        mut progress: F,
    ) -> Result<Vec<f32>>
    where
        R: Read,
        F: FnMut(usize, usize),
    {
        let total_bytes = param_count * mem::size_of::<f32>();
        let mut params = vec![0.0f32; param_count];
        let mut read = 0;
        
        // Read in chunks
        let chunk_size = self.buffer_size / mem::size_of::<f32>();
        for chunk in params.chunks_mut(chunk_size) {
            let mut bytes = vec![0u8; chunk.len() * mem::size_of::<f32>()];
            
            reader.read_exact(&mut bytes)
                .map_err(|e| io_error(&format!("Failed to read parameters: {}", e)))?;
            
            // Convert bytes to f32 using little-endian
            for (i, val) in chunk.iter_mut().enumerate() {
                let start = i * mem::size_of::<f32>();
                let end = start + mem::size_of::<f32>();
                let arr: [u8; 4] = bytes[start..end].try_into().unwrap();
                *val = f32::from_le_bytes(arr);
            }
            
            read += bytes.len();
            progress(read, total_bytes);
        }
        
        Ok(params)
    }
}

impl Default for ParameterSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata for serialized models.
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model architecture name.
    pub architecture: String,
    /// Training configuration.
    pub training_config: Option<String>,
    /// Custom metadata as key-value pairs.
    pub custom: Vec<(String, String)>,
}

impl ModelMetadata {
    /// Creates new metadata.
    pub fn new(architecture: impl Into<String>) -> Self {
        Self {
            architecture: architecture.into(),
            training_config: None,
            custom: Vec::new(),
        }
    }
    
    /// Adds a custom metadata entry.
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.push((key.into(), value.into()));
        self
    }
    
    /// Serializes metadata to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        
        // Write architecture
        let arch_bytes = self.architecture.as_bytes();
        bytes.extend_from_slice(&(arch_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(arch_bytes);
        
        // Write training config
        if let Some(config) = &self.training_config {
            bytes.push(1); // Has config
            let config_bytes = config.as_bytes();
            bytes.extend_from_slice(&(config_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(config_bytes);
        } else {
            bytes.push(0); // No config
        }
        
        // Write custom metadata
        bytes.extend_from_slice(&(self.custom.len() as u32).to_le_bytes());
        for (key, value) in &self.custom {
            let key_bytes = key.as_bytes();
            bytes.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(key_bytes);
            
            let value_bytes = value.as_bytes();
            bytes.extend_from_slice(&(value_bytes.len() as u32).to_le_bytes());
            bytes.extend_from_slice(value_bytes);
        }
        
        Ok(bytes)
    }
    
    /// Deserializes metadata from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = 0;
        
        // Read architecture
        if bytes.len() < cursor + 4 {
            return Err(validation_error("Invalid metadata format"));
        }
        let arch_len = u32::from_le_bytes([
            bytes[cursor], bytes[cursor + 1], bytes[cursor + 2], bytes[cursor + 3]
        ]) as usize;
        cursor += 4;
        
        if bytes.len() < cursor + arch_len {
            return Err(validation_error("Invalid metadata format"));
        }
        let architecture = String::from_utf8(bytes[cursor..cursor + arch_len].to_vec())
            .map_err(|_| validation_error("Invalid UTF-8 in metadata"))?;
        cursor += arch_len;
        
        // Read training config
        if bytes.len() < cursor + 1 {
            return Err(validation_error("Invalid metadata format"));
        }
        let has_config = bytes[cursor] != 0;
        cursor += 1;
        
        let training_config = if has_config {
            if bytes.len() < cursor + 4 {
                return Err(validation_error("Invalid metadata format"));
            }
            let config_len = u32::from_le_bytes([
                bytes[cursor], bytes[cursor + 1], bytes[cursor + 2], bytes[cursor + 3]
            ]) as usize;
            cursor += 4;
            
            if bytes.len() < cursor + config_len {
                return Err(validation_error("Invalid metadata format"));
            }
            let config = String::from_utf8(bytes[cursor..cursor + config_len].to_vec())
                .map_err(|_| validation_error("Invalid UTF-8 in metadata"))?;
            cursor += config_len;
            Some(config)
        } else {
            None
        };
        
        // Read custom metadata
        if bytes.len() < cursor + 4 {
            return Err(validation_error("Invalid metadata format"));
        }
        let custom_count = u32::from_le_bytes([
            bytes[cursor], bytes[cursor + 1], bytes[cursor + 2], bytes[cursor + 3]
        ]) as usize;
        cursor += 4;
        
        let mut custom = Vec::with_capacity(custom_count);
        for _ in 0..custom_count {
            // Read key
            if bytes.len() < cursor + 4 {
                return Err(validation_error("Invalid metadata format"));
            }
            let key_len = u32::from_le_bytes([
                bytes[cursor], bytes[cursor + 1], bytes[cursor + 2], bytes[cursor + 3]
            ]) as usize;
            cursor += 4;
            
            if bytes.len() < cursor + key_len {
                return Err(validation_error("Invalid metadata format"));
            }
            let key = String::from_utf8(bytes[cursor..cursor + key_len].to_vec())
                .map_err(|_| validation_error("Invalid UTF-8 in metadata"))?;
            cursor += key_len;
            
            // Read value
            if bytes.len() < cursor + 4 {
                return Err(validation_error("Invalid metadata format"));
            }
            let value_len = u32::from_le_bytes([
                bytes[cursor], bytes[cursor + 1], bytes[cursor + 2], bytes[cursor + 3]
            ]) as usize;
            cursor += 4;
            
            if bytes.len() < cursor + value_len {
                return Err(validation_error("Invalid metadata format"));
            }
            let value = String::from_utf8(bytes[cursor..cursor + value_len].to_vec())
                .map_err(|_| validation_error("Invalid UTF-8 in metadata"))?;
            cursor += value_len;
            
            custom.push((key, value));
        }
        
        Ok(Self {
            architecture,
            training_config,
            custom,
        })
    }
}

/// Calculates a simple checksum for parameter data.
pub fn calculate_checksum(params: &[f32]) -> u64 {
    let mut checksum = 0u64;
    
    for &param in params {
        let bits = param.to_bits();
        checksum = checksum.wrapping_add(bits as u64);
        checksum = checksum.rotate_left(1);
    }
    
    checksum
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_header() {
        let header = ModelHeader::new(Version::new(1, 0, 0), 1000, 4000);
        assert_eq!(&header.magic, MAGIC);
        assert_eq!(header.format_version, FORMAT_VERSION);
        assert_eq!(header.param_count, 1000);
        assert_eq!(header.param_bytes, 4000);
        assert!(header.validate().is_ok());
    }
    
    #[test]
    fn test_parameter_serialization() {
        let params = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let serializer = ParameterSerializer::new();
        
        // Write to buffer
        let mut buffer = Vec::new();
        let written = serializer.write_parameters(&mut buffer, &params, |_, _| {}).unwrap();
        assert_eq!(written as usize, params.len() * mem::size_of::<f32>());
        
        // Read back
        let mut cursor = std::io::Cursor::new(buffer);
        let read_params = serializer.read_parameters(&mut cursor, params.len(), |_, _| {}).unwrap();
        assert_eq!(params, read_params);
    }
    
    #[test]
    fn test_metadata_serialization() {
        let metadata = ModelMetadata::new("transformer")
            .with_custom("layers", "12")
            .with_custom("heads", "8");
        
        let bytes = metadata.to_bytes().unwrap();
        let deserialized = ModelMetadata::from_bytes(&bytes).unwrap();
        
        assert_eq!(metadata.architecture, deserialized.architecture);
        assert_eq!(metadata.custom, deserialized.custom);
    }
    
    #[test]
    fn test_checksum() {
        let params = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let checksum1 = calculate_checksum(&params);
        let checksum2 = calculate_checksum(&params);
        assert_eq!(checksum1, checksum2);
        
        let different = vec![1.0, 2.0, 3.0, 4.0, 6.0];
        let checksum3 = calculate_checksum(&different);
        assert_ne!(checksum1, checksum3);
    }
}