//! Persistence domain - Model serialization and storage.
//!
//! This domain handles model persistence and checkpointing.

use crate::foundation::error::Result;

/// Persistence service.
pub struct PersistenceService;

/// Storage backend types.
#[derive(Debug, Clone)]
pub enum StorageBackend {
    FileSystem { path: String },
    Memory,
}

/// Serialization format.
#[derive(Debug, Clone)]
pub enum SerializationFormat {
    GGUF,
    SafeTensors,
    Custom(String),
}

/// Checkpoint.
#[derive(Debug)]
pub struct Checkpoint {
    pub id: String,
    pub model_id: String,
    pub epoch: usize,
}

/// Persistence events.
#[derive(Debug, Clone)]
pub enum PersistenceEvent {
    Saved { checkpoint_id: String },
    Loaded { checkpoint_id: String },
}