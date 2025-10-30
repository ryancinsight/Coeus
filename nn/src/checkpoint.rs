//! Model checkpointing for saving and loading training state.
//!
//! This module provides PyTorch-compatible checkpointing functionality that includes:
//! - Model parameters (state_dict)
//! - Optimizer state (momentum, RMSprop estimates, etc.)
//! - Training metadata (epoch, loss, learning rate, etc.)
//!
//! # Examples
//! ```rust
//! use coeus_nn::{Linear, Module};
//! use coeus_backend::CpuBackend;
//! use coeus_storage::DenseStorage;
//! use coeus_dtype::float::Float32;
//!
//! // Create a simple model
//! let model = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(784, 10).unwrap();
//!
//! // Train for some epochs...
//! let epoch = 10;
//! let loss = 0.123;
//!
//! // Note: Full checkpointing example requires optimizer serialization support
//! // let mut metadata = HashMap::new();
//! // metadata.insert("epoch".to_string(), epoch.to_string());
//! // checkpoint::save_checkpoint(&model, &optimizer, &metadata, Path::new("checkpoint.json")).unwrap();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::{NNError, Result};
use crate::module::StateDict;
#[cfg(feature = "safetensors")]
use crate::module::{Module, ModuleSerialize};
#[cfg(feature = "safetensors")]
use coeus_backend::Backend;
use coeus_dtype::{float::Float32, DataType};
#[cfg(feature = "safetensors")]
use coeus_storage::Storage;

/// Async checkpointing functionality for distributed training
///
/// Provides asynchronous checkpointing with improved I/O performance
/// and compatibility with streaming data pipelines.
pub mod async_checkpoint {

    use super::*;

    /// Asynchronously save a training checkpoint to a file
    ///
    /// # Arguments
    /// * `model` - The neural network model
    /// * `metadata` - Training metadata (epoch, loss, learning rate, etc.)
    /// * `path` - Path to save the checkpoint
    ///
    /// # Returns
    /// Future that resolves to Result indicating success or failure
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if serialization or file I/O fails
    #[cfg(feature = "safetensors")]
    pub async fn save_checkpoint_async<B, S, T, M>(
        model: &M,
        metadata: &HashMap<String, String>,
        path: &Path,
    ) -> Result<()>
    where
        B: Backend<Data = T> + Clone + std::default::Default,
        S: Storage<T> + Clone + 'static + coeus_storage::StorageFromVec<T>,
        T: DataType + Serialize + for<'de> Deserialize<'de>,
        M: Module<B, S, T> + ModuleSerialize<B, S, T>,
    {
        let checkpoint = Checkpoint {
            model_state: model.state_dict(),
            metadata: metadata.clone(),
        };

        // Serialize in background task to avoid blocking
        let json = tokio::task::spawn_blocking(move || {
            serde_json::to_string_pretty(&checkpoint).map_err(|e| NNError::SerializationError {
                message: format!("Failed to serialize checkpoint: {}", e),
            })
        })
        .await
        .map_err(|e| NNError::SerializationError {
            message: format!("Serialization task failed: {}", e),
        })??;

        // Async file I/O
        tokio::fs::write(path, json)
            .await
            .map_err(|e| NNError::SerializationError {
                message: format!("Failed to write checkpoint to file: {}", e),
            })?;

        Ok(())
    }

    /// Asynchronously load a training checkpoint from a file
    ///
    /// # Arguments
    /// * `path` - Path to load the checkpoint from
    ///
    /// # Returns
    /// Future that resolves to checkpoint data tuple
    ///
    /// # Errors
    /// Returns `NNError::SerializationError` if deserialization or file I/O fails
    pub async fn load_checkpoint_async<T>(path: &Path) -> Result<CheckpointData<T>>
    where
        T: DataType + for<'de> Deserialize<'de> + Send + 'static,
    {
        // Async file I/O
        let json =
            tokio::fs::read_to_string(path)
                .await
                .map_err(|e| NNError::SerializationError {
                    message: format!("Failed to read checkpoint from file: {}", e),
                })?;

        // Deserialize in background task to avoid blocking
        let checkpoint = tokio::task::spawn_blocking(move || {
            serde_json::from_str::<Checkpoint<T>>(&json).map_err(|e| NNError::SerializationError {
                message: format!("Failed to deserialize checkpoint: {}", e),
            })
        })
        .await
        .map_err(|e| NNError::SerializationError {
            message: format!("Deserialization task failed: {}", e),
        })??;

        Ok((checkpoint.model_state, checkpoint.metadata))
    }

    /// Streaming checkpoint writer for real-time checkpointing
    pub struct StreamingCheckpointWriter<W: tokio::io::AsyncWrite + Send + Unpin> {
        writer: W,
        buffer: Vec<u8>,
    }

    impl<W: tokio::io::AsyncWrite + Send + Unpin> StreamingCheckpointWriter<W> {
        /// Create a new streaming checkpoint writer
        pub fn new(writer: W) -> Self {
            Self {
                writer,
                buffer: Vec::new(),
            }
        }

        /// Append metadata to the checkpoint stream
        pub async fn write_metadata(&mut self, metadata: &HashMap<String, String>) -> Result<()> {
            let metadata_json =
                serde_json::to_string(metadata).map_err(|e| NNError::SerializationError {
                    message: format!("Failed to serialize metadata: {}", e),
                })?;

            self.buffer.extend_from_slice(b"METADATA:");
            self.buffer.extend_from_slice(metadata_json.as_bytes());
            self.buffer.extend_from_slice(b"\n");
            Ok(())
        }

        /// Flush buffered data to the writer
        pub async fn flush(&mut self) -> Result<()> {
            tokio::io::AsyncWriteExt::write_all(&mut self.writer, &self.buffer)
                .await
                .map_err(|e| NNError::SerializationError {
                    message: format!("Failed to flush checkpoint data: {}", e),
                })?;
            self.buffer.clear();
            Ok(())
        }
    }

    /// Async checkpoint loader that supports partial loading
    pub struct AsyncCheckpointLoader {
        path: std::path::PathBuf,
    }

    impl AsyncCheckpointLoader {
        /// Create a new async checkpoint loader
        pub fn new<P: AsRef<Path>>(path: P) -> Self {
            Self {
                path: path.as_ref().to_path_buf(),
            }
        }

        /// Load checkpoint metadata only (useful for quick validation)
        pub async fn load_metadata_only(&self) -> Result<HashMap<String, String>> {
            use tokio::io::AsyncReadExt;

            let mut file = tokio::fs::File::open(&self.path).await.map_err(|e| {
                NNError::SerializationError {
                    message: format!("Failed to open checkpoint file: {}", e),
                }
            })?;

            let mut contents = String::new();
            file.read_to_string(&mut contents)
                .await
                .map_err(|e| NNError::SerializationError {
                    message: format!("Failed to read checkpoint file: {}", e),
                })?;

            let checkpoint: Checkpoint<Float32> =
                serde_json::from_str(&contents).map_err(|e| NNError::SerializationError {
                    message: format!("Failed to parse checkpoint: {}", e),
                })?;

            Ok(checkpoint.metadata)
        }
    }

    #[cfg(all(test, feature = "safetensors"))]
    mod tests {
        use super::*;
        use crate::Linear;
        use coeus_backend::CpuBackend;
        use coeus_dtype::float::Float32;
        use coeus_storage::DenseStorage;
        use std::collections::HashMap;

        #[tokio::test]
        async fn test_async_checkpoint_save_load() {
            let model =
                Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 5).unwrap();

            let mut metadata = HashMap::new();
            metadata.insert("epoch".to_string(), "10".to_string());
            metadata.insert("loss".to_string(), "0.123".to_string());

            let path = std::path::Path::new("test_async_checkpoint.json");

            // Save checkpoint asynchronously
            save_checkpoint_async(&model, &metadata, path)
                .await
                .unwrap();

            // Load checkpoint asynchronously
            let (_model_state, loaded_metadata) =
                load_checkpoint_async::<Float32>(path).await.unwrap();

            // Verify metadata
            assert_eq!(loaded_metadata.get("epoch"), Some(&"10".to_string()));
            assert_eq!(loaded_metadata.get("loss"), Some(&"0.123".to_string()));

            // Cleanup
            tokio::fs::remove_file(path).await.ok();
        }

        #[tokio::test]
        async fn test_streaming_checkpoint_writer() {
            use tokio::io::BufWriter;

            let buffer = Vec::new();
            let writer = BufWriter::new(buffer);
            let mut checkpoint_writer = StreamingCheckpointWriter::new(writer);

            let mut metadata = HashMap::new();
            metadata.insert("test".to_string(), "value".to_string());

            checkpoint_writer.write_metadata(&metadata).await.unwrap();
            // Note: Would need to extract the writer to verify contents in real implementation
        }
    }
}

/// Type alias for checkpoint load results: (model_state, optimizer_state, metadata).
pub type CheckpointData<T> = (StateDict<T>, HashMap<String, String>);

// Re-export from coeus-optim for convenience
// coeus_optim temporarily disabled
// pub use coeus_optim::{OptimizerSerialize, OptimizerStateDict};

/// Training checkpoint containing model state, optimizer state, and metadata.
///
/// This structure is compatible with PyTorch's checkpoint format:
/// ```python
/// torch.save({
///     'model_state_dict': model.state_dict(),
///     'optimizer_state_dict': optimizer.state_dict(),
///     'epoch': epoch,
///     'loss': loss,
/// }, 'checkpoint.pth')
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint<T: DataType> {
    /// Model parameters (state_dict)
    pub model_state: StateDict<T>,

    /// Optimizer state (momentum, RMSprop estimates, etc.)
    // pub optimizer_state: OptimizerStateDict<T>, // Temporarily disabled

    /// Training metadata (epoch, loss, learning rate, etc.)
    pub metadata: HashMap<String, String>,
}

/// Save a training checkpoint to a JSON file.
///
/// This saves the model state, optimizer state, and training metadata to enable
/// resuming training from a specific point.
///
/// # Arguments
/// * `model` - The neural network model
/// * `optimizer` - The optimizer (must implement state_dict())
/// * `metadata` - Training metadata (epoch, loss, learning rate, etc.)
/// * `path` - Path to save the checkpoint
///
/// # Returns
/// Result indicating success or failure of the save operation.
///
/// # Errors
/// Returns `NNError::SerializationError` if serialization or file I/O fails.
///
/// # Examples
/// ```rust
/// use coeus_nn::{Linear, Module};
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
/// use std::collections::HashMap;
///
/// // Create a simple model for demonstration
/// let model = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(784, 10).unwrap();
/// let _params = model.parameters(); // Parameters would be used for checkpointing
/// // Note: Full checkpointing example requires optimizer serialization support
///
/// let _metadata: HashMap<String, String> = HashMap::new(); // Would contain training metadata
/// // checkpoint::save_checkpoint(&model, &optimizer, &metadata, Path::new("checkpoint.json")).unwrap();
/// ```
#[cfg(feature = "safetensors")]
pub fn save_checkpoint<B, S, T, M>(
    model: &M,
    metadata: &HashMap<String, String>,
    path: &Path,
) -> Result<()>
where
    B: Backend<Data = T> + Clone + std::default::Default,
    S: Storage<T> + Clone + 'static + coeus_storage::StorageFromVec<T>,
    T: DataType + Serialize + for<'de> Deserialize<'de>,
    M: Module<B, S, T> + ModuleSerialize<B, S, T>,
{
    let checkpoint = Checkpoint {
        model_state: model.state_dict(),
        // optimizer_state: optimizer.state_dict(), // Temporarily disabled
        metadata: metadata.clone(),
    };

    let json =
        serde_json::to_string_pretty(&checkpoint).map_err(|e| NNError::SerializationError {
            message: format!("Failed to serialize checkpoint: {}", e),
        })?;

    std::fs::write(path, json).map_err(|e| NNError::SerializationError {
        message: format!("Failed to write checkpoint to file: {}", e),
    })?;

    Ok(())
}

/// Load a training checkpoint from a JSON file.
///
/// This loads the model state, optimizer state, and training metadata to enable
/// resuming training from a specific point.
///
/// # Arguments
/// * `path` - Path to load the checkpoint from
///
/// # Returns
/// A tuple containing (model_state, optimizer_state, metadata).
///
/// # Errors
/// Returns `NNError::SerializationError` if deserialization or file I/O fails.
///
/// # Examples
/// ```rust
/// use coeus_nn::{Linear, Module};
/// use coeus_backend::CpuBackend;
/// use coeus_storage::DenseStorage;
/// use coeus_dtype::float::Float32;
///
/// // Create a model for demonstration
/// let model = Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(784, 10).unwrap();
/// let _params = model.parameters();
/// // Note: Full checkpointing example requires optimizer serialization support
/// // let (model_state, optimizer_state, loaded_metadata) = checkpoint::load_checkpoint::<Float32>(Path::new("checkpoint.json")).unwrap();
/// ```
pub fn load_checkpoint<T>(path: &Path) -> Result<CheckpointData<T>>
where
    T: DataType + for<'de> Deserialize<'de>,
{
    let json = std::fs::read_to_string(path).map_err(|e| NNError::SerializationError {
        message: format!("Failed to read checkpoint from file: {}", e),
    })?;

    let checkpoint: Checkpoint<T> =
        serde_json::from_str(&json).map_err(|e| NNError::SerializationError {
            message: format!("Failed to deserialize checkpoint: {}", e),
        })?;

    Ok((checkpoint.model_state, checkpoint.metadata))
}

#[cfg(all(test, feature = "safetensors"))]
mod tests {
    use super::*;
    use crate::Linear;
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_storage::DenseStorage;
    use std::collections::HashMap;
    use std::path::Path;

    // Mock types for testing without optim dependency
    type OptimizerStateDict<T> = HashMap<String, Vec<T>>;
    trait OptimizerSerialize<T> {
        fn state_dict(&self) -> OptimizerStateDict<T>;
        fn load_state_dict(
            &mut self,
            state_dict: &OptimizerStateDict<T>,
        ) -> std::result::Result<(), String>;
    }

    // Mock optimizer for testing
    struct MockOptimizer {
        state: OptimizerStateDict<Float32>,
    }

    impl MockOptimizer {
        fn new() -> Self {
            Self {
                state: HashMap::new(),
            }
        }
    }

    impl OptimizerSerialize<Float32> for MockOptimizer {
        fn state_dict(&self) -> OptimizerStateDict<Float32> {
            self.state.clone()
        }

        fn load_state_dict(
            &mut self,
            state_dict: &OptimizerStateDict<Float32>,
        ) -> std::result::Result<(), String> {
            self.state = state_dict.clone();
            Ok(())
        }
    }

    #[test]
    fn test_checkpoint_save_load() {
        let model =
            Linear::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(10, 5).unwrap();
        let optimizer = MockOptimizer::new();

        let mut metadata = HashMap::new();
        metadata.insert("epoch".to_string(), "10".to_string());
        metadata.insert("loss".to_string(), "0.123".to_string());

        let path = Path::new("test_checkpoint.json");

        // Save checkpoint
        save_checkpoint(&model, &metadata, path).unwrap();

        // Load checkpoint
        let (_model_state, loaded_metadata) = load_checkpoint::<Float32>(path).unwrap();

        // Verify metadata
        assert_eq!(loaded_metadata.get("epoch"), Some(&"10".to_string()));
        assert_eq!(loaded_metadata.get("loss"), Some(&"0.123".to_string()));

        // Cleanup
        std::fs::remove_file(path).ok();
    }
}
