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
//! let model = Linear::<CpuBackend, DenseStorage<Float32>, Float32>::new(784, 10).unwrap();
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

/// Type alias for checkpoint load results: (model_state, optimizer_state, metadata).
pub type CheckpointData<T> = (StateDict<T>, OptimizerStateDict<T>, HashMap<String, String>);

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use coeus_backend::Backend;
use coeus_dtype::DataType;
use coeus_storage::Storage;

use crate::error::{NNError, Result};
use crate::module::{Module, ModuleSerialize, StateDict};

// Re-export from coeus-optim for convenience
pub use coeus_optim::{OptimizerSerialize, OptimizerStateDict};

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
    pub optimizer_state: OptimizerStateDict<T>,

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
/// let model = Linear::<CpuBackend, DenseStorage<Float32>, Float32>::new(784, 10).unwrap();
/// let _params = model.parameters(); // Parameters would be used for checkpointing
/// // Note: Full checkpointing example requires optimizer serialization support
///
/// let _metadata: HashMap<String, String> = HashMap::new(); // Would contain training metadata
/// // checkpoint::save_checkpoint(&model, &optimizer, &metadata, Path::new("checkpoint.json")).unwrap();
/// ```
pub fn save_checkpoint<B, S, T, M, O>(
    model: &M,
    optimizer: &O,
    metadata: &HashMap<String, String>,
    path: &Path,
) -> Result<()>
where
    B: Backend + Clone + std::default::Default,
    S: Storage<T> + Clone + 'static + coeus_storage::StorageFromVec<T>,
    T: DataType + Serialize + for<'de> Deserialize<'de>,
    M: Module<B, S, T> + ModuleSerialize<B, S, T>,
    O: OptimizerSerialize<T>,
{
    let checkpoint = Checkpoint {
        model_state: model.state_dict(),
        optimizer_state: optimizer.state_dict(),
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
/// let model = Linear::<CpuBackend, DenseStorage<Float32>, Float32>::new(784, 10).unwrap();
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

    Ok((
        checkpoint.model_state,
        checkpoint.optimizer_state,
        checkpoint.metadata,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Linear;
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_storage::DenseStorage;
    use std::path::Path;

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
        ) -> std::result::Result<(), coeus_optim::OptimizerError> {
            self.state = state_dict.clone();
            Ok(())
        }
    }

    #[test]
    fn test_checkpoint_save_load() {
        let model = Linear::<CpuBackend, DenseStorage<Float32>, Float32>::new(10, 5).unwrap();
        let optimizer = MockOptimizer::new();

        let mut metadata = HashMap::new();
        metadata.insert("epoch".to_string(), "10".to_string());
        metadata.insert("loss".to_string(), "0.123".to_string());

        let path = Path::new("test_checkpoint.json");

        // Save checkpoint
        save_checkpoint(&model, &optimizer, &metadata, path).unwrap();

        // Load checkpoint
        let (_model_state, _optimizer_state, loaded_metadata) =
            load_checkpoint::<Float32>(path).unwrap();

        // Verify metadata
        assert_eq!(loaded_metadata.get("epoch"), Some(&"10".to_string()));
        assert_eq!(loaded_metadata.get("loss"), Some(&"0.123".to_string()));

        // Cleanup
        std::fs::remove_file(path).ok();
    }
}
