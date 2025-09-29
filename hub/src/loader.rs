//! State dictionary loading and saving functionality

use crate::{HubError, Result, StateDict};
use coeus_tensor::Tensor;
use coeus_backend::cpu::CpuBackend;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

/// Load a state dictionary from a PyTorch .pth file
pub fn load_state_dict<P: AsRef<Path>>(path: P) -> Result<StateDict> {
    let path = path.as_ref();

    // Read the file
    let mut file = File::open(path).map_err(|e| HubError::Io {
        source: std::io::Error::new(
            e.kind(),
            format!("Failed to open {}: {}", path.display(), e),
        ),
    })?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Try to load as pickle (PyTorch format)
    load_pickle_state_dict(&buffer)
}

/// Save a state dictionary to a file
pub fn save_state_dict<P: AsRef<Path>>(state_dict: &StateDict, path: P) -> Result<()> {
    let path = path.as_ref();

    // For now, we'll save as JSON for simplicity
    // In a full implementation, this would save as PyTorch format
    let json = save_json_state_dict(state_dict)?;

    let mut file = File::create(path).map_err(|e| HubError::Io {
        source: std::io::Error::new(
            e.kind(),
            format!("Failed to create {}: {}", path.display(), e),
        ),
    })?;

    file.write_all(json.as_bytes())?;

    Ok(())
}

/// Load state dict from pickle data (PyTorch format)
fn load_pickle_state_dict(data: &[u8]) -> Result<StateDict> {
    // PyTorch uses a specific pickle protocol with torch-specific opcodes
    // For now, we'll implement a basic pickle parser that can handle simple PyTorch state dicts
    // This is a simplified implementation - production code would need more robust parsing

    if data.is_empty() {
        return Err(HubError::invalid_format("Empty pickle data"));
    }

    // PyTorch state dicts are typically pickled dictionaries
    // For basic compatibility, try to parse as a simple key-value structure
    // This is a placeholder - real implementation would need proper pickle parsing

    // For now, return an error with more informative message
    Err(HubError::invalid_format(
        "PyTorch pickle loading requires specialized parsing. Use torch.save() with pickle_protocol=2 for compatibility, or convert to JSON format.",
    ))
}

/// Load state dict from JSON data
pub fn load_json_state_dict(json: &str) -> Result<StateDict> {
    let parameters: std::collections::HashMap<String, Vec<f32>> = serde_json::from_str(json)?;

    let mut state_dict = StateDict::new();

    for (name, data) in parameters {
        // Assume 1D tensors for now
        // In a full implementation, this would handle shapes and types properly
        let len = data.len();
        let tensor = Tensor::from_vec(CpuBackend::default(), data, vec![len as usize]).unwrap();
        state_dict.insert(name, tensor);
    }

    Ok(state_dict)
}

/// Save state dict as JSON
pub fn save_json_state_dict(state_dict: &StateDict) -> Result<String> {
    // Convert tensors to vectors for JSON serialization
    let mut json_data = std::collections::HashMap::new();

    for (name, tensor) in &state_dict.parameters {
        let data: Vec<f32> = tensor.data().to_vec();
        json_data.insert(name.clone(), data);
    }

    serde_json::to_string_pretty(&json_data).map_err(|e| HubError::Json { source: e })
}

/// Load state dict from a remote URL
pub async fn load_state_dict_from_url(url: &str) -> Result<StateDict> {
    let response = reqwest::get(url).await?;
    let bytes = response.bytes().await?;
    load_pickle_state_dict(&bytes)
}

/// Download a file from URL to local path with progress
pub async fn download_file(url: &str, dest_path: &Path, expected_hash: Option<&str>) -> Result<()> {
    // Create parent directories if needed
    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let response = reqwest::get(url).await?;
    let bytes = response.bytes().await?;

    // Verify hash if provided
    if let Some(expected) = expected_hash {
        let actual = hash_data(&bytes);
        if actual != *expected {
            return Err(HubError::HashMismatch {
                expected: expected.to_string(),
                actual,
            });
        }
    }

    // Write to file
    let mut file = File::create(dest_path)?;
    file.write_all(&bytes)?;

    Ok(())
}

/// Compute SHA256 hash of data
fn hash_data(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    hex::encode(result)
}
