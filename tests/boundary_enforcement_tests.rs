//! Domain Boundary Enforcement Tests
//!
//! These tests verify that domain boundaries are properly enforced across crates.
//! They ensure that:
//! - Sparse operations only exist in sparse crate
//! - Quantization only exists in quantization crate
//! - Backend-specific code only exists in backend crate
//! - No cross-domain leakage occurs
//!
//! Requirements: 16.4, 16.6
//!
//! Note: These are static analysis tests that check file structure and code patterns.
//! They do not require the crates to compile successfully.

use std::fs;
use std::path::Path;

/// Test that tensor crate does not contain sparse operation implementations
///
/// Validates: Requirement 16.1 - Sparse operations exclusively in sparse crate
#[test]
fn test_no_sparse_implementations_in_tensor() {
    let tensor_src = Path::new("tensor/src");
    
    // Files that are allowed to reference sparse (for delegation)
    let allowed_files = vec![
        "tensor/src/lib.rs",           // Re-exports sparse storage types
        "tensor/src/ops/sparse.rs",    // Thin wrappers that delegate to sparse crate
        "tensor/src/ops/arithmetic/dispatch.rs", // Dispatch to sparse crate
    ];
    
    // Patterns that indicate sparse implementations (not just usage)
    let forbidden_patterns = vec![
        r"impl.*CsrStorage.*\{",         // CSR implementation blocks
        r"impl.*CscStorage.*\{",         // CSC implementation blocks
        r"impl.*CooStorage.*\{",         // COO implementation blocks
        r"fn.*csr.*\{",                  // CSR-specific functions
        r"fn.*csc.*\{",                  // CSC-specific functions
        r"fn.*coo.*\{",                  // COO-specific functions
        r"// CSR algorithm",            // Algorithm comments
        r"// Sparse matrix multiplication", // Sparse matmul implementation
    ];
    
    check_directory_for_patterns(
        tensor_src,
        &allowed_files,
        &forbidden_patterns,
        "Tensor crate should not contain sparse operation implementations",
    );
}

/// Test that dtype crate does not contain quantization logic
///
/// Validates: Requirement 16.2 - Quantization extracted to quantization crate
#[test]
fn test_no_quantization_logic_in_dtype() {
    let dtype_src = Path::new("dtype/src");
    
    // Files that are allowed to have minimal quantization references
    let allowed_files = vec![
        "dtype/src/lib.rs",      // Type query method is_quantized()
        "dtype/src/traits.rs",   // Trait method is_quantized()
    ];
    
    // Patterns that indicate quantization logic (not just type queries)
    let forbidden_patterns = vec![
        r"struct.*Quantizer",           // Quantizer implementations
        r"fn quantize\(",               // Quantization functions
        r"fn dequantize\(",             // Dequantization functions
        r"fn calibrate",                // Calibration functions
        r"fake_quantize",               // Fake quantization
        r"scale.*zero_point",           // Quantization parameters
        r"// Quantization algorithm",   // Algorithm comments
    ];
    
    check_directory_for_patterns(
        dtype_src,
        &allowed_files,
        &forbidden_patterns,
        "Dtype crate should not contain quantization logic",
    );
}

/// Test that storage crate does not contain backend-specific code
///
/// Validates: Requirement 16.3 - Backend-specific implementations in backend crate
#[test]
fn test_no_backend_specific_code_in_storage() {
    let storage_src = Path::new("storage/src");
    
    // No files are allowed to have backend-specific code
    let allowed_files: Vec<&str> = vec![];
    
    // Patterns that indicate backend-specific code
    let forbidden_patterns = vec![
        r"use.*simd",                   // SIMD intrinsics
        r"#\[target_feature",          // Target-specific features
        r"unsafe.*simd",                // Unsafe SIMD code
        r"CpuBackend",                  // Concrete backend types
        r"GpuBackend",
        r"TpuBackend",
        r"NpuBackend",
        r"cuda",                        // GPU-specific
        r"opencl",
        r"metal",
        r"vulkan",
        r"// SIMD optimization",        // Backend optimization comments
        r"// GPU kernel",
    ];
    
    check_directory_for_patterns(
        storage_src,
        &allowed_files,
        &forbidden_patterns,
        "Storage crate should not contain backend-specific code",
    );
}

/// Test that sparse operations only exist in sparse crate
///
/// Validates: Requirement 16.1 - Domain separation for sparse operations
#[test]
fn test_sparse_operations_only_in_sparse_crate() {
    // Check that sparse crate has the expected operations
    let sparse_src = Path::new("sparse/src");
    assert!(sparse_src.exists(), "Sparse crate should exist");
    
    // Verify sparse crate has proper structure
    let expected_dirs = vec![
        "sparse/src/formats/csr/arithmetic",
        "sparse/src/formats/csc/arithmetic",
        "sparse/src/formats/coo/arithmetic",
    ];
    
    for dir in expected_dirs {
        let path = Path::new(dir);
        assert!(
            path.exists(),
            "Sparse crate should have directory: {}",
            dir
        );
    }
    
    // Verify sparse operations exist
    let expected_files = vec![
        "sparse/src/formats/csr/arithmetic/add.rs",
        "sparse/src/formats/csr/arithmetic/mul.rs",
        "sparse/src/formats/csr/arithmetic/matmul.rs",
    ];
    
    for file in expected_files {
        let path = Path::new(file);
        assert!(
            path.exists(),
            "Sparse crate should have file: {}",
            file
        );
    }
}

/// Test that quantization only exists in quantization crate
///
/// Validates: Requirement 16.2 - Domain separation for quantization
#[test]
fn test_quantization_only_in_quantization_crate() {
    // Check that quantization crate has the expected structure
    let quant_src = Path::new("quantization/src");
    assert!(quant_src.exists(), "Quantization crate should exist");
    
    // Verify quantization crate has proper structure
    let expected_dirs = vec![
        "quantization/src/algorithms",
        "quantization/src/calibration",
        "quantization/src/fake_quantize",
    ];
    
    for dir in expected_dirs {
        let path = Path::new(dir);
        assert!(
            path.exists(),
            "Quantization crate should have directory: {}",
            dir
        );
    }
    
    // Verify quantization operations exist
    let expected_files = vec![
        "quantization/src/algorithms/symmetric.rs",
        "quantization/src/algorithms/asymmetric.rs",
        "quantization/src/calibration/entropy.rs",
    ];
    
    for file in expected_files {
        let path = Path::new(file);
        assert!(
            path.exists(),
            "Quantization crate should have file: {}",
            file
        );
    }
}

/// Test that dense operations only exist in dense crate
///
/// Validates: Requirement 16.2 - Domain separation for dense operations
#[test]
fn test_dense_operations_only_in_dense_crate() {
    // Check that dense crate has the expected structure
    let dense_src = Path::new("dense/src");
    assert!(dense_src.exists(), "Dense crate should exist");
    
    // Verify dense crate has proper structure
    let expected_files = vec![
        "dense/src/lib.rs",
        "dense/src/arithmetic.rs",
    ];
    
    for file in expected_files {
        let path = Path::new(file);
        assert!(
            path.exists(),
            "Dense crate should have file: {}",
            file
        );
    }
}

/// Test that no circular dependencies exist
///
/// Validates: Requirement 20.7 - No circular dependencies
#[test]
fn test_no_circular_dependencies() {
    // This test verifies the dependency hierarchy by checking Cargo.toml files
    
    // dtype should have no dependencies
    let dtype_toml = fs::read_to_string("dtype/Cargo.toml")
        .expect("Should read dtype Cargo.toml");
    assert!(
        !dtype_toml.contains("backend =") && !dtype_toml.contains("storage ="),
        "Dtype should not depend on backend or storage"
    );
    
    // backend should only depend on dtype
    let backend_toml = fs::read_to_string("backend/Cargo.toml")
        .expect("Should read backend Cargo.toml");
    assert!(
        !backend_toml.contains("storage =") && !backend_toml.contains("tensor ="),
        "Backend should not depend on storage or tensor"
    );
    
    // storage should only depend on backend and dtype
    let storage_toml = fs::read_to_string("storage/Cargo.toml")
        .expect("Should read storage Cargo.toml");
    assert!(
        !storage_toml.contains("tensor =") && !storage_toml.contains("nn ="),
        "Storage should not depend on tensor or nn"
    );
    
    // dense should only depend on storage and dtype
    let dense_toml = fs::read_to_string("dense/Cargo.toml")
        .expect("Should read dense Cargo.toml");
    assert!(
        !dense_toml.contains("tensor =") && !dense_toml.contains("nn ="),
        "Dense should not depend on tensor or nn"
    );
    
    // sparse should only depend on storage and dtype
    let sparse_toml = fs::read_to_string("sparse/Cargo.toml")
        .expect("Should read sparse Cargo.toml");
    assert!(
        !sparse_toml.contains("tensor =") && !sparse_toml.contains("nn ="),
        "Sparse should not depend on tensor or nn"
    );
    
    // quantization should only depend on storage and dtype
    let quant_toml = fs::read_to_string("quantization/Cargo.toml")
        .expect("Should read quantization Cargo.toml");
    assert!(
        !quant_toml.contains("tensor =") && !quant_toml.contains("nn ="),
        "Quantization should not depend on tensor or nn"
    );
}

/// Helper function to check directory for forbidden patterns
fn check_directory_for_patterns(
    dir: &Path,
    allowed_files: &[&str],
    forbidden_patterns: &[&str],
    error_message: &str,
) {
    if !dir.exists() {
        return;
    }
    
    let mut violations = Vec::new();
    
    // Walk through all Rust files in directory
    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "rs"))
    {
        let path = entry.path();
        let path_str = path.to_str().unwrap().replace('\\', "/");
        
        // Skip allowed files
        if allowed_files.iter().any(|&allowed| path_str.contains(allowed)) {
            continue;
        }
        
        // Read file content
        if let Ok(content) = fs::read_to_string(path) {
            // Check for forbidden patterns
            for pattern in forbidden_patterns {
                if let Ok(re) = regex::Regex::new(pattern) {
                    if re.is_match(&content) {
                        violations.push(format!(
                            "File {} contains forbidden pattern: {}",
                            path_str, pattern
                        ));
                    }
                }
            }
        }
    }
    
    if !violations.is_empty() {
        panic!(
            "{}\nViolations found:\n{}",
            error_message,
            violations.join("\n")
        );
    }
}

/// Test that storage only contains basic operations
///
/// Validates: Requirement 18.1-18.4 - Storage basic operations only
#[test]
fn test_storage_basic_operations_only() {
    let storage_src = Path::new("storage/src");
    
    // Files that are allowed to have complex operations (for now, until MatMul is moved)
    let allowed_files = vec![
        "storage/src/traits.rs",  // Trait definitions (MatMul trait exists but should be moved)
    ];
    
    // Patterns that indicate complex operations (not basic)
    let forbidden_patterns = vec![
        r"fn conv",                     // Convolution operations
        r"fn pool",                     // Pooling operations
        r"fn batch_norm",               // Batch normalization
        r"fn layer_norm",               // Layer normalization
        r"fn softmax",                  // Softmax activation
        r"fn relu",                     // ReLU activation
        r"fn gelu",                     // GELU activation
        r"// Convolution algorithm",   // Complex algorithm comments
        r"// Neural network",           // NN-specific operations
    ];
    
    check_directory_for_patterns(
        storage_src,
        &allowed_files,
        &forbidden_patterns,
        "Storage crate should only contain basic operations (add, sub, mul, div, reshape, transpose)",
    );
}

/// Test that tensor crate properly delegates to specialized crates
///
/// Validates: Requirement 16.5 - Clear interfaces for inter-crate communication
#[test]
fn test_tensor_delegates_to_specialized_crates() {
    // Check that tensor uses dense crate
    let tensor_cargo = fs::read_to_string("tensor/Cargo.toml")
        .expect("Should read tensor Cargo.toml");
    assert!(
        tensor_cargo.contains("dense =") || tensor_cargo.contains("dense.workspace"),
        "Tensor should depend on dense crate"
    );
    assert!(
        tensor_cargo.contains("sparse =") || tensor_cargo.contains("coeus-sparse"),
        "Tensor should depend on sparse crate"
    );
    
    // Check that tensor imports from specialized crates
    let tensor_lib = fs::read_to_string("tensor/src/lib.rs")
        .expect("Should read tensor lib.rs");
    assert!(
        tensor_lib.contains("use dense") || tensor_lib.contains("pub use dense"),
        "Tensor should use dense crate"
    );
}

/// Test that nn crate properly delegates to functional/ops
///
/// Validates: Requirement 1.3 - Layers delegate to functional/ops
#[test]
fn test_nn_layers_delegate_to_ops() {
    // This is a structural test - we verify the pattern exists
    let nn_src = Path::new("nn/src");
    
    // Verify functional/ops directory exists
    let ops_dir = nn_src.join("functional/ops");
    assert!(
        ops_dir.exists(),
        "NN crate should have functional/ops directory"
    );
    
    // Verify modules directory exists
    let modules_dir = nn_src.join("modules");
    assert!(
        modules_dir.exists(),
        "NN crate should have modules directory"
    );
    
    // Check that operations exist in functional/ops
    let expected_op_dirs = vec![
        "nn/src/functional/ops/activation",
        "nn/src/functional/ops/loss",
        "nn/src/functional/ops/convolution",
    ];
    
    for dir in expected_op_dirs {
        let path = Path::new(dir);
        assert!(
            path.exists(),
            "NN crate should have operation directory: {}",
            dir
        );
    }
}

#[cfg(test)]
mod dependency_tests {
    use super::*;
    
    /// Test the complete dependency hierarchy
    ///
    /// Validates: Requirement 20.1-20.7 - Clear dependency hierarchy
    #[test]
    fn test_dependency_hierarchy() {
        // Expected hierarchy:
        // dtype (no deps)
        //   ↑
        // backend (deps: dtype)
        //   ↑
        // storage (deps: backend, dtype)
        //   ↑
        // dense/sparse/quantization (deps: storage, dtype)
        //   ↑
        // tensor (deps: dense, sparse, quantization, storage)
        //   ↑
        // nn (deps: tensor, dense, sparse, quantization)
        
        // This is verified by test_no_circular_dependencies above
        // and by the Rust compiler (circular deps won't compile)
        
        // Additional verification: check that higher layers don't skip layers
        let nn_toml = fs::read_to_string("nn/Cargo.toml")
            .expect("Should read nn Cargo.toml");
        
        // NN should depend on tensor, not directly on storage
        assert!(
            nn_toml.contains("tensor =") || nn_toml.contains("tensor.workspace"),
            "NN should depend on tensor"
        );
        
        // NN can depend on dense/sparse/quantization for specialized operations
        // This is allowed per the architecture
    }
}
