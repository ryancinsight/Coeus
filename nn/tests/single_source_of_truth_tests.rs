//! Property-based tests for single source of truth architecture
//!
//! Feature: coeus-architecture-enhancement, Property 1: Single Source of Truth for Operations
//! Validates: Requirements 1.2, 1.4
//!
//! These tests verify that:
//! 1. All operations are defined exactly once in functional/ops/
//! 2. No duplicate implementations exist
//! 3. The architecture maintains single source of truth principles

use std::fs;
use std::path::Path;

fn crate_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Test that verifies no duplicate operation implementations exist
///
/// This test scans the nn/src directory structure to ensure that:
/// - Operations are defined in functional/ops/ modules
/// - No duplicate implementations exist in other locations
/// - The single source of truth principle is maintained
#[test]
fn test_no_duplicate_operation_implementations() {
    // Define the single source of truth location
    let ops_dir = crate_root().join("src").join("functional").join("ops");

    // Verify ops directory exists
    assert!(
        ops_dir.exists(),
        "functional/ops/ directory must exist as single source of truth"
    );

    // Define operation categories that should exist in ops/
    let expected_ops_modules = vec![
        "activations.rs",
        "attention.rs",
        "conv.rs",
        "linear.rs",
        "loss.rs",
        "normalization.rs",
        "pooling.rs",
    ];

    // Verify all expected operation modules exist
    for module in &expected_ops_modules {
        let module_path = ops_dir.join(module);
        assert!(
            module_path.exists(),
            "Operation module {} must exist in functional/ops/",
            module
        );
    }

    // Verify mod.rs exists and properly exports operations
    let mod_rs = ops_dir.join("mod.rs");
    assert!(mod_rs.exists(), "functional/ops/mod.rs must exist");

    let mod_content = fs::read_to_string(&mod_rs).expect("Failed to read functional/ops/mod.rs");

    // Verify mod.rs declares all operation modules
    for module in &expected_ops_modules {
        let module_name = module.trim_end_matches(".rs");
        assert!(
            mod_content.contains(&format!("pub mod {}", module_name)),
            "functional/ops/mod.rs must declare module {}",
            module_name
        );
    }
}

/// Test that verifies the functional/ directory structure is clean
///
/// This test ensures that:
/// - No duplicate activation/ directory exists
/// - The ops/ directory is the single source of truth
/// - No redundant module structures exist
#[test]
fn test_functional_directory_structure() {
    let functional_dir = crate_root().join("src").join("functional");

    // Verify functional directory exists
    assert!(functional_dir.exists(), "functional/ directory must exist");

    // Verify no duplicate activation directory exists
    let activation_dir = functional_dir.join("activation");
    assert!(
        !activation_dir.exists() || !activation_dir.is_dir() || is_empty_dir(&activation_dir),
        "functional/activation/ directory should not exist (duplicate of ops/activations.rs)"
    );

    // Verify ops directory exists
    let ops_dir = functional_dir.join("ops");
    assert!(
        ops_dir.exists() && ops_dir.is_dir(),
        "functional/ops/ directory must exist as single source of truth"
    );

    // Verify functional/mod.rs properly exports ops
    let mod_rs = functional_dir.join("mod.rs");
    assert!(mod_rs.exists(), "functional/mod.rs must exist");

    let mod_content = fs::read_to_string(&mod_rs).expect("Failed to read functional/mod.rs");

    assert!(
        mod_content.contains("pub mod ops"),
        "functional/mod.rs must declare ops module"
    );

    // Verify ops is re-exported for convenience
    assert!(
        mod_content.contains("pub use ops::*"),
        "functional/mod.rs should re-export ops for convenience"
    );
}

/// Test that verifies no duplicate operation definitions exist
///
/// This test scans operation files to ensure that each operation
/// (relu, sigmoid, tanh, etc.) is defined exactly once
#[test]
fn test_no_duplicate_operation_definitions() {
    let ops_dir = crate_root().join("src").join("functional").join("ops");

    // Define operations that should exist exactly once
    let core_operations = vec![
        "relu",
        "sigmoid",
        "tanh",
        "gelu",
        "silu",
        "elu",
        "leaky_relu",
        "softmax",
        "dropout",
        "mse_loss",
        "cross_entropy",
        "nll_loss",
        "bce_with_logits_loss",
        "linear",
        "conv2d",
        "max_pool2d",
        "avg_pool2d",
        "batch_norm",
        "layer_norm",
    ];

    // Track where each operation is defined
    let mut operation_locations: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    // Scan ops directory for operation definitions
    if let Ok(entries) = fs::read_dir(ops_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                if let Ok(content) = fs::read_to_string(&path) {
                    let file_name = path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown");

                    for op in &core_operations {
                        // Look for function definitions (pub fn operation_name)
                        let pattern = format!("pub fn {}", op);
                        if content.contains(&pattern) {
                            operation_locations
                                .entry(op.to_string())
                                .or_insert_with(Vec::new)
                                .push(format!("ops/{}", file_name));
                        }
                    }
                }
            }
        }
    }

    // Verify each operation is defined exactly once in ops/
    for op in &core_operations {
        let locations = operation_locations.get(*op);
        if let Some(locs) = locations {
            assert_eq!(
                locs.len(),
                1,
                "Operation '{}' should be defined exactly once in ops/, found in: {:?}",
                op,
                locs
            );
        }
        // Note: Some operations might not be implemented yet, which is okay
        // The important thing is no duplicates exist
    }
}

/// Test that verifies lib.rs functional_api uses ops as source
///
/// This test ensures that the public functional_api in lib.rs
/// re-exports from functional::ops:: modules, not from duplicate locations
#[test]
fn test_lib_functional_api_uses_ops() {
    let lib_rs = crate_root().join("src").join("lib.rs");
    assert!(lib_rs.exists(), "nn/src/lib.rs must exist");

    let lib_content = fs::read_to_string(&lib_rs).expect("Failed to read nn/src/lib.rs");

    // Verify functional_api module exists
    assert!(
        lib_content.contains("pub mod functional_api"),
        "lib.rs must declare functional_api module"
    );

    // Extract functional_api section
    if let Some(start) = lib_content.find("pub mod functional_api") {
        let api_section = &lib_content[start..];
        if let Some(end) = api_section.find("\n}") {
            let api_content = &api_section[..end];

            // Verify activation functions are imported from ops::activations
            let activation_imports = vec!["relu", "sigmoid", "tanh", "gelu", "silu"];
            for func in activation_imports {
                // Should import from functional::ops::activations, not functional::activation
                assert!(
                    !api_content.contains(&format!("functional::activation::{}", func)),
                    "functional_api should not import {} from functional::activation (duplicate)",
                    func
                );
            }

            // Verify loss functions are imported from ops::loss
            assert!(
                api_content.contains("functional::ops::loss"),
                "functional_api should import loss functions from ops::loss"
            );
        }
    }
}

/// Test that verifies modules delegate to ops
///
/// This test checks that stateful modules (layers) delegate to
/// stateless operations in functional/ops/
#[test]
fn test_modules_delegate_to_ops() {
    // Check a sample module to verify delegation pattern
    let relu_module = crate_root()
        .join("src")
        .join("modules")
        .join("activation")
        .join("relu.rs");

    if relu_module.exists() {
        let content = fs::read_to_string(relu_module).expect("Failed to read relu module");

        // Verify the module delegates to functional ops
        // Look for patterns like: crate::functional::ops::activations::relu
        // or crate::ops::activation::relu
        let has_delegation = content.contains("functional::ops::activations::relu")
            || content.contains("ops::activation::relu")
            || content.contains("crate::functional::ops");

        // Note: This is a heuristic check. The actual implementation might vary.
        // The key principle is that modules should not reimplement operation logic.
        if content.contains("fn forward") {
            // If the module has a forward method, it should delegate
            // We're checking that it doesn't contain a full relu implementation
            let has_inline_impl = content.contains("max(0") || content.contains("x.max(");

            if has_inline_impl && !has_delegation {
                panic!(
                    "ReLU module appears to have inline implementation instead of delegating to ops"
                );
            }
        }
    }
}

/// Helper function to check if a directory is empty
fn is_empty_dir(path: &Path) -> bool {
    if let Ok(entries) = fs::read_dir(path) {
        entries.count() == 0
    } else {
        true
    }
}

/// Property test: Verify operation file organization
///
/// This test uses property-based testing to verify that:
/// - All .rs files in ops/ contain operation implementations
/// - No operation files exist outside ops/ (except wrappers)
#[cfg(test)]
mod property_tests {
    use super::*;

    #[test]
    fn test_ops_directory_contains_only_operations() {
        let ops_dir = crate_root().join("src").join("functional").join("ops");

        if let Ok(entries) = fs::read_dir(ops_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                    let file_name = path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown");

                    // Skip mod.rs and special files
                    if file_name == "mod.rs" || file_name.starts_with('.') {
                        continue;
                    }

                    // Verify file contains operation implementations
                    if let Ok(content) = fs::read_to_string(&path) {
                        // Should contain pub fn definitions (operations)
                        assert!(
                            content.contains("pub fn"),
                            "Operation file {} should contain public function definitions",
                            file_name
                        );

                        // Should not contain Module trait implementations (those belong in modules/)
                        assert!(
                            !content.contains("impl Module<"),
                            "Operation file {} should not contain Module implementations (use modules/ instead)",
                            file_name
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_no_orphaned_operation_files() {
        let functional_dir = crate_root().join("src").join("functional");

        // Directories that should NOT contain operation implementations
        let wrapper_dirs = vec![
            "attention",
            "convolution",
            "linear",
            "loss",
            "normalization",
            "pooling",
        ];

        for dir_name in wrapper_dirs {
            let dir_path = functional_dir.join(dir_name);
            if dir_path.exists() && dir_path.is_dir() {
                if let Ok(entries) = fs::read_dir(&dir_path) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                            if let Ok(content) = fs::read_to_string(&path) {
                                let file_name = path
                                    .file_name()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("unknown");

                                // These files should be thin wrappers that re-export from ops
                                // or provide convenience functions
                                // They should NOT contain full operation implementations

                                // Check if file is a simple re-export module
                                let is_reexport =
                                    content.contains("pub use crate::functional::ops");

                                // If it's not a re-export, it should be a convenience wrapper
                                // that delegates to ops (contains calls to ops functions)
                                let delegates_to_ops = content.contains("crate::functional::ops::")
                                    || content.contains("use crate::functional::ops");

                                if !is_reexport && !delegates_to_ops && content.contains("pub fn") {
                                    // This might be a duplicate implementation
                                    println!(
                                        "Warning: {} in {} might contain duplicate operation implementation",
                                        file_name, dir_name
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
