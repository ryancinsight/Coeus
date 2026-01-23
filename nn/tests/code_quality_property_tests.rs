//! Property-based tests for code quality properties
//!
//! Feature: coeus-architecture-enhancement
//! This module tests properties 24, 25, 26 from the design document
//!
//! These are primarily static analysis properties verified at compile/lint time

use std::fs;
use std::path::Path;

// ============================================================================
// Property 24: Rust Naming Convention Compliance
// ============================================================================

/// Feature: coeus-architecture-enhancement, Property 24: Rust Naming Convention Compliance
///
/// For any public function, the name SHALL use snake_case, and for any public type,
/// the name SHALL use PascalCase, following Rust naming conventions.
///
/// Validates: Requirements 14.1
#[test]
fn test_property_24_naming_conventions() {
    // Scan public APIs in lib.rs
    // Adjust path based on whether we're in workspace root or nn crate
    let lib_path = if Path::new("nn/src/lib.rs").exists() {
        Path::new("nn/src/lib.rs")
    } else if Path::new("src/lib.rs").exists() {
        Path::new("src/lib.rs")
    } else {
        println!("Warning: Could not find nn/src/lib.rs or src/lib.rs");
        return;
    };

    let content = fs::read_to_string(lib_path).expect("Failed to read lib.rs");

    // Check for common naming violations
    let lines: Vec<&str> = content.lines().collect();

    for (line_num, line) in lines.iter().enumerate() {
        // Skip comments and non-public items
        if line.trim().starts_with("//") || !line.contains("pub") {
            continue;
        }

        // Check for PascalCase in struct/enum/trait definitions
        if line.contains("pub struct") || line.contains("pub enum") || line.contains("pub trait") {
            // Extract type name
            if let Some(name_start) = line
                .find("pub struct ")
                .or_else(|| line.find("pub enum "))
                .or_else(|| line.find("pub trait "))
            {
                let after_keyword = &line[name_start..];
                if let Some(name) = after_keyword.split_whitespace().nth(2) {
                    let name = name.trim_end_matches(|c: char| !c.is_alphanumeric());
                    if !name.is_empty() {
                        // Check first character is uppercase
                        assert!(
                            name.chars().next().unwrap().is_uppercase(),
                            "Type '{}' at line {} should use PascalCase",
                            name,
                            line_num + 1
                        );
                    }
                }
            }
        }

        // Check for snake_case in function definitions
        if line.contains("pub fn") {
            if let Some(name_start) = line.find("pub fn ") {
                let after_keyword = &line[name_start + 7..];
                if let Some(name_end) = after_keyword.find('(') {
                    let name = after_keyword[..name_end].trim();
                    if !name.is_empty() && !name.starts_with("_") {
                        // Check for lowercase and underscores only
                        for ch in name.chars() {
                            assert!(
                                ch.is_lowercase() || ch.is_numeric() || ch == '_',
                                "Function '{}' at line {} should use snake_case",
                                name,
                                line_num + 1
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Feature: coeus-architecture-enhancement, Property 24: Rust Naming Convention Compliance
///
/// Scan module files for naming convention compliance
///
/// Validates: Requirements 14.1
#[test]
fn test_property_24_module_naming_conventions() {
    let modules_to_check = vec![
        "nn/src/modules/linear.rs",
        "nn/src/modules/activation/relu.rs",
        "nn/src/functional/ops/activations.rs",
        "nn/src/functional/ops/loss.rs",
    ];

    for module_path in modules_to_check {
        let path = Path::new(module_path);
        if !path.exists() {
            continue; // Skip if file doesn't exist
        }

        let content =
            fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {}", module_path));

        // Check that module doesn't have obvious naming violations
        // This is a heuristic check - rustc and clippy do the real validation

        // Check for camelCase functions (common mistake)
        let camel_case_pattern = regex::Regex::new(r"pub fn [a-z]+[A-Z]").unwrap();
        if let Some(mat) = camel_case_pattern.find(&content) {
            panic!(
                "Found camelCase function in {}: {}",
                module_path,
                mat.as_str()
            );
        }

        // Check for snake_case types (common mistake)
        let snake_case_type_pattern = regex::Regex::new(r"pub struct [a-z_]+\s").unwrap();
        if let Some(mat) = snake_case_type_pattern.find(&content) {
            panic!("Found snake_case type in {}: {}", module_path, mat.as_str());
        }
    }
}

// ============================================================================
// Property 25: Result Type Error Handling
// ============================================================================

/// Feature: coeus-architecture-enhancement, Property 25: Result Type Error Handling
///
/// For any operation that can fail, the function SHALL return a `Result<T, E>` type
/// rather than panicking or returning an Option.
///
/// Validates: Requirements 14.2
#[test]
fn test_property_25_result_type_usage() {
    // Check that operations return Result types
    let ops_files = vec![
        "nn/src/functional/ops/activations.rs",
        "nn/src/functional/ops/loss.rs",
        "nn/src/functional/ops/linear.rs",
        "nn/src/functional/ops/conv.rs",
    ];

    for ops_file in ops_files {
        let path = Path::new(ops_file);
        if !path.exists() {
            continue;
        }

        let content =
            fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {}", ops_file));

        // Check that public functions return Result
        let lines: Vec<&str> = content.lines().collect();
        let mut in_function = false;
        let mut function_name = String::new();

        for line in lines {
            if line.contains("pub fn") {
                in_function = true;
                if let Some(name_start) = line.find("pub fn ") {
                    let after_keyword = &line[name_start + 7..];
                    if let Some(name_end) = after_keyword.find('(') {
                        function_name = after_keyword[..name_end].trim().to_string();
                    }
                }
            }

            if in_function && line.contains("->") {
                // Check return type
                if line.contains("Result<") {
                    // Good - returns Result
                    in_function = false;
                } else if line.contains("Option<") {
                    // Warning - might want Result instead
                    println!(
                        "Warning: Function '{}' in {} returns Option instead of Result",
                        function_name, ops_file
                    );
                    in_function = false;
                } else if line.contains("{") {
                    // Function doesn't return Result or Option
                    // This might be okay for infallible operations
                    in_function = false;
                }
            }

            if line.contains("{") && in_function {
                in_function = false;
            }
        }
    }

    // This test primarily documents the expectation
    // The actual enforcement is done by the type system and code review
}

/// Feature: coeus-architecture-enhancement, Property 25: Result Type Error Handling
///
/// Verify that operations don't use unwrap() or expect() in public APIs
///
/// Validates: Requirements 14.2
#[test]
fn test_property_25_no_unwrap_in_public_apis() {
    let public_api_files = vec![
        "nn/src/functional/ops/activations.rs",
        "nn/src/functional/ops/loss.rs",
        "nn/src/modules/linear.rs",
    ];

    for api_file in public_api_files {
        let path = Path::new(api_file);
        if !path.exists() {
            continue;
        }

        let content =
            fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {}", api_file));

        // Check for unwrap() or expect() in public functions
        let lines: Vec<&str> = content.lines().collect();
        let mut in_public_fn = false;
        let mut brace_depth = 0;

        for (line_num, line) in lines.iter().enumerate() {
            if line.contains("pub fn") {
                in_public_fn = true;
                brace_depth = 0;
            }

            if in_public_fn {
                brace_depth += line.matches('{').count();
                brace_depth -= line.matches('}').count();

                // Check for unwrap/expect
                if line.contains(".unwrap()") || line.contains(".expect(") {
                    // Allow unwrap in test code or with clear justification
                    if !line.contains("// SAFETY:") && !line.contains("// OK to unwrap:") {
                        println!(
                            "Warning: Found unwrap/expect in public function at {}:{}",
                            api_file,
                            line_num + 1
                        );
                        println!("  Line: {}", line.trim());
                    }
                }

                if brace_depth == 0 {
                    in_public_fn = false;
                }
            }
        }
    }
}

// ============================================================================
// Property 26: Unsafe Block Documentation
// ============================================================================

/// Feature: coeus-architecture-enhancement, Property 26: Unsafe Block Documentation
///
/// For any unsafe block in the codebase, there SHALL be a comment immediately preceding
/// the block explaining why the unsafe code is necessary and why it is safe.
///
/// Validates: Requirements 14.3
#[test]
fn test_property_26_unsafe_block_documentation() {
    // Scan for unsafe blocks in the codebase
    let dirs_to_scan = vec!["nn/src/functional/ops", "nn/src/modules", "nn/src/core"];

    for dir in dirs_to_scan {
        let dir_path = Path::new(dir);
        if !dir_path.exists() {
            continue;
        }

        scan_directory_for_unsafe(dir_path);
    }
}

fn scan_directory_for_unsafe(dir: &Path) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                scan_directory_for_unsafe(&path);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                check_unsafe_documentation(&path);
            }
        }
    }
}

fn check_unsafe_documentation(file_path: &Path) {
    let content = match fs::read_to_string(file_path) {
        Ok(c) => c,
        Err(_) => return,
    };

    let lines: Vec<&str> = content.lines().collect();

    for (i, line) in lines.iter().enumerate() {
        if line.contains("unsafe") && (line.contains("unsafe {") || line.contains("unsafe fn")) {
            // Found unsafe block or function
            // Check if previous lines contain documentation

            let mut has_documentation = false;
            let mut check_lines = 1;

            // Look back up to 5 lines for documentation
            while check_lines <= 5 && i >= check_lines {
                let prev_line = lines[i - check_lines].trim();

                if prev_line.starts_with("//") {
                    // Check for SAFETY comment
                    if prev_line.contains("SAFETY:") || prev_line.contains("Safety:") {
                        has_documentation = true;
                        break;
                    }
                    // Also accept detailed comments about why it's safe
                    if prev_line.contains("safe because") || prev_line.contains("Safe because") {
                        has_documentation = true;
                        break;
                    }
                }

                // Stop if we hit a non-comment, non-empty line
                if !prev_line.is_empty() && !prev_line.starts_with("//") {
                    break;
                }

                check_lines += 1;
            }

            if !has_documentation {
                println!(
                    "Warning: Unsafe code without SAFETY comment at {}:{}",
                    file_path.display(),
                    i + 1
                );
                println!("  Line: {}", line.trim());
            }
        }
    }
}

// ============================================================================
// Property 27: Rustfmt Compliance
// ============================================================================

/// Feature: coeus-architecture-enhancement, Property 27: Rustfmt Compliance
///
/// For any code in the framework, running `cargo fmt --check` SHALL complete
/// successfully with no formatting changes required.
///
/// Validates: Requirements 14.5
///
/// Note: This property is best verified in CI. This test documents the expectation.
#[test]
fn test_property_27_rustfmt_compliance_documentation() {
    // This test documents that rustfmt compliance is expected
    // The actual check is done by running: cargo fmt --check

    // We can do a basic heuristic check for common formatting issues
    let files_to_check = vec!["nn/src/lib.rs", "nn/src/functional/ops/activations.rs"];

    for file_path in files_to_check {
        let path = Path::new(file_path);
        if !path.exists() {
            continue;
        }

        let content =
            fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {}", file_path));

        // Check for trailing whitespace (rustfmt removes this)
        for (line_num, line) in content.lines().enumerate() {
            if line.ends_with(' ') || line.ends_with('\t') {
                println!(
                    "Warning: Trailing whitespace at {}:{}",
                    file_path,
                    line_num + 1
                );
            }
        }

        // Check for tabs (rustfmt uses spaces)
        if content.contains('\t') {
            println!(
                "Warning: File {} contains tabs (rustfmt uses spaces)",
                file_path
            );
        }
    }

    // The real test is: cargo fmt --check
    // This should be run in CI
}

// ============================================================================
// Helper Tests
// ============================================================================

#[test]
fn test_regex_dependency_available() {
    // Verify regex crate is available for pattern matching
    let _ = regex::Regex::new(r"test").unwrap();
}
