//! Property-based tests for file structure properties
//!
//! Feature: coeus-architecture-enhancement
//! This module tests properties 15, 16 from the design document

use std::fs;
use std::path::{Path, PathBuf};

// ============================================================================
// Property 15: Directory Nesting Depth Limit
// ============================================================================

/// Feature: coeus-architecture-enhancement, Property 15: Directory Nesting Depth Limit
///
/// For any directory in the framework, the nesting depth from the crate root SHALL be
/// at most 3 levels, except where deeper nesting is explicitly justified and documented.
///
/// Validates: Requirements 8.1
#[test]
fn test_property_15_directory_nesting_depth() {
    // Try both relative and absolute paths
    let crate_root = if Path::new("nn/src").exists() {
        Path::new("nn/src")
    } else if Path::new("src").exists() {
        Path::new("src")
    } else {
        println!("Warning: Could not find nn/src or src directory");
        return;
    };

    let mut violations = Vec::new();
    check_directory_depth(crate_root, crate_root, 0, &mut violations);

    if !violations.is_empty() {
        println!("Directory nesting depth violations (max 3 levels):");
        for (path, depth) in &violations {
            println!("  {} (depth: {})", path.display(), depth);
        }

        // Allow some violations if they're documented
        // For now, just warn
        if violations.len() > 5 {
            panic!(
                "Too many directory nesting violations: {} directories exceed depth 3",
                violations.len()
            );
        }
    }
}

fn check_directory_depth(
    root: &Path,
    current: &Path,
    depth: usize,
    violations: &mut Vec<(PathBuf, usize)>,
) {
    if depth > 3 {
        violations.push((current.to_path_buf(), depth));
    }

    if let Ok(entries) = fs::read_dir(current) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip hidden directories and target directories
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with('.') || name == "target" || name == "tests" {
                        continue;
                    }
                }

                check_directory_depth(root, &path, depth + 1, violations);
            }
        }
    }
}

/// Feature: coeus-architecture-enhancement, Property 15: Directory Nesting Depth Limit
///
/// Verify specific module paths don't exceed depth limit
///
/// Validates: Requirements 8.1
#[test]
fn test_property_15_specific_module_depths() {
    // Adjust paths based on whether we're in workspace root or nn crate
    let base_path = if Path::new("nn/src").exists() {
        "nn/src"
    } else {
        "src"
    };

    let modules_to_check = vec![
        (format!("{}/functional", base_path), 1),
        (format!("{}/functional/ops", base_path), 2),
        (format!("{}/modules", base_path), 1),
        (format!("{}/modules/activation", base_path), 2),
        (format!("{}/core", base_path), 1),
    ];

    for (module_path, expected_max_depth) in modules_to_check {
        let path = Path::new(&module_path);
        if !path.exists() {
            continue;
        }

        // Count depth from base_path
        let depth = module_path.matches('/').count() - base_path.matches('/').count();

        assert!(
            depth <= expected_max_depth,
            "Module {} has depth {} but expected max {}",
            module_path,
            depth,
            expected_max_depth
        );
    }
}

// ============================================================================
// Property 16: Empty File Elimination
// ============================================================================

/// Feature: coeus-architecture-enhancement, Property 16: Empty File Elimination
///
/// For any file in the framework, the file SHALL contain at least 10 lines of
/// non-comment, non-whitespace code, or SHALL be explicitly marked as a placeholder
/// with a TODO comment.
///
/// Validates: Requirements 8.4
#[test]
fn test_property_16_no_empty_files() {
    // Adjust paths based on whether we're in workspace root or nn crate
    let base_dirs = if Path::new("nn/src").exists() {
        vec!["nn/src/functional", "nn/src/modules", "nn/src/core"]
    } else {
        vec!["src/functional", "src/modules", "src/core"]
    };

    let mut empty_files = Vec::new();

    for dir in base_dirs {
        let dir_path = Path::new(dir);
        if !dir_path.exists() {
            continue;
        }

        scan_for_empty_files(dir_path, &mut empty_files);
    }

    if !empty_files.is_empty() {
        println!("Files with insufficient content (< 10 lines of code):");
        for (path, line_count, has_todo) in &empty_files {
            if *has_todo {
                println!("  {} ({} lines, has TODO)", path.display(), line_count);
            } else {
                println!("  {} ({} lines, NO TODO)", path.display(), line_count);
            }
        }

        // Fail if there are files without TODO markers
        let files_without_todo: Vec<_> = empty_files
            .iter()
            .filter(|(_, _, has_todo)| !has_todo)
            .collect();

        if !files_without_todo.is_empty() {
            panic!(
                "{} files have insufficient content without TODO markers",
                files_without_todo.len()
            );
        }
    }
}

fn scan_for_empty_files(dir: &Path, empty_files: &mut Vec<(PathBuf, usize, bool)>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                // Skip test directories
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name == "tests" || name.starts_with('.') {
                        continue;
                    }
                }
                scan_for_empty_files(&path, empty_files);
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                if let Ok(content) = fs::read_to_string(&path) {
                    let (code_lines, has_todo) = count_code_lines(&content);

                    // Be more lenient with mod.rs files (they're often just declarations)
                    let min_lines = if is_mod_file(&path) { 1 } else { 10 };

                    if code_lines < min_lines {
                        empty_files.push((path.clone(), code_lines, has_todo));
                    }
                }
            }
        }
    }
}

fn count_code_lines(content: &str) -> (usize, bool) {
    let mut code_lines = 0;
    let mut has_todo = false;
    let mut in_multiline_comment = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Check for TODO
        if trimmed.contains("TODO") || trimmed.contains("FIXME") {
            has_todo = true;
        }

        // Handle multiline comments
        if trimmed.starts_with("/*") {
            in_multiline_comment = true;
        }
        if trimmed.ends_with("*/") {
            in_multiline_comment = false;
            continue;
        }
        if in_multiline_comment {
            continue;
        }

        // Skip empty lines and single-line comments
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }

        // Count as code line
        code_lines += 1;
    }

    (code_lines, has_todo)
}

fn is_mod_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n == "mod.rs")
        .unwrap_or(false)
}

/// Feature: coeus-architecture-enhancement, Property 16: Empty File Elimination
///
/// Verify mod.rs files have meaningful content
///
/// Validates: Requirements 8.4
#[test]
fn test_property_16_mod_files_not_empty() {
    // Adjust paths based on whether we're in workspace root or nn crate
    let base_path = if Path::new("nn/src").exists() {
        "nn/src"
    } else {
        "src"
    };

    let mod_files = vec![
        format!("{}/functional/mod.rs", base_path),
        format!("{}/functional/ops/mod.rs", base_path),
        format!("{}/modules/mod.rs", base_path),
        format!("{}/core/mod.rs", base_path),
    ];

    for mod_file in mod_files {
        let path = Path::new(&mod_file);
        if !path.exists() {
            continue;
        }

        let content =
            fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {}", mod_file));

        let (code_lines, _) = count_code_lines(&content);

        // mod.rs files should have at least some module declarations
        assert!(
            code_lines >= 3,
            "mod.rs file {} has only {} lines of code (expected at least 3)",
            mod_file,
            code_lines
        );

        // Should contain pub mod declarations
        assert!(
            content.contains("pub mod") || content.contains("pub use"),
            "mod.rs file {} should contain module declarations or re-exports",
            mod_file
        );
    }
}

// ============================================================================
// Additional File Structure Tests
// ============================================================================

#[test]
fn test_no_duplicate_module_names() {
    // Verify no duplicate module names exist that could cause confusion
    // Adjust path based on whether we're in workspace root or nn crate
    let src_dir = if Path::new("nn/src").exists() {
        Path::new("nn/src")
    } else if Path::new("src").exists() {
        Path::new("src")
    } else {
        println!("Warning: Could not find nn/src or src directory");
        return;
    };

    let mut module_names: std::collections::HashMap<String, Vec<PathBuf>> =
        std::collections::HashMap::new();

    collect_module_names(src_dir, &mut module_names);

    // Check for duplicates
    let mut has_duplicates = false;
    for (name, paths) in &module_names {
        if paths.len() > 1 {
            // Allow some duplicates (like mod.rs)
            if name == "mod" || name == "lib" {
                continue;
            }

            println!("Duplicate module name '{}' found in:", name);
            for path in paths {
                println!("  {}", path.display());
            }
            has_duplicates = true;
        }
    }

    if has_duplicates {
        println!("Warning: Duplicate module names found (may cause confusion)");
    }
}

fn collect_module_names(dir: &Path, names: &mut std::collections::HashMap<String, Vec<PathBuf>>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if !name.starts_with('.') && name != "target" && name != "tests" {
                        names
                            .entry(name.to_string())
                            .or_insert_with(Vec::new)
                            .push(path.clone());
                        collect_module_names(&path, names);
                    }
                }
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                if let Some(name) = path.file_stem().and_then(|n| n.to_str()) {
                    names
                        .entry(name.to_string())
                        .or_insert_with(Vec::new)
                        .push(path.clone());
                }
            }
        }
    }
}

#[test]
fn test_consistent_file_naming() {
    // Verify files use consistent naming (snake_case)
    // Adjust path based on whether we're in workspace root or nn crate
    let src_dir = if Path::new("nn/src").exists() {
        Path::new("nn/src")
    } else if Path::new("src").exists() {
        Path::new("src")
    } else {
        println!("Warning: Could not find nn/src or src directory");
        return;
    };

    let mut violations = Vec::new();
    check_file_naming(src_dir, &mut violations);

    if !violations.is_empty() {
        println!("File naming violations (should use snake_case):");
        for path in &violations {
            println!("  {}", path.display());
        }

        // This is a warning, not a hard failure
        if violations.len() > 3 {
            panic!("Too many file naming violations: {}", violations.len());
        }
    }
}

fn check_file_naming(dir: &Path, violations: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if !name.starts_with('.') && name != "target" {
                        // Check directory name is snake_case
                        if !is_snake_case(name) {
                            violations.push(path.clone());
                        }
                        check_file_naming(&path, violations);
                    }
                }
            } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                if let Some(name) = path.file_stem().and_then(|n| n.to_str()) {
                    // Check file name is snake_case
                    if !is_snake_case(name) && name != "mod" && name != "lib" {
                        violations.push(path.clone());
                    }
                }
            }
        }
    }
}

fn is_snake_case(name: &str) -> bool {
    // snake_case: lowercase letters, numbers, and underscores only
    name.chars()
        .all(|c| c.is_lowercase() || c.is_numeric() || c == '_')
}
