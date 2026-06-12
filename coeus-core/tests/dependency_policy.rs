use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

const FORBIDDEN_PRODUCTION_CRATES: [&str; 2] = ["rayon", "tokio"];

#[test]
fn production_sources_do_not_import_non_ssot_parallel_runtimes() {
    let root = workspace_root();
    let mut violations = Vec::new();

    for crate_dir in workspace_crate_dirs(&root) {
        let src_dir = crate_dir.join("src");
        if !src_dir.is_dir() {
            continue;
        }

        for file in rust_files(&src_dir) {
            let contents = fs::read_to_string(&file)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", file.display()));
            for (line_index, line) in contents.lines().enumerate() {
                let code = line.split_once("//").map_or(line, |(code, _)| code);
                for crate_name in FORBIDDEN_PRODUCTION_CRATES {
                    let module_prefix = format!("{crate_name}::");
                    let import_prefix = format!("use {crate_name}");
                    if code.contains(&module_prefix)
                        || code.trim_start().starts_with(&import_prefix)
                    {
                        violations.push(format!(
                            "{}:{} imports `{}` outside the Moirai SSOT",
                            file.display(),
                            line_index + 1,
                            crate_name
                        ));
                    }
                }
            }
        }
    }

    assert_eq!(violations, Vec::<String>::new());
}

#[test]
fn production_manifests_do_not_depend_on_non_ssot_parallel_runtimes() {
    let root = workspace_root();
    let mut violations = Vec::new();

    for manifest in cargo_manifests(&root) {
        let contents = fs::read_to_string(&manifest)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", manifest.display()));
        let mut section = String::new();

        for (line_index, line) in contents.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') && trimmed.ends_with(']') {
                section.clear();
                section.push_str(trimmed);
                continue;
            }

            if !is_production_dependency_section(&section) {
                continue;
            }

            let code = trimmed
                .split_once('#')
                .map_or(trimmed, |(code, _)| code)
                .trim();
            for crate_name in FORBIDDEN_PRODUCTION_CRATES {
                if dependency_line_names_crate(code, crate_name) {
                    violations.push(format!(
                        "{}:{} declares production dependency `{}` outside the Moirai SSOT",
                        manifest.display(),
                        line_index + 1,
                        crate_name
                    ));
                }
            }
        }
    }

    assert_eq!(violations, Vec::<String>::new());
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("invariant: coeus-core lives directly under the workspace root")
        .to_path_buf()
}

fn workspace_crate_dirs(root: &Path) -> Vec<PathBuf> {
    fs::read_dir(root)
        .unwrap_or_else(|error| panic!("failed to read workspace root {}: {error}", root.display()))
        .filter_map(|entry| {
            let path = entry
                .unwrap_or_else(|error| panic!("failed to read directory entry: {error}"))
                .path();
            let name = path.file_name()?.to_str()?;
            (path.is_dir() && name.starts_with("coeus-")).then_some(path)
        })
        .collect()
}

fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_files(root, OsStr::new("rs"), &mut files);
    files
}

fn cargo_manifests(root: &Path) -> Vec<PathBuf> {
    let mut manifests = vec![root.join("Cargo.toml")];
    manifests.extend(
        workspace_crate_dirs(root)
            .into_iter()
            .map(|crate_dir| crate_dir.join("Cargo.toml"))
            .filter(|path| path.is_file()),
    );
    manifests
}

fn collect_files(root: &Path, extension: &OsStr, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root)
        .unwrap_or_else(|error| panic!("failed to read directory {}: {error}", root.display()))
    {
        let path = entry
            .unwrap_or_else(|error| panic!("failed to read directory entry: {error}"))
            .path();
        if path.is_dir() {
            collect_files(&path, extension, files);
        } else if path.extension() == Some(extension) {
            files.push(path);
        }
    }
}

fn is_production_dependency_section(section: &str) -> bool {
    section.contains("dependencies")
        && !section.contains("dev-dependencies")
        && !section.contains("build-dependencies")
}

fn dependency_line_names_crate(line: &str, crate_name: &str) -> bool {
    line.strip_prefix(crate_name)
        .or_else(|| line.strip_prefix(&format!("\"{crate_name}\"")))
        .is_some_and(|rest| rest.trim_start().starts_with('='))
}
