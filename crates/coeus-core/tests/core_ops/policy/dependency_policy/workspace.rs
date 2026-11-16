//! Workspace discovery and manifest/source scanning helpers.

use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

pub(super) fn cargo_binary() -> PathBuf {
    std::env::var_os("CARGO")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("cargo"))
}

pub(super) fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("invariant: coeus-core lives under crates/ beneath the workspace root")
        .to_path_buf()
}

pub(super) fn workspace_crate_dirs(root: &Path) -> Vec<PathBuf> {
    fs::read_dir(root.join("crates"))
        .unwrap_or_else(|error| {
            panic!(
                "failed to read workspace crates dir {}: {error}",
                root.display()
            )
        })
        .filter_map(|entry| {
            let path = entry
                .unwrap_or_else(|error| panic!("failed to read directory entry: {error}"))
                .path();
            let name = path.file_name()?.to_str()?;
            (path.is_dir() && name.starts_with("coeus-")).then_some(path)
        })
        .collect()
}

pub(super) fn rust_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_files(root, OsStr::new("rs"), &mut files);
    files
}

pub(super) fn cargo_manifests(root: &Path) -> Vec<PathBuf> {
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

pub(super) fn is_production_dependency_section(section: &str) -> bool {
    section.contains("dependencies")
        && !section.contains("dev-dependencies")
        && !section.contains("build-dependencies")
}

pub(super) fn dependency_line_names_crate(line: &str, crate_name: &str) -> bool {
    line.strip_prefix(crate_name)
        .or_else(|| line.strip_prefix(&format!("\"{crate_name}\"")))
        .is_some_and(|rest| rest.trim_start().starts_with('='))
        || line.contains(&format!("package = \"{crate_name}\""))
}

pub(super) fn contains_crate_path(code: &str, crate_name: &str) -> bool {
    let path_prefix = format!("{crate_name}::");
    let mut search_start = 0;

    while let Some(relative_index) = code[search_start..].find(&path_prefix) {
        let index = search_start + relative_index;
        let has_identifier_prefix = code[..index]
            .chars()
            .next_back()
            .is_some_and(is_rust_identifier_char);

        if !has_identifier_prefix {
            return true;
        }

        search_start = index + path_prefix.len();
    }

    false
}

pub(super) fn imports_crate(code: &str, crate_name: &str) -> bool {
    let trimmed = code.trim_start();
    let import_body = trimmed
        .strip_prefix("use ")
        .or_else(|| trimmed.strip_prefix("pub use "));

    import_body.is_some_and(|body| {
        body.strip_prefix(crate_name).is_some_and(|rest| {
            rest.chars()
                .next()
                .is_some_and(|next| !is_rust_identifier_char(next))
        })
    })
}

fn is_rust_identifier_char(character: char) -> bool {
    character == '_' || character.is_ascii_alphanumeric()
}
