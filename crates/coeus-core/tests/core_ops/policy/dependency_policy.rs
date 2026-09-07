use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy)]
struct ForbiddenProductionCrate {
    manifest_name: &'static str,
    import_name: &'static str,
    owner: &'static str,
    /// Atlas-owned SSOT replacement crates through which the forbidden crate
    /// may legitimately reach Coeus. Path-dependent Atlas crates are the
    /// Atlas's responsibility; Coeus only forbids reachability that escapes
    /// the Atlas replacement stack.
    atlas_parents: &'static [&'static str],
}

const FORBIDDEN_PRODUCTION_CRATES: [ForbiddenProductionCrate; 8] = [
    ForbiddenProductionCrate {
        manifest_name: "pollster",
        import_name: "pollster",
        owner: "Moirai async SSOT",
        atlas_parents: &[],
    },
    ForbiddenProductionCrate {
        manifest_name: "rayon",
        import_name: "rayon",
        owner: "Moirai parallel SSOT",
        atlas_parents: &[],
    },
    ForbiddenProductionCrate {
        manifest_name: "tokio",
        import_name: "tokio",
        owner: "Moirai async SSOT",
        atlas_parents: &[],
    },
    ForbiddenProductionCrate {
        manifest_name: "burn",
        import_name: "burn",
        owner: "Coeus runtime tensor/autograd stack",
        atlas_parents: &[],
    },
    ForbiddenProductionCrate {
        manifest_name: "nalgebra",
        import_name: "nalgebra",
        owner: "Coeus/Leto tensor kernel stack",
        atlas_parents: &["leto"],
    },
    ForbiddenProductionCrate {
        manifest_name: "ndarray",
        import_name: "ndarray",
        owner: "Coeus/Leto tensor kernel stack",
        // `apollo-fft` is the Atlas FFT SSOT replacement that itself depends
        // on ndarray-as-substrate at the apollo workspace level. Reachability
        // through apollo-fft is the Atlas's own concern; only paths that
        // **also** touch Coeus without passing through the Apollo/Leto stack
        // are real Coeus violations.
        atlas_parents: &["leto", "apollo-fft"],
    },
    ForbiddenProductionCrate {
        manifest_name: "tch",
        import_name: "tch",
        owner: "Coeus runtime tensor/autograd stack",
        atlas_parents: &[],
    },
    ForbiddenProductionCrate {
        manifest_name: "rustfft",
        import_name: "rustfft",
        owner: "Atlas-owned Apollo FFT implementation",
        atlas_parents: &["apollo-fft"],
    },
];

#[test]
fn production_sources_do_not_import_non_ssot_runtime_or_replacement_crates() {
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
                for forbidden_crate in FORBIDDEN_PRODUCTION_CRATES {
                    if contains_crate_path(code, forbidden_crate.import_name)
                        || imports_crate(code, forbidden_crate.import_name)
                    {
                        violations.push(format!(
                            "{}:{} imports `{}` outside {}",
                            file.display(),
                            line_index + 1,
                            forbidden_crate.import_name,
                            forbidden_crate.owner
                        ));
                    }
                }
            }
        }
    }

    assert_eq!(violations, Vec::<String>::new());
}

#[test]
fn production_manifests_do_not_depend_on_non_ssot_runtime_or_replacement_crates() {
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
            for forbidden_crate in FORBIDDEN_PRODUCTION_CRATES {
                if dependency_line_names_crate(code, forbidden_crate.manifest_name) {
                    violations.push(format!(
                        "{}:{} declares production dependency `{}` outside {}",
                        manifest.display(),
                        line_index + 1,
                        forbidden_crate.manifest_name,
                        forbidden_crate.owner
                    ));
                }
            }
        }
    }

    assert_eq!(violations, Vec::<String>::new());
}

#[test]
fn resolved_normal_dependency_tree_excludes_non_ssot_runtime_or_replacement_crates() {
    let root = workspace_root();
    let mut violations = Vec::new();
    let cargo_tree_directory = std::env::temp_dir();

    for forbidden_crate in FORBIDDEN_PRODUCTION_CRATES {
        let output = Command::new(cargo_binary())
            // Resolve the standalone git-sourced graph rather than the Atlas
            // overlay. The lockfile audit must be read-only; running from the
            // workspace root lets Cargo discover the overlay and rewrite the
            // lockfile while evaluating the tree.
            .current_dir(&cargo_tree_directory)
            .args([
                "tree",
                "--quiet",
                "--workspace",
                "--edges",
                "normal",
                "-i",
                forbidden_crate.manifest_name,
                "--locked",
                "--manifest-path",
                root.join("Cargo.toml").to_str().unwrap(),
            ])
            .output()
            .unwrap_or_else(|error| {
                panic!(
                    "failed to run cargo tree for `{}` from {}: {error}",
                    forbidden_crate.manifest_name,
                    root.display()
                )
            });

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tree = stdout.trim();

        if output.status.success() {
            if !tree.is_empty() {
                // If atlas_parents are specified, check whether every path in
                // the tree goes through at least one atlas-owned parent.  If
                // all reachability passes through atlas parents, the forbidden
                // crate is within the SSOT chain and not a Coeus violation.
                if !forbidden_crate.atlas_parents.is_empty()
                    && tree_only_reaches_through_atlas_parents(tree, forbidden_crate.atlas_parents)
                {
                    // All paths go through Atlas SSOT parents — not a Coeus
                    // violation; the Atlas crate is responsible for that dep.
                    continue;
                }
                violations.push(format!(
                    "normal dependency tree resolves `{}` outside {}:\n{}",
                    forbidden_crate.manifest_name, forbidden_crate.owner, tree
                ));
            }
            continue;
        }

        if !stderr.contains("did not match any packages") {
            violations.push(format!(
                "cargo tree failed while checking `{}`: {}",
                forbidden_crate.manifest_name,
                stderr.trim()
            ));
        }
    }

    assert_eq!(violations, Vec::<String>::new());
}

/// Return `true` when every path in the inverted `cargo tree -i` output from the
/// forbidden crate to workspace members passes through at least one of the given
/// Atlas-owned parent crate names.
///
/// The `cargo tree -i <crate>` output has the forbidden crate on the first line
/// and its immediate dependents on the subsequent top-level lines (those starting
/// with `└── ` or `├── ` with no leading whitespace).  We check that all those
/// direct dependents are Atlas-owned parents; if every direct dependent is in
/// `atlas_parents`, then no path escapes the Atlas SSOT chain.
fn tree_only_reaches_through_atlas_parents(tree: &str, atlas_parents: &[&str]) -> bool {
    let mut lines = tree.lines();
    // First line is the forbidden crate itself — skip it.
    lines.next();

    for line in lines {
        // Direct dependents start with `└── ` or `├── ` (no leading whitespace).
        let is_direct =
            line.starts_with("└── ") || line.starts_with("├── ") || line.starts_with("|-- ");
        if !is_direct {
            continue;
        }
        // Extract crate name: first word after the tree glyph.
        let rest = line
            .trim_start_matches("└── ")
            .trim_start_matches("├── ")
            .trim_start_matches("|-- ");
        let name = rest.split_whitespace().next().unwrap_or("");
        if !atlas_parents.iter().any(|p| name.starts_with(p)) {
            return false;
        }
    }
    true
}

fn cargo_binary() -> PathBuf {
    std::env::var_os("CARGO")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("cargo"))
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("invariant: coeus-core lives under crates/ beneath the workspace root")
        .to_path_buf()
}

fn workspace_crate_dirs(root: &Path) -> Vec<PathBuf> {
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
        || line.contains(&format!("package = \"{crate_name}\""))
}

fn contains_crate_path(code: &str, crate_name: &str) -> bool {
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

fn imports_crate(code: &str, crate_name: &str) -> bool {
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
