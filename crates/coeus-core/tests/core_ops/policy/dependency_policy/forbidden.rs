//! The forbidden-crate table and the source and manifest audits over it.

use super::workspace::{
    cargo_manifests, contains_crate_path, dependency_line_names_crate, imports_crate,
    is_production_dependency_section, rust_files, workspace_crate_dirs, workspace_root,
};
use std::fs;

pub(super) struct ForbiddenProductionCrate {
    pub(super) manifest_name: &'static str,
    pub(super) import_name: &'static str,
    pub(super) owner: &'static str,
    /// Atlas-owned SSOT replacement crates through which the forbidden crate
    /// may legitimately reach Coeus. Path-dependent Atlas crates are the
    /// Atlas's responsibility; Coeus only forbids reachability that escapes
    /// the Atlas replacement stack.
    pub(super) atlas_parents: &'static [&'static str],
}

pub(super) const FORBIDDEN_PRODUCTION_CRATES: [ForbiddenProductionCrate; 8] = [
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
        // Moirai 0.11's transport stack reaches tokio through `quinn`, its
        // QUIC implementation. `quinn` is therefore the only direct dependent
        // and the whole path stays inside the Moirai-owned async/transport
        // SSOT; no Coeus crate depends on tokio directly.
        atlas_parents: &["quinn"],
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
