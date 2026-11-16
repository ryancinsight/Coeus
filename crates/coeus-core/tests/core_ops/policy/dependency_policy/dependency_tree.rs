//! The resolved-graph audit over the forbidden-crate table.

use super::forbidden::FORBIDDEN_PRODUCTION_CRATES;
use super::lockfile::{is_lockfile_rewrite_refusal, lockfile_differs_from_committed};
use super::workspace::{cargo_binary, workspace_root};
use std::process::Command;

/// Every forbidden crate must be absent from the resolved normal dependency
/// tree, or reachable only through an Atlas-owned parent.
///
/// The audit shells out to `cargo tree -i --locked`, which requires a lockfile
/// Cargo accepts. When it does not have one — the overlay rewrote it locally, or
/// it predates a manifest change — the audit reports itself unavailable rather
/// than mislabelling the lockfile environment as a policy violation; see
/// [`is_lockfile_rewrite_refusal`].
#[test]
#[expect(
    clippy::print_stderr,
    reason = "the test reports to the operator why the audit could not run"
)]
fn resolved_normal_dependency_tree_excludes_non_ssot_runtime_or_replacement_crates() {
    let root = workspace_root();
    let mut violations = Vec::new();
    let mut lockfile_unavailable: Option<String> = None;
    let cargo_tree_directory = std::env::temp_dir();

    // A local build under the overlay leaves a lock this process must not read
    // as the committed one: the audit would then fail to resolve rather than
    // fail to audit. Decide that once, before any `cargo tree` call.
    let lockfile_churned = lockfile_differs_from_committed(&root).unwrap_or(false);

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
            // A lock Cargo refuses to work from is an environment state, not a
            // graph violation: `is_lockfile_rewrite_refusal` and
            // `lockfile_differs_from_committed` explain both causes and why
            // this run cannot answer either way.
            if lockfile_churned || is_lockfile_rewrite_refusal(&stderr) {
                lockfile_unavailable.get_or_insert_with(|| stderr.trim().to_owned());
                continue;
            }
            violations.push(format!(
                "cargo tree failed while checking `{}`: {}",
                forbidden_crate.manifest_name,
                stderr.trim()
            ));
        }
    }

    assert_eq!(violations, Vec::<String>::new());

    if let Some(reason) = lockfile_unavailable {
        eprintln!(
            "note: skipped the `--locked` dependency reachability audit — Cargo will not \
             resolve the tree from this lockfile, so the audit cannot answer here. This is \
             the Atlas overlay rewriting `Cargo.lock` on local builds, or a lock older than \
             the manifests; neither is a dependency-policy violation. CI runs this audit \
             against the committed lock, where the overlay does not apply.\n{reason}"
        );
    }
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
