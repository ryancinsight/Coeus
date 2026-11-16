//! Lockfile-state detection shared by the dependency reachability audit.

use std::fs;
use std::path::Path;
use std::process::Command;

/// Return `true` when `cargo tree` refused to run because the lockfile is not
/// the resolution for the manifests it was given.
///
/// Cargo prints this when `--locked` is passed and the lock would have to be
/// rewritten. Two environments produce it, and neither is a dependency-policy
/// violation:
///
/// - the committed lock predates a manifest change (a missing package, e.g. an
///   optional dependency a newly enabled feature activates), and
/// - a local build under the Atlas overlay rewrote the lockfile (dropped
///   `source` lines, `[[patch.unused]]` entries), which makes it unusable
///   outside the overlay.
///
/// The reachability audit needs a lock Cargo accepts, so it cannot answer in
/// either state. Treating the refusal as a violation would report the lockfile
/// environment as a dependency-policy breach; treating it as a pass would hide
/// one. The caller reports it as "unavailable" instead — which is why CI, where
/// the overlay does not apply and the lock is the committed one, is the
/// authoritative run.
pub(super) fn is_lockfile_rewrite_refusal(stderr: &str) -> bool {
    stderr.contains("cannot update the lock file") && stderr.contains("--locked was passed")
}

/// Return `true` when the working `Cargo.lock` is not the committed one.
///
/// The audit's premise is the committed, standalone lock: CI resolves the real
/// git-sourced graph from it. A local build under the Atlas overlay rewrites the
/// working lock (patched crates become path packages, sometimes at a different
/// version), and resolving *that* file from outside the overlay is what Cargo
/// refuses — so the run would report a resolution error that says nothing about
/// Coeus's dependency graph. Comparing against the committed bytes is the exact
/// signal, where looking for a rewrite symptom in the lock's text is a guess.
///
/// `None` means "could not tell" (no git, or the file is untracked): the audit
/// then runs and any failure is treated as a violation.
pub(super) fn lockfile_differs_from_committed(root: &Path) -> Option<bool> {
    let committed = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["show", "HEAD:Cargo.lock"])
        .output()
        .ok()?;
    if !committed.status.success() {
        return None;
    }
    let working = fs::read(root.join("Cargo.lock")).ok()?;
    // The checkout may be CRLF while the blob is LF; that is not a rewrite.
    Some(normalize_newlines(&working) != normalize_newlines(&committed.stdout))
}

/// Compare lock contents ignoring the checkout's line endings.
fn normalize_newlines(text: &[u8]) -> Vec<u8> {
    text.iter().copied().filter(|byte| *byte != b'\r').collect()
}

#[test]
fn lockfile_rewrite_refusal_is_recognised() {
    // Verbatim Cargo output, as captured from this workspace under the overlay.
    let refusal = "error: cannot update the lock file D:\\atlas\\repos\\coeus\\Cargo.lock \
                   because --locked was passed to prevent this\n\
                   help: to generate the lock file without accessing the network, \
                   remove the --locked flag and use --offline instead.\n";
    assert!(is_lockfile_rewrite_refusal(refusal));
}

#[test]
fn lock_newline_style_is_not_a_rewrite() {
    assert_eq!(
        normalize_newlines(b"a = 1\r\nb = 2\r\n"),
        normalize_newlines(b"a = 1\nb = 2\n"),
        "a CRLF checkout must compare equal to its LF blob"
    );
    assert_ne!(
        normalize_newlines(b"a = 1\n"),
        normalize_newlines(b"a = 2\n"),
        "an actual content change must still be seen"
    );
}

#[test]
fn unrelated_cargo_failures_are_not_treated_as_a_lockfile_refusal() {
    // The absence of a forbidden crate is the pass case, not a refusal.
    assert!(!is_lockfile_rewrite_refusal(
        "error: package ID specification `pollster` did not match any packages\n"
    ));
    // A toolchain or network failure must still surface as a violation.
    assert!(!is_lockfile_rewrite_refusal(
        "error: failed to fetch `https://github.com/ryancinsight/leto.git`\n"
    ));
}
