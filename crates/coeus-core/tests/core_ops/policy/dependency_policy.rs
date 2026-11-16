//! Dependency-policy contract tests.
//!
//! The audit has four parts, each in its own leaf module: the forbidden-crate
//! table with the source and manifest audits over it, the lockfile-state
//! detection, the resolved-graph reachability audit, and the workspace
//! discovery helpers they share.
#![expect(
    clippy::unwrap_used,
    reason = "test assertions surface failures immediately by design"
)]

#[path = "dependency_policy/dependency_tree.rs"]
mod dependency_tree;
#[path = "dependency_policy/forbidden.rs"]
mod forbidden;
#[path = "dependency_policy/lockfile.rs"]
mod lockfile;
#[path = "dependency_policy/workspace.rs"]
mod workspace;
