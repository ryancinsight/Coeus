# ADR 0021: Place workspace crates under `crates/`

- Status: Accepted
- Date: 2026-07-24
- Scope: Coeus workspace topology, Cargo manifests, release workflow paths,
  and repository documentation

## Context

Coeus currently stores all 13 workspace crate directories at the repository
root. The workspace has no package or API requirement for those physical
locations, and sibling Atlas workspaces use `crates/` as the canonical
workspace-member boundary. The flat layout adds package directories to the
repository root and makes the topology less consistent across the stack.

## Decision

Move every Coeus workspace crate into `crates/` with `git mv`. Update the root
workspace member list, root workspace-local dependency paths, release workflow
manifest paths, and repository documentation that names crate filesystem
paths. Preserve each package name, version, manifest contents, dependency
direction, and public API. Keep crate-local sibling path dependencies relative
to their new common directory; verify them through Cargo metadata rather than
assuming their validity.

The migration is one mechanical change. It does not rename packages, change
the Rust edition or resolver, alter dependency versions, or introduce path
aliases or compatibility wrappers. Consumers that refer to published Cargo
package identities remain unchanged; scripts that hard-code repository paths
must follow the moved manifests.

## Rejected alternatives

- Keep the flat layout: preserves historical placement but leaves Coeus as an
  outlier and keeps the repository root noisy.
- Add a second path or compatibility directory: duplicates the workspace
  topology and creates two sources of truth for each crate.
- Move only selected crates: leaves the workspace inconsistent and requires
  mixed path conventions.

## Consequences

The root becomes a repository boundary containing `crates/`, documentation,
tests, and tooling. Git history is preserved as directory renames. Cargo
package identities and Rust APIs do not change. Repository-local manifest paths
and release automation paths change and are verified in the same increment.

## Verification

Run `cargo metadata --locked --no-deps --format-version 1` and assert every
workspace manifest is under `crates/`. Scan tracked text for stale
`coeus-*/` repository paths, run the configured formatting/check/test/doctest/
documentation gates that fit the exact revision, and run package validation
for the moved Python manifest. Review the staged diff to confirm only the
directory relocation, path consumers, and synchronized architecture records
are included.
