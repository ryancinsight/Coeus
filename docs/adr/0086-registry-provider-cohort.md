# ADR 0086: Coeus 0.13.0 registry provider cohort

## Status

Accepted for the Atlas Aequitas 0.4 migration and Coeus registry release
on 2026-11-19.

## Context

Published Coeus 0.12.0 packages expose Eunomia 0.9 and Leto 0.45 types.
Rust treats types from incompatible versions as distinct. A consumer that
exchanges scalars or array storage with these packages cannot adopt
Eunomia 0.12 and Leto 0.51 by changing its own lockfile alone.

## Decision

Publish Core, CPU, Tensor, Sparse, Ops, Autograd, NN, and Optim at 0.13.0.
Keep the Rust workspace and internal version requirements aligned. Python
and the other Rust packages are outside this registry release.

An alias or adapter could hide some version conflicts at local call sites.
It would leave two nominal type families in the registry dependency graph.
The coordinated version change makes the shared provider requirements
explicit at each package boundary.

Keep Autograd's NN test dependency workspace-local. NN depends on Autograd
at runtime, so including the reverse dev edge in the normalized manifest
creates a registry bootstrap cycle. Workspace integration tests continue
to use the real NN implementation.

## Verification

Confirm each archive resolves against published dependencies on its
declared Rust version. Verify Core before the first CPU upload, then
publish Tensor, Sparse and Ops, Autograd, and NN and Optim in dependency
order. Check registry checksums and the release revision. Run a fresh
registry consumer that crosses the Leto, CPU backend, and Autograd
boundaries and asserts its computed values.

Core and CPU are published from the #681 merge. Tensor, Sparse, Ops,
Autograd, NN, and Optim are published from the #683 merge using the
archived-lock API checks. All eight archives match their crates.io
SHA-256 checksums and source revisions. A fresh registry-only Rust 1.95
consumer verifies a typed Leto view, rejects a short input buffer, and
obtains the exact derivative 6 for x² at x = 3.
[Coeus #681](https://github.com/ryancinsight/coeus/pull/681)
and [Coeus release run](https://github.com/ryancinsight/coeus/actions/runs/44660251129).
