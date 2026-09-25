//! Integration harness for `coeus-ops` output-ownership contracts across
//! operation families and the CPU backend.
//!
//! Split out of the former `tests/ops.rs` (see `ops_numeric.rs` for the
//! general measured compile-memory rationale). Kept as a single compile
//! unit -- unlike the other families, these leaf modules cross-reference
//! each other's helpers directly (`accumulated_outputs.rs` uses
//! `super::optimizer`, `cpu/state_updates.rs` uses
//! `super::accumulated_outputs`), so splitting them across separate
//! integration-test binaries does not compile: each top-level `tests/*.rs`
//! file is its own crate and `super::` cannot cross that boundary.
//! Isolated into its own binary anyway because it is measured to be the
//! single most memory-dense family in `coeus-ops`. `indexing` and
//! `elementwise` carry no such cross-reference and are split out into
//! `ops_dispatch.rs`.
//!
//! `device_outputs.rs`'s `preserves_output_clones`/`rejects_invalid_output_write`
//! are each instantiated once per (scalar type, backend, `OutputWrite` impl)
//! triple -- required generic-instantiation coverage, not a redundancy --
//! and *reducing the number of `OutputWrite` impls* is the lever this file
//! now pulls twice:
//!
//! 1. (2026-09-25) The index-arithmetic and assertion logic that does not
//!    vary per triple was hoisted into plain/`T`-only helper functions so
//!    it monomorphizes once instead of once per triple.
//! 2. (2026-09-25) `Add`, `Sum`, and `Product` needed no per-type bound
//!    beyond `Scalar` (unlike `Negate`, which needs `Neg<Output = T>`, and
//!    `Square`, which needs `Float`), so they were folded into one runtime
//!    `NumericOp` enum -- the same pattern `Scan` already used -- cutting
//!    their combined instantiation count roughly 3x wherever a `(T, B)`
//!    pair is shared across the three (mechanically confirmed: 144
//!    `NumericOp::`-value call sites now compile through two
//!    `OutputWrite<T, B>` impls, `Negate` and `NumericOp`, instead of four).
//!
//! Measured before/after both changes on this host (single-package
//! `--locked --tests` builds, pinned 1.97.0, shared `CARGO_TARGET_DIR`
//! under variable multi-agent load): peak `rustc` RSS for this file alone
//! ranged 11.5-11.8 GiB before and 11.6 GiB after one post-merge
//! measurement -- within this host's demonstrated run-to-run noise (single
//! runs of the *unchanged* pre-merge file spanned 7.5-12.9 GiB across
//! samples taken minutes apart), so this is not proof the merge reduced
//! peak RSS, only that it did not obviously fail to. The instantiation
//! count reduction is real and mechanically verified regardless of what
//! peak RSS sampling on a contended host can resolve; the remaining cost
//! is the (type x backend x operation) surface itself, which cannot shrink
//! further without dropping shipped coverage or accepting the same
//! per-type-bound restructuring risk `Negate`/`Square` were kept out of.
//! No test was removed or weakened; every leaf module below is unchanged
//! from the prior harness, and the full source-level `#[test]` name set is
//! identical before and after every edit made to this harness.
//!
//! Memory safety on CI now rests on this file compiling in isolation from
//! its (much lighter) `ops_numeric`/`ops_structural`/`ops_dispatch`
//! siblings rather than on a `CARGO_BUILD_JOBS` throttle: a prior revision
//! bounded `CARGO_BUILD_JOBS` to contain this file's peak, which serialized
//! the *entire* cargo invocation (every dependency crate built alongside
//! it, not just these four binaries) and pushed the WGPU, CUDA, and
//! workspace-wide Tests CI jobs from their normal 7-24 minute range past
//! their 45-minute timeout with no compiler progress for the whole window.
//! The throttle was removed for that reason; if a future CI run shows
//! renewed OOM under normal parallelism, the fix is to shrink this file's
//! own footprint further (e.g. reducing the backend or type list actually
//! shipped, or restructuring the dispatch to avoid re-deriving the full
//! `Tensor<T, B>` surface per operation) rather than reintroducing a
//! workspace-wide throttle.

#[path = "ops/ownership/accumulated_outputs.rs"]
mod accumulated_outputs;
#[path = "ops/ownership/cpu/device_outputs.rs"]
mod cpu_device_outputs;
#[path = "ops/ownership/cpu/state_updates.rs"]
mod cpu_state_updates;
#[path = "ops/ownership/device_outputs.rs"]
mod device_outputs;
#[path = "ops/ownership/optimizer.rs"]
mod optimizer;
#[path = "ops/ownership/staggered.rs"]
mod staggered;
