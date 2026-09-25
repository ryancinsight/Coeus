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
//! Isolated into its own binary anyway because it is measured
//! (2026-09-24) to be the single most memory-dense family in `coeus-ops`:
//! combined with `indexing` and `elementwise` (1881 lines total) it peaked
//! at ~10.8 GiB resident in one `rustc` process, far more than its share of
//! the pre-split single-crate peak (~11.8 GiB for all 7721 lines together)
//! would predict from line count alone. `indexing` and `elementwise` carry
//! no such cross-reference and are split out into `ops_dispatch.rs`. No
//! test was removed or weakened; every leaf module below is unchanged from
//! the prior harness.

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
