//! S3b-Float deletion gate: `coeus_core::Float` vs the eunomia SSOT.
//!
//! Every same-name transcendental on `Float` routes (for f32/f64) to the
//! std inherent (system libm), while `eunomia::FloatElement` routes through
//! the `libm` crate (pure Rust) — f32 directly, f64 via native
//! double-precision overrides. Deleting the `Float` redeclarations is only
//! behavior-preserving if both routes agree, so each test below pins their
//! agreement over edges plus a deterministic sweep:
//!
//! - NaN agrees with NaN (payload ignored), ±inf agrees with same-sign ±inf.
//! - Finite pairs must be within the per-test ULP tolerance.
//! - A test whose measured worst case exceeds 2 ulp names a function BLOCKED
//!   for deletion until the provider seam is reconciled (see ADR 0069).
//!
//! `round` here is ties-away on both sides (std `round` vs `libm::roundf`);
//! the ties-even `CpuUnaryOp::Round` arm routes via `f64::round_ties_even`
//! and is out of scope. F16/Bf16 are out of scope: their `Float` impls
//! already delegate to eunomia, so deletion is trivially safe for them.

// Tolerances below are measured baselines: exact ops bitwise 0, every other
// same-name function 0–1 ulp over edges + 2000-sample sweeps — except
// `powi`, where rounding-order growth reaches 3 ulp (f32) / 1 ulp (f64)
// at |n| ≤ 130, and negative f32 bases at extreme odd exponents diverge in
// overflow/underflow SIGN (pinned in
// f32_powi_neg_base_extreme_divergence: eunomia preserves the C/libm sign,
// std drops it). Deleting `Float::powi` flips those f32 edges to the
// correct values — an accepted, recorded behavior change, not a blocker.

#[macro_use]
#[path = "float_ssot_diff/support.rs"]
mod support;
#[path = "float_ssot_diff/exact.rs"]
mod exact;
#[path = "float_ssot_diff/exp_log.rs"]
mod exp_log;
#[path = "float_ssot_diff/power.rs"]
mod power;
#[path = "float_ssot_diff/trig.rs"]
mod trig;
