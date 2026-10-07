//! Remainder of the S3b-Float deletion gate: the `Float` redeclarations it
//! measured are deleted (exact ops were bitwise 0, the rest 0–1 ulp —
//! envelope recorded in ADR 0069), leaving only the staged `powi`
//! differential below. The `powi` tests ship `#[ignore]`d until eunomia's
//! order-aware `powi` lands on its main (member CI resolves providers
//! from git mains); see `power.rs`, which activates by deleting three
//! attributes.

#[path = "float_ssot_diff/power.rs"]
mod power;
#[path = "float_ssot_diff/support.rs"]
mod support;
