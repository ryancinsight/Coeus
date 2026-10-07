//! Binary power baselines, plus the `powi` differential suite.
//!
//! The `powi` tests are `#[ignore]`d: they pin agreement with eunomia's
//! order-aware `powi`, which is still on eunomia's feature branch —
//! member CI resolves providers from git mains, where the old
//! invert-first `powi` (up to 8 ulp off, flushes subnormals, panics on
//! `i32::MIN`) still stands. They compile (no rot), skip by default, and
//! activate by deleting three attributes once the provider lands. Until
//! then the `powf` baselines below carry this module's gate.

use super::support::{
    check_f32, check_f64, log_spread_f32, log_spread_f64, sweep_f32, sweep_f64, CoeusFloat,
    EunomiaFloat,
};

// ── Binary power and integer power ──

#[test]
fn f32_powf() {
    let bases = log_spread_f32(53);
    let exps = sweep_f32(-3.0, 3.0, 40, 54);
    let mut extra = vec![
        (2.0f32, 3.0f32),
        (-2.0, 3.0),
        (-2.0, 2.0),
        (0.0, 0.0),
        (0.0, -1.0),
        (f32::INFINITY, 2.0),
        (f32::NAN, 1.0),
    ];
    for &x in &bases {
        for &y in &exps {
            let a = <f32 as CoeusFloat>::powf(x, y);
            let b = <f32 as EunomiaFloat>::powf(x, y);
            check_f32("powf", x, a, b, 1);
        }
    }
    for (x, y) in extra.drain(..) {
        let a = <f32 as CoeusFloat>::powf(x, y);
        let b = <f32 as EunomiaFloat>::powf(x, y);
        check_f32("powf", x, a, b, 1);
    }
}

#[test]
fn f64_powf() {
    let bases = log_spread_f64(55);
    let exps = sweep_f64(-3.0, 3.0, 40, 56);
    let mut extra = vec![
        (2.0f64, 3.0f64),
        (-2.0, 3.0),
        (-2.0, 2.0),
        (0.0, 0.0),
        (0.0, -1.0),
        (f64::INFINITY, 2.0),
        (f64::NAN, 1.0),
    ];
    for &x in &bases {
        for &y in &exps {
            let a = <f64 as CoeusFloat>::powf(x, y);
            let b = <f64 as EunomiaFloat>::powf(x, y);
            check_f64("powf", x, a, b, 1);
        }
    }
    for (x, y) in extra.drain(..) {
        let a = <f64 as CoeusFloat>::powf(x, y);
        let b = <f64 as EunomiaFloat>::powf(x, y);
        check_f64("powf", x, a, b, 1);
    }
}

#[test]
#[ignore = "pending eunomia order-aware powi on main; see module docs"]
fn f32_powi() {
    // Positive and zero bases agree at every exponent probed (through
    // |n| = 2^31 - 1); negative bases agree through |n| = 130.
    // Negative bases at extreme exponents DIVERGE (see
    // f32_powi_neg_base_extreme_divergence) and are excluded here.
    let pos_bases = [2.0f32, 0.5, 1.5, 10.0, 0.0];
    let mut wide: Vec<i32> = (-10..=10).collect();
    wide.extend([i32::MIN, i32::MAX]);
    for &x in &pos_bases {
        for &n in &wide {
            let a = <f32 as CoeusFloat>::powi(x, n);
            let b = <f32 as EunomiaFloat>::powi(x, n);
            check_f32("powi", x, a, b, 3);
        }
    }
    for &x in &[-2.0f32, -1.5] {
        for n in -130..=130 {
            let a = <f32 as CoeusFloat>::powi(x, n);
            let b = <f32 as EunomiaFloat>::powi(x, n);
            check_f32("powi", x, a, b, 3);
        }
    }
}

#[test]
#[ignore = "pending eunomia order-aware powi on main; see module docs"]
fn f64_powi() {
    let pos_bases = [2.0f64, 0.5, 1.5, 10.0, 0.0];
    let mut wide: Vec<i32> = (-10..=10).collect();
    wide.extend([i32::MIN, i32::MAX]);
    for &x in &pos_bases {
        for &n in &wide {
            let a = <f64 as CoeusFloat>::powi(x, n);
            let b = <f64 as EunomiaFloat>::powi(x, n);
            check_f64("powi", x, a, b, 1);
        }
    }
    for &x in &[-2.0f64, -1.5] {
        for n in -130..=130 {
            let a = <f64 as CoeusFloat>::powi(x, n);
            let b = <f64 as EunomiaFloat>::powi(x, n);
            check_f64("powi", x, a, b, 1);
        }
    }
    // f64 agrees at the extreme exponents too (the f32-only sign fork does
    // not reproduce here); pinned, not carved out.
    for &n in &[i32::MIN, i32::MAX] {
        let a = <f64 as CoeusFloat>::powi(-2.0, n);
        let b = <f64 as EunomiaFloat>::powi(-2.0, n);
        check_f64("powi", -2.0, a, b, 1);
    }
}

/// DOCUMENTED DIVERGENCE (f32 only): for negative bases at extreme odd
/// exponents, std `powi` (coeus's current route) drops the
/// overflow/underflow sign while eunomia preserves it per C/libm
/// (`pow(-2, MAX)` is `-inf`, `pow(-2, MIN+1)` is `-0`). Deleting
/// `Float::powi` flips these edges from std's values to the correct ones.
/// Both sides are pinned so neither drifts silently.
#[test]
#[ignore = "pending eunomia order-aware powi on main; see module docs"]
fn f32_powi_neg_base_extreme_divergence() {
    // (base, exp, coeus/std, eunomia)
    let cases: &[(f32, i32, f32, f32)] = &[
        (-2.0, i32::MAX, f32::INFINITY, f32::NEG_INFINITY),
        (-2.0, i32::MIN + 1, 0.0, -0.0),
        (-1.5, i32::MAX, f32::INFINITY, f32::NEG_INFINITY),
        (-1.5, i32::MIN + 1, 0.0, -0.0),
        (-10.0, i32::MAX, f32::INFINITY, f32::NEG_INFINITY),
        (-10.0, i32::MIN + 1, 0.0, -0.0),
        (-0.5, i32::MAX, 0.0, -0.0),
        (-0.5, i32::MIN + 1, f32::INFINITY, f32::NEG_INFINITY),
    ];
    for &(x, n, coeus, eunomia) in cases {
        assert_eq!(
            <f32 as CoeusFloat>::powi(x, n).to_bits(),
            coeus.to_bits(),
            "coeus powi({x}, {n})"
        );
        assert_eq!(
            <f32 as EunomiaFloat>::powi(x, n).to_bits(),
            eunomia.to_bits(),
            "eunomia powi({x}, {n})"
        );
    }
}
