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

use coeus_core::Float as CoeusFloat;
use eunomia::FloatElement as EunomiaFloat;

/// Deterministic xorshift64* — no `rand` dependency for a conformance sweep.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f64(&mut self) -> f64 {
        const DIV: f64 = (1u64 << 53) as f64;
        ((self.next_u64() >> 11) as f64) / DIV
    }
}

fn sweep_f32(lo: f32, hi: f32, n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| lo + (rng.next_f64() as f32) * (hi - lo))
        .collect()
}

fn sweep_f64(lo: f64, hi: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    (0..n).map(|_| lo + rng.next_f64() * (hi - lo)).collect()
}

/// Edge values every transcendental sweep covers (callers add domain edges).
fn edges_f32() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        10.0,
        100.0,
        1.0e10,
        1.0e-10,
        f32::MIN_POSITIVE,
        f32::MIN,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        core::f32::consts::PI,
        core::f32::consts::E,
    ]
}

fn edges_f64() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        -2.0,
        10.0,
        100.0,
        1.0e10,
        1.0e-10,
        f64::MIN_POSITIVE,
        f64::MIN,
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        core::f64::consts::PI,
        core::f64::consts::E,
    ]
}

/// Order-preserving integer map, so ULP distance is a subtraction.
fn ordered_f32(x: f32) -> u32 {
    let bits = x.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits ^ 0x8000_0000
    } else {
        !bits
    }
}

fn ordered_f64(x: f64) -> u64 {
    let bits = x.to_bits();
    if bits & 0x8000_0000_0000_0000 == 0 {
        bits ^ 0x8000_0000_0000_0000
    } else {
        !bits
    }
}

/// Assert class agreement + ULP tolerance; returns the observed ULP distance.
fn check_f32(name: &str, x: f32, coeus: f32, eunomia: f32, max_ulp: u32) -> u32 {
    if coeus.is_nan() || eunomia.is_nan() {
        assert!(
            coeus.is_nan() && eunomia.is_nan(),
            "{name}({x}): NaN-class mismatch: coeus={coeus}, eunomia={eunomia}"
        );
        return 0;
    }
    if coeus.is_infinite() || eunomia.is_infinite() {
        assert_eq!(
            coeus, eunomia,
            "{name}({x}): inf-class mismatch: coeus={coeus}, eunomia={eunomia}"
        );
        return 0;
    }
    let ulp = ordered_f32(coeus).abs_diff(ordered_f32(eunomia));
    assert!(
        ulp <= max_ulp,
        "{name}({x}): {ulp} ulp over tolerance {max_ulp}: coeus={coeus} eunomia={eunomia}"
    );
    ulp
}

fn check_f64(name: &str, x: f64, coeus: f64, eunomia: f64, max_ulp: u64) -> u64 {
    if coeus.is_nan() || eunomia.is_nan() {
        assert!(
            coeus.is_nan() && eunomia.is_nan(),
            "{name}({x}): NaN-class mismatch: coeus={coeus}, eunomia={eunomia}"
        );
        return 0;
    }
    if coeus.is_infinite() || eunomia.is_infinite() {
        assert_eq!(
            coeus, eunomia,
            "{name}({x}): inf-class mismatch: coeus={coeus}, eunomia={eunomia}"
        );
        return 0;
    }
    let ulp = ordered_f64(coeus).abs_diff(ordered_f64(eunomia));
    assert!(
        ulp <= max_ulp,
        "{name}({x}): {ulp} ulp over tolerance {max_ulp}: coeus={coeus} eunomia={eunomia}"
    );
    ulp
}

macro_rules! diff_f32 {
    ($test:ident, $m:ident, $samples:expr, $tol:expr) => {
        #[test]
        fn $test() {
            for x in $samples {
                let a = <f32 as CoeusFloat>::$m(x);
                let b = <f32 as EunomiaFloat>::$m(x);
                check_f32(stringify!($m), x, a, b, $tol);
            }
        }
    };
}

macro_rules! diff_f64 {
    ($test:ident, $m:ident, $samples:expr, $tol:expr) => {
        #[test]
        fn $test() {
            for x in $samples {
                let a = <f64 as CoeusFloat>::$m(x);
                let b = <f64 as EunomiaFloat>::$m(x);
                check_f64(stringify!($m), x, a, b, $tol);
            }
        }
    };
}

// Tolerances below are measured baselines: exact ops bitwise 0, every other
// same-name function 0–1 ulp over edges + 2000-sample sweeps — except
// `powi`, where rounding-order growth reaches 3 ulp (f32) / 1 ulp (f64)
// at |n| ≤ 130, and negative f32 bases at extreme odd exponents diverge in
// overflow/underflow SIGN (pinned in
// f32_powi_neg_base_extreme_divergence: eunomia preserves the C/libm sign,
// std drops it). Deleting `Float::powi` flips those f32 edges to the
// correct values — an accepted, recorded behavior change, not a blocker.

const N: usize = 2000;

// ── Exact ops: expect bitwise ──

diff_f32!(
    f32_floor,
    floor,
    edges_f32()
        .into_iter()
        .chain(sweep_f32(-1.0e6, 1.0e6, N, 11)),
    0
);
diff_f64!(
    f64_floor,
    floor,
    edges_f64()
        .into_iter()
        .chain(sweep_f64(-1.0e6, 1.0e6, N, 12)),
    0
);
diff_f32!(
    f32_ceil,
    ceil,
    edges_f32()
        .into_iter()
        .chain(sweep_f32(-1.0e6, 1.0e6, N, 13)),
    0
);
diff_f64!(
    f64_ceil,
    ceil,
    edges_f64()
        .into_iter()
        .chain(sweep_f64(-1.0e6, 1.0e6, N, 14)),
    0
);
diff_f32!(
    f32_round_ties_away,
    round,
    edges_f32()
        .into_iter()
        .chain(sweep_f32(-1.0e6, 1.0e6, N, 15)),
    0
);
diff_f64!(
    f64_round_ties_away,
    round,
    edges_f64()
        .into_iter()
        .chain(sweep_f64(-1.0e6, 1.0e6, N, 16)),
    0
);
diff_f32!(
    f32_trunc,
    trunc,
    edges_f32()
        .into_iter()
        .chain(sweep_f32(-1.0e6, 1.0e6, N, 17)),
    0
);
diff_f64!(
    f64_trunc,
    trunc,
    edges_f64()
        .into_iter()
        .chain(sweep_f64(-1.0e6, 1.0e6, N, 18)),
    0
);
diff_f32!(
    f32_signum,
    signum,
    edges_f32()
        .into_iter()
        .chain(sweep_f32(-1.0e6, 1.0e6, N, 19)),
    0
);
diff_f64!(
    f64_signum,
    signum,
    edges_f64()
        .into_iter()
        .chain(sweep_f64(-1.0e6, 1.0e6, N, 20)),
    0
);

// ── sqrt: NumericElement on the eunomia side ──

#[test]
fn f32_sqrt() {
    let samples = edges_f32()
        .into_iter()
        .chain(sweep_f32(0.0, 1.0e6, N, 21))
        .chain(sweep_f32(-100.0, 0.0, 200, 22));
    for x in samples {
        let a = <f32 as CoeusFloat>::sqrt(x);
        let b = <f32 as eunomia::NumericElement>::sqrt(x);
        check_f32("sqrt", x, a, b, 0);
    }
}

#[test]
fn f64_sqrt() {
    let samples = edges_f64()
        .into_iter()
        .chain(sweep_f64(0.0, 1.0e6, N, 23))
        .chain(sweep_f64(-100.0, 0.0, 200, 24));
    for x in samples {
        let a = <f64 as CoeusFloat>::sqrt(x);
        let b = <f64 as eunomia::NumericElement>::sqrt(x);
        check_f64("sqrt", x, a, b, 0);
    }
}

// ── Exponentials and logarithms ──

diff_f32!(
    f32_exp,
    exp,
    edges_f32().into_iter().chain(sweep_f32(-80.0, 80.0, N, 25)),
    1
);
diff_f64!(
    f64_exp,
    exp,
    edges_f64()
        .into_iter()
        .chain(sweep_f64(-700.0, 700.0, N, 26)),
    1
);
diff_f32!(
    f32_exp2,
    exp2,
    edges_f32()
        .into_iter()
        .chain(sweep_f32(-120.0, 120.0, N, 27)),
    1
);
diff_f64!(
    f64_exp2,
    exp2,
    edges_f64()
        .into_iter()
        .chain(sweep_f64(-1000.0, 1000.0, N, 28)),
    0
);

fn log_spread_f32(seed: u64) -> Vec<f32> {
    sweep_f32(-6.0, 6.0, N, seed)
        .into_iter()
        .map(|u| 10.0f32.powf(u))
        .collect()
}

fn log_spread_f64(seed: u64) -> Vec<f64> {
    sweep_f64(-6.0, 6.0, N, seed)
        .into_iter()
        .map(|u| 10.0f64.powf(u))
        .collect()
}

diff_f32!(
    f32_ln,
    ln,
    edges_f32().into_iter().chain(log_spread_f32(29)),
    1
);
diff_f64!(
    f64_ln,
    ln,
    edges_f64().into_iter().chain(log_spread_f64(30)),
    1
);
diff_f32!(
    f32_log2,
    log2,
    edges_f32().into_iter().chain(log_spread_f32(31)),
    1
);
diff_f64!(
    f64_log2,
    log2,
    edges_f64().into_iter().chain(log_spread_f64(32)),
    1
);
diff_f32!(
    f32_log10,
    log10,
    edges_f32().into_iter().chain(log_spread_f32(33)),
    1
);
diff_f64!(
    f64_log10,
    log10,
    edges_f64().into_iter().chain(log_spread_f64(34)),
    1
);

// ── Trigonometry (bounded arguments; large-argument range reduction is
// implementation-defined and excluded by design) ──

diff_f32!(
    f32_sin,
    sin,
    edges_f32()
        .into_iter()
        .chain(sweep_f32(-12.57, 12.57, N, 35)),
    1
);
diff_f64!(
    f64_sin,
    sin,
    edges_f64()
        .into_iter()
        .chain(sweep_f64(-12.57, 12.57, N, 36)),
    1
);
diff_f32!(
    f32_cos,
    cos,
    edges_f32()
        .into_iter()
        .chain(sweep_f32(-12.57, 12.57, N, 37)),
    1
);
diff_f64!(
    f64_cos,
    cos,
    edges_f64()
        .into_iter()
        .chain(sweep_f64(-12.57, 12.57, N, 38)),
    1
);
diff_f32!(
    f32_tan,
    tan,
    edges_f32()
        .into_iter()
        .chain(sweep_f32(-12.57, 12.57, N, 39)),
    1
);
diff_f64!(
    f64_tan,
    tan,
    edges_f64()
        .into_iter()
        .chain(sweep_f64(-12.57, 12.57, N, 40)),
    1
);
diff_f32!(
    f32_asin,
    asin,
    edges_f32().into_iter().chain(sweep_f32(-1.0, 1.0, N, 41)),
    1
);
diff_f64!(
    f64_asin,
    asin,
    edges_f64().into_iter().chain(sweep_f64(-1.0, 1.0, N, 42)),
    1
);
diff_f32!(
    f32_acos,
    acos,
    edges_f32().into_iter().chain(sweep_f32(-1.0, 1.0, N, 43)),
    1
);
diff_f64!(
    f64_acos,
    acos,
    edges_f64().into_iter().chain(sweep_f64(-1.0, 1.0, N, 44)),
    1
);
diff_f32!(
    f32_atan,
    atan,
    edges_f32()
        .into_iter()
        .chain(sweep_f32(-1.0e3, 1.0e3, N, 45)),
    1
);
diff_f64!(
    f64_atan,
    atan,
    edges_f64()
        .into_iter()
        .chain(sweep_f64(-1.0e3, 1.0e3, N, 46)),
    0
);
diff_f32!(
    f32_sinh,
    sinh,
    edges_f32().into_iter().chain(sweep_f32(-80.0, 80.0, N, 47)),
    1
);
diff_f64!(
    f64_sinh,
    sinh,
    edges_f64()
        .into_iter()
        .chain(sweep_f64(-700.0, 700.0, N, 48)),
    1
);
diff_f32!(
    f32_cosh,
    cosh,
    edges_f32().into_iter().chain(sweep_f32(-80.0, 80.0, N, 49)),
    1
);
diff_f64!(
    f64_cosh,
    cosh,
    edges_f64()
        .into_iter()
        .chain(sweep_f64(-700.0, 700.0, N, 50)),
    1
);
diff_f32!(
    f32_tanh,
    tanh,
    edges_f32().into_iter().chain(sweep_f32(-80.0, 80.0, N, 51)),
    1
);
diff_f64!(
    f64_tanh,
    tanh,
    edges_f64()
        .into_iter()
        .chain(sweep_f64(-700.0, 700.0, N, 52)),
    1
);

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
