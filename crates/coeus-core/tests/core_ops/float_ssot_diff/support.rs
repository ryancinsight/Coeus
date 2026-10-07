//! Shared harness for the S3b-Float SSOT differential gate: sampling,
//! ULP comparison, per-type check helpers, and the `diff_f32`/`diff_f64`
//! macros (imported by siblings via `#[macro_use]` on the root module).

pub(crate) use coeus_core::Float as CoeusFloat;
pub(crate) use eunomia::FloatElement as EunomiaFloat;

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

pub(crate) fn sweep_f32(lo: f32, hi: f32, n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Rng(seed);
    (0..n)
        .map(|_| lo + (rng.next_f64() as f32) * (hi - lo))
        .collect()
}

pub(crate) fn sweep_f64(lo: f64, hi: f64, n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    (0..n).map(|_| lo + rng.next_f64() * (hi - lo)).collect()
}

/// Edge values every transcendental sweep covers (callers add domain edges).
pub(crate) fn edges_f32() -> Vec<f32> {
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

pub(crate) fn edges_f64() -> Vec<f64> {
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
pub(crate) fn check_f32(name: &str, x: f32, coeus: f32, eunomia: f32, max_ulp: u32) -> u32 {
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

pub(crate) fn check_f64(name: &str, x: f64, coeus: f64, eunomia: f64, max_ulp: u64) -> u64 {
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

pub(crate) fn log_spread_f32(seed: u64) -> Vec<f32> {
    sweep_f32(-6.0, 6.0, N, seed)
        .into_iter()
        .map(|u| 10.0f32.powf(u))
        .collect()
}

pub(crate) fn log_spread_f64(seed: u64) -> Vec<f64> {
    sweep_f64(-6.0, 6.0, N, seed)
        .into_iter()
        .map(|u| 10.0f64.powf(u))
        .collect()
}

pub(crate) const N: usize = 2000;
