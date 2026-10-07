//! Exact ops (bitwise 0) plus `sqrt` (eunomia side: `NumericElement`).

use super::support::{
    check_f32, check_f64, edges_f32, edges_f64, sweep_f32, sweep_f64, CoeusFloat, EunomiaFloat, N,
};

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
