//! Trigonometry (bounded arguments; 0–1 ulp baselines).

use super::support::{
    check_f32, check_f64, edges_f32, edges_f64, sweep_f32, sweep_f64, CoeusFloat, EunomiaFloat, N,
};

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
