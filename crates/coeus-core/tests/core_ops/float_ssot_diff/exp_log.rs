//! Exponentials and logarithms (0–1 ulp baselines).

use super::support::{
    check_f32, check_f64, edges_f32, edges_f64, log_spread_f32, log_spread_f64, sweep_f32,
    sweep_f64, CoeusFloat, EunomiaFloat, N,
};

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
