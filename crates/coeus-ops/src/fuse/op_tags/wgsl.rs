//! WGSL expression helpers for operations without a native intrinsic.

/// WGSL expression for an Abramowitz-Stegun erf approximation.
///
/// The maximum absolute error is below the existing backend parity tolerance
/// used by WGPU tests, and WGSL has no native `erf` intrinsic.
#[must_use]
pub fn wgsl_erf_approx_expr(arg: &str) -> String {
    let z = format!("({arg})");
    let t = format!("(1.0 / (1.0 + 0.3275911 * abs({z})))");
    format!(
        "(sign({z}) * (1.0 - (((((1.061405429 * {t} - 1.453152027) * {t} + 1.421413741) * {t} - 0.284496736) * {t} + 0.254829592) * {t}) * exp(-({z}) * ({z}))))"
    )
}

/// WGSL expression for exact-contract GELU using `erf(x / sqrt(2))`.
#[must_use]
pub fn wgsl_gelu_expr(arg: &str) -> String {
    let x = format!("({arg})");
    let erf = wgsl_erf_approx_expr(&format!("{x} * 0.7071067811865476"));
    format!("(0.5 * {x} * (1.0 + {erf}))")
}

/// WGSL expression for the exact-contract GELU derivative.
#[must_use]
pub fn wgsl_gelu_grad_expr(arg: &str) -> String {
    let x = format!("({arg})");
    let erf = wgsl_erf_approx_expr(&format!("{x} * 0.7071067811865476"));
    format!("(0.5 * (1.0 + {erf}) + {x} * exp(-0.5 * {x} * {x}) * 0.3989422804014327)")
}
