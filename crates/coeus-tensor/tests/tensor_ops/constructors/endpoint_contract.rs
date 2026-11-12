//! Endpoint and interior contract for the native-`T` constructors.
//!
//! Oracle: NumPy 2.5.2. NumPy builds `linspace` as
//! `arange(n) * (delta / div) + start` and then assigns the inclusive
//! endpoint, so the expected values below are reproducible bit for bit by
//! the native-`T` formula once the final sample is taken from `end` rather
//! than from the accumulated step.
//!
//! The `logspace` and `geomspace` interiors are compared with a derived
//! bound. The exponent chain `start + (end - start) * i / (n - 1)` rounds at
//! most four times (subtract, multiply, divide, add) and the power rounds
//! once more, so the relative error stays below `8 * eps` (first-order bound
//! `(1 + eps)^5 - 1 ≈ 5 eps`, rounded up to `8 eps`). The geometric ratio
//! `(end / start)^(1 / (n - 1))` rounds three times (divide, reciprocal
//! divide, power) and the per-sample power and product round once each, so
//! the same bound covers it. NumPy's vectorized power and the platform
//! `powf` may differ by one unit in the last place, which the bound also
//! covers.

use coeus_core::{Float, FloatElement, SequentialBackend};
use coeus_tensor::Tensor;

/// Scale a unit-in-the-last-place bound to the magnitude of `value`.
///
/// The `max(1, |value|)` floor keeps the bound meaningful for samples whose
/// magnitude is below one, where the absolute rounding error is the
/// relevant scale.
fn magnitude_bound<T: Float + FloatElement>(epsilon: T, value: T) -> T {
    let absolute = Float::abs(value);
    let scale = if absolute > T::ONE { absolute } else { T::ONE };
    T::from_count(8) * epsilon * scale
}

fn check_linspace<T: Float + FloatElement>(
    label: &str,
    start: T,
    end: T,
    n: usize,
    expected: &[T],
) {
    let backend = SequentialBackend::new();
    let tensor = Tensor::<T, SequentialBackend>::linspace_on(start, end, n, &backend)
        .expect("invariant: the sample count fits the element range");
    assert_eq!(tensor.as_slice(), expected, "{label}: linspace values");
}

fn check_logspace<T: Float + FloatElement>(
    label: &str,
    start: T,
    end: T,
    n: usize,
    base: T,
    interior: &[T],
    epsilon: T,
) {
    let backend = SequentialBackend::new();
    let tensor = Tensor::<T, SequentialBackend>::logspace_on(start, end, n, base, &backend)
        .expect("invariant: the sample count fits the element range");
    let values = tensor.as_slice();
    assert_eq!(values.len(), n, "{label}: logspace length");
    assert_eq!(
        values[0],
        Float::powf(base, start),
        "{label}: logspace first sample"
    );
    assert_eq!(
        values[n - 1],
        Float::powf(base, end),
        "{label}: logspace last sample"
    );
    for (offset, (&actual, &expected)) in values[1..n - 1].iter().zip(interior).enumerate() {
        let bound = magnitude_bound(epsilon, expected);
        assert!(
            Float::abs(actual - expected) <= bound,
            "{label}: logspace interior[{offset}] = {actual:?} vs {expected:?} (bound {bound:?})"
        );
    }
}

fn check_geomspace_exact<T: Float + FloatElement>(
    label: &str,
    start: T,
    end: T,
    n: usize,
    expected: &[T],
) {
    let backend = SequentialBackend::new();
    let tensor = Tensor::<T, SequentialBackend>::geomspace_on(start, end, n, &backend)
        .expect("invariant: the sample count fits the element range");
    assert_eq!(tensor.as_slice(), expected, "{label}: geomspace values");
}

fn check_geomspace_bounded<T: Float + FloatElement>(
    label: &str,
    start: T,
    end: T,
    n: usize,
    interior: &[T],
    epsilon: T,
) {
    let backend = SequentialBackend::new();
    let tensor = Tensor::<T, SequentialBackend>::geomspace_on(start, end, n, &backend)
        .expect("invariant: the sample count fits the element range");
    let values = tensor.as_slice();
    assert_eq!(values.len(), n, "{label}: geomspace length");
    assert_eq!(values[0], start, "{label}: geomspace first sample");
    assert_eq!(values[n - 1], end, "{label}: geomspace last sample");
    for (offset, (&actual, &expected)) in values[1..n - 1].iter().zip(interior).enumerate() {
        let bound = magnitude_bound(epsilon, expected);
        assert!(
            Float::abs(actual - expected) <= bound,
            "{label}: geomspace interior[{offset}] = {actual:?} vs {expected:?} (bound {bound:?})"
        );
    }
}

#[test]
fn linspace_matches_the_numpy_endpoint_contract() {
    // NumPy 2.5.2 `np.linspace(-9.034271527463753, 6.425485839826166, 8)`.
    // Without the endpoint assignment the final sample is
    // `6.425485839826164`, one unit in the last place below `end`.
    check_linspace(
        "f64",
        -9.034271527463753_f64,
        6.425485839826166_f64,
        8,
        &[
            -9.034271527463753,
            -6.82573476070805,
            -4.617197993952348,
            -2.408661227196646,
            -0.2001244604409429,
            2.00841230631476,
            4.216949073070461,
            6.425485839826166,
        ],
    );
    // NumPy 2.5.2
    // `np.linspace(-0.28572267293930054, 0.3586343228816986, 9, dtype=float32)`.
    check_linspace(
        "f32",
        -0.28572267_f32,
        0.35863432_f32,
        9,
        &[
            -0.28572267,
            -0.20517805,
            -0.12463343,
            -0.04408881,
            0.03645581,
            0.11700043,
            0.19754505,
            0.2780897,
            0.35863432,
        ],
    );
}

#[test]
fn logspace_endpoints_come_from_the_endpoint_exponents() {
    // NumPy 2.5.2 `np.logspace(1.3906814054416197, -3.4938357597647607, 4,
    // base=10.0)`. The interpolated exponent at the last sample is
    // `-3.493835759764761`, one unit in the last place away from `end`, so
    // the unpinned sample is `0.0003207482091501903` instead of
    // `0.00032074820915019064`.
    check_logspace(
        "f64",
        1.3906814054416197_f64,
        -3.4938357597647607_f64,
        4,
        10.0_f64,
        &[0.5787740050200676, 0.013625003692214912],
        f64::EPSILON,
    );
    // NumPy 2.5.2 `np.logspace(-0.7692398428916931, 2.2106728553771973, 8,
    // base=10.0, dtype=float32)`. The unpinned last sample is
    // `162.43255615234375` instead of `162.43247985839844`.
    check_logspace(
        "f32",
        -0.76923984_f32,
        2.2106729_f32,
        8,
        10.0_f32,
        &[
            0.4533796, 1.2082695, 3.2200723, 8.581584, 22.870157, 60.94962,
        ],
        f32::EPSILON,
    );
}

#[test]
fn geomspace_endpoints_come_from_the_endpoints() {
    // A power-of-two ratio is exact in every binary float format: the
    // uniform ratio `(16 / 1)^(1 / 4)` is `2`, so every interior sample is
    // exactly representable.
    check_geomspace_exact("f64", 1.0_f64, 16.0_f64, 5, &[1.0, 2.0, 4.0, 8.0, 16.0]);
    check_geomspace_exact("f32", 1.0_f32, 16.0_f32, 5, &[1.0, 2.0, 4.0, 8.0, 16.0]);
    check_geomspace_exact(
        "f64 negative",
        -1.0_f64,
        -16.0_f64,
        5,
        &[-1.0, -2.0, -4.0, -8.0, -16.0],
    );
    // NumPy 2.5.2 `np.geomspace(1.0, 10.0, 4)`: the exact progression is
    // `1, 10^(1/3), 10^(2/3), 10`. The unpinned last sample is
    // `10.000000000000002`, one unit in the last place above `end`.
    check_geomspace_bounded(
        "f64",
        1.0_f64,
        10.0_f64,
        4,
        &[2.154434690031884, 4.641588833612778],
        f64::EPSILON,
    );
}

#[test]
fn linspace_integer_step_truncates_and_still_reaches_the_endpoint() {
    let backend = SequentialBackend::new();
    let tensor = Tensor::<i32, SequentialBackend>::linspace_on(0, 10, 4, &backend)
        .expect("invariant: the sample count fits the element range");
    assert_eq!(
        tensor.as_slice(),
        &[0, 3, 6, 10],
        "the truncated step advances the interior samples and the endpoint stays exact"
    );
}
