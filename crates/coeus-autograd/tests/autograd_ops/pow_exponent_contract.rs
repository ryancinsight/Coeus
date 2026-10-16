//! Analytical integer exponent boundaries after conversion to the tensor dtype.

use coeus_autograd::{pow, Var};
use coeus_core::{Float, FloatElement, SequentialBackend};
use coeus_tensor::Tensor;
use eunomia::{Bf16, F16};

fn forward_case<T: Float + FloatElement + std::ops::Neg<Output = T>, const N: usize>(
    exponent: f64,
    input: [f64; N],
    expected: [f64; N],
) where
    SequentialBackend: coeus_ops::BackendOps<T>,
{
    let input = input.map(<T as FloatElement>::from_f64);
    let input = Var::new(
        Tensor::<T, SequentialBackend>::from_slice([N], &input).expect("fixture tensor"),
        false,
    )
    .expect("fixture variable");
    let output = pow(&input, exponent).expect("scalar power");
    assert_eq!(output.tensor.shape(), &[N]);
    assert_eq!(
        output.tensor.as_slice(),
        &expected.map(<T as FloatElement>::from_f64),
        "exponent {exponent}"
    );
}

#[test]
fn large_integer_does_not_wrap_to_zero() {
    // 2^32 is even. Its powers of 2 overflow, and powers of 1/2 underflow.
    forward_case::<f64, 4>(
        4_294_967_296.0,
        [2.0, 0.5, -2.0, -0.5],
        [f64::INFINITY, 0.0, f64::INFINITY, 0.0],
    );
}

#[test]
fn large_positive_integer_does_not_become_negative() {
    // 2^31 + 1 is positive and odd, so negative bases retain their sign.
    forward_case::<f64, 4>(
        2_147_483_649.0,
        [2.0, 0.5, -2.0, -0.5],
        [f64::INFINITY, 0.0, f64::NEG_INFINITY, -0.0],
    );
}

#[test]
fn minimum_signed_integer_has_a_representable_unsigned_magnitude() {
    forward_case::<f64, 3>(
        -2_147_483_648.0,
        [-1.0, 0.5, 2.0],
        [1.0, f64::INFINITY, 0.0],
    );
}

#[test]
fn rounded_single_integer_controls_the_power() {
    // The API rounds the exponent once to T. This value becomes exactly 1.
    forward_case::<f32, 3>(0.999_999_99, [-2.0, 0.5, 2.0], [-2.0, 0.5, 2.0]);
}

#[test]
fn rounded_half_negative_integer_controls_the_power() {
    // This value becomes exactly -1 in binary16, without narrowing the bases.
    forward_case::<F16, 3>(-0.9999, [-2.0, 0.5, 2.0], [-0.5, 2.0, 0.5]);
}

#[test]
fn rounded_single_integer_controls_parity() {
    // 2^24 + 1 rounds to the even integer 2^24 in binary32.
    forward_case::<f32, 2>(16_777_217.0, [-1.0, -2.0], [1.0, f64::INFINITY]);
}

#[test]
fn largest_finite_exponent_keeps_its_integer_magnitude() {
    // The largest finite binary64 value is an even integer, not -1.
    forward_case::<f64, 4>(
        f64::MAX,
        [2.0, 0.5, -1.0, 1.0],
        [f64::INFINITY, 0.0, 1.0, 1.0],
    );
}

// At x = ±1 all integer powers are exact; the derivative is n*x^(n-1).
// Dyadic, nonuniform seeds keep even the largest finite coefficient finite.
fn seeded_unit_case<T: Float + FloatElement + std::ops::Neg<Output = T>>(
    exponent: f64,
    effective_exponent: f64,
    odd: bool,
) where
    SequentialBackend: coeus_ops::BackendOps<T>,
{
    let sign = if odd { -1.0 } else { 1.0 };
    let values = [sign, 1.0, sign, 1.0].map(<T as FloatElement>::from_f64);
    let n = effective_exponent;
    let gradients =
        [-sign * n * 0.5, -n * 0.25, -sign * n * 0.25, -n * 0.5].map(<T as FloatElement>::from_f64);
    for strided in [false, true] {
        for tracked in [false, true] {
            let storage = if strided {
                [-1.0, -1.0, 1.0, 1.0]
            } else {
                [-1.0, 1.0, -1.0, 1.0]
            }
            .map(<T as FloatElement>::from_f64);
            let tensor =
                Tensor::<T, SequentialBackend>::from_slice([2, 2], &storage).expect("unit fixture");
            let tensor = if strided {
                tensor.permute(&[1, 0])
            } else {
                tensor
            };
            let input = Var::new(tensor, tracked).expect("unit variable");
            let output = pow(&input, exponent).expect("integer boundary power");
            assert_eq!(output.tensor.shape(), &[2, 2]);
            assert_eq!(output.tensor.as_slice(), &values, "exponent {exponent}");
            let seed = if strided {
                [0.5, 0.25, -0.25, -0.5]
            } else {
                [0.5, -0.25, 0.25, -0.5]
            }
            .map(<T as FloatElement>::from_f64);
            let seed = Tensor::from_slice([2, 2], &seed).expect("unit seed");
            let seed = if strided { seed.permute(&[1, 0]) } else { seed };
            output
                .backward_with_seed(seed)
                .expect("seeded boundary backward");
            if tracked {
                let gradient = input.grad().expect("unit input gradient");
                assert_eq!(gradient.shape(), &[2, 2]);
                assert_eq!(gradient.as_slice(), &gradients, "exponent {exponent}");
            } else {
                assert!(input.grad().is_none());
            }
        }
    }
}

fn finite_magnitude_case<T: Float + FloatElement + std::ops::Neg<Output = T>>(exponent: f64)
where
    SequentialBackend: coeus_ops::BackendOps<T>,
{
    forward_case::<T, 4>(
        exponent,
        [-2.0, -0.5, 0.5, 2.0],
        [f64::INFINITY, 0.0, 0.0, f64::INFINITY],
    );
    forward_case::<T, 4>(
        -exponent,
        [-2.0, -0.5, 0.5, 2.0],
        [0.0, f64::INFINITY, f64::INFINITY, 0.0],
    );
}

#[test]
fn integer_boundary_gradients_native_double() {
    // Exercise signed narrowing, a full zero low word, and cross-word borrow.
    for exponent in [
        2_147_483_648.0,
        4_294_967_296.0,
        9_223_372_036_854_775_808.0,
        18_446_744_073_709_551_616.0,
        f64::MAX,
    ] {
        seeded_unit_case::<f64>(exponent, exponent, false);
        seeded_unit_case::<f64>(-exponent, -exponent, false);
    }
    seeded_unit_case::<f64>(2_147_483_649.0, 2_147_483_649.0, true);
    seeded_unit_case::<f64>(-2_147_483_649.0, -2_147_483_649.0, true);
    finite_magnitude_case::<f64>(f64::MAX);
}

#[test]
fn integer_boundary_gradients_native_single() {
    for (raw, effective, odd) in [
        (0.999_999_99, 1.0, true),
        (16_777_217.0, 16_777_216.0, false),
        (4_294_967_296.0, 4_294_967_296.0, false),
        (
            3.402_823_466_385_288_6e38,
            3.402_823_466_385_288_6e38,
            false,
        ),
    ] {
        seeded_unit_case::<f32>(raw, effective, odd);
        seeded_unit_case::<f32>(-raw, -effective, odd);
    }
    finite_magnitude_case::<f32>(3.402_823_466_385_288_6e38);
}

#[test]
fn integer_boundary_gradients_half() {
    for (raw, effective, odd) in [
        (0.9999, 1.0, true),
        (2049.0, 2048.0, false),
        (65_504.0, 65_504.0, false),
    ] {
        seeded_unit_case::<F16>(raw, effective, odd);
        seeded_unit_case::<F16>(-raw, -effective, odd);
    }
    finite_magnitude_case::<F16>(65_504.0);
}

#[test]
fn integer_boundary_gradients_brain_half() {
    for (raw, effective, odd) in [
        (0.9999, 1.0, true),
        (257.0, 256.0, false),
        (4_294_967_296.0, 4_294_967_296.0, false),
        (
            3.389_531_389_251_535_5e38,
            3.389_531_389_251_535_5e38,
            false,
        ),
    ] {
        seeded_unit_case::<Bf16>(raw, effective, odd);
        seeded_unit_case::<Bf16>(-raw, -effective, odd);
    }
    finite_magnitude_case::<Bf16>(3.389_531_389_251_535_5e38);
}
