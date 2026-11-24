//! `Scalar::has_zero_bit_pattern` is a representation-level predicate: it holds
//! exactly when every stored byte is zero, so it distinguishes signed zero from
//! zero and is not a numeric `== 0` test. One generic function is instantiated
//! over every shipped `Scalar` family; the expected answers are stated per
//! value, not computed from the implementation's own byte view.

use coeus_core::{Complex, Scalar};
use eunomia::{Bf16, F16};

/// `zero` is the all-zero value, `distinct` has a nonzero stored byte, and
/// `numerically_zero_but_set` (when the type has one) compares equal to zero
/// yet stores a nonzero byte.
fn assert_zero_bit_pattern<T: Scalar + core::fmt::Debug>(
    zero: T,
    distinct: T,
    numerically_zero_but_set: Option<T>,
) {
    assert!(
        zero.has_zero_bit_pattern(),
        "{zero:?} must be all-zero bytes"
    );
    assert!(
        !distinct.has_zero_bit_pattern(),
        "{distinct:?} stores a nonzero byte"
    );
    if let Some(negative_zero) = numerically_zero_but_set {
        assert!(
            !negative_zero.has_zero_bit_pattern(),
            "{negative_zero:?} stores the sign bit"
        );
    }
}

#[test]
fn zero_bit_pattern_holds_exactly_for_all_zero_bytes_in_every_scalar_family() {
    assert_zero_bit_pattern(0.0_f32, 1.0_f32, Some(-0.0_f32));
    assert_zero_bit_pattern(0.0_f64, 1.0_f64, Some(-0.0_f64));
    assert_zero_bit_pattern(
        F16::from_bits(0),
        F16::from_f32(1.0),
        Some(F16::from_bits(0x8000)),
    );
    assert_zero_bit_pattern(
        Bf16::from_bits(0),
        Bf16::from_f32(1.0),
        Some(Bf16::from_bits(0x8000)),
    );
    assert_zero_bit_pattern(0_i8, i8::MIN, None);
    assert_zero_bit_pattern(0_i16, i16::MIN, None);
    assert_zero_bit_pattern(0_i32, -1_i32, None);
    assert_zero_bit_pattern(0_i64, i64::MIN, None);
    assert_zero_bit_pattern(0_u8, 1_u8, None);
    assert_zero_bit_pattern(0_u16, 1_u16, None);
    assert_zero_bit_pattern(0_u32, u32::MAX, None);
    assert_zero_bit_pattern(0_u64, 1_u64, None);
}

#[test]
fn zero_bit_pattern_of_complex_requires_both_components_all_zero() {
    let zero = Complex {
        re: 0.0_f32,
        im: 0.0,
    };
    assert_zero_bit_pattern(
        zero,
        Complex {
            re: 0.0_f32,
            im: 1.0,
        },
        Some(Complex {
            re: 0.0_f32,
            im: -0.0,
        }),
    );
    assert!(!Complex {
        re: 1.0_f64,
        im: 0.0
    }
    .has_zero_bit_pattern());
}
