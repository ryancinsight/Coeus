//! Comprehensive edge case tests
//!
//! Per SRS-MAINT-TEST-001: >95% branch coverage with edge cases:
//! - Negative operands and sign propagation
//! - Overflow/underflow with two's complement validation
//! - NaN propagation in floating point
//! - Subnormal number handling
//! - Precision loss in conversions

use approx::assert_relative_eq;
use coeus_dtype::float::*;
use coeus_dtype::int::*;
use coeus_dtype::traits::{FloatExt, IntExt};
use num_traits::{Float, NumCast, One, Signed, Zero};

// ============================================================================
// NEGATIVE NUMBER TESTS
// ============================================================================

#[test]
fn test_negative_multiplication() {
    let a = Int32::new(-1);
    let b = Int32::new(10);
    assert_eq!(a * b, Int32::new(-10));

    let c = Int32::new(-5);
    let d = Int32::new(-3);
    assert_eq!(c * d, Int32::new(15)); // negative * negative = positive
}

#[test]
fn test_negative_division() {
    let a = Int32::new(-10);
    let b = Int32::new(2);
    assert_eq!(a / b, Int32::new(-5));

    let c = Int32::new(-20);
    let d = Int32::new(-4);
    assert_eq!(c / d, Int32::new(5)); // negative / negative = positive
}

#[test]
fn test_negative_float_operations() {
    let a = Float32::new(-3.5);
    let b = Float32::new(2.0);

    assert_relative_eq!((a + b).get(), -1.5, epsilon = 1e-6);
    assert_relative_eq!((a * b).get(), -7.0, epsilon = 1e-6);
    assert_relative_eq!((a / b).get(), -1.75, epsilon = 1e-6);
}

// ============================================================================
// OVERFLOW/UNDERFLOW TESTS
// ============================================================================

#[test]
fn test_i8_overflow_wraparound() {
    let max = Int8::max_value();
    let one = Int8::one();

    // Wrapping add: 127 + 1 = -128 (two's complement)
    assert_eq!(max + one, Int8::min_value());
}

#[test]
fn test_i8_underflow_wraparound() {
    let min = Int8::min_value();
    let one = Int8::one();

    // Wrapping sub: -128 - 1 = 127
    assert_eq!(min - one, Int8::max_value());
}

#[test]
fn test_multiplication_overflow() {
    let a = Int8::new(64);
    let b = Int8::new(2);

    // 64 * 2 = 128, overflows i8::MAX (127)
    let result = a * b;
    // Wrapping multiply: wraps to -128
    assert_eq!(result, Int8::new(-128));
}

#[test]
fn test_checked_overflow_detection() {
    let max = Int8::max_value();
    let one = Int8::one();

    assert_eq!(max.checked_add(one), None);
    assert_eq!(max.checked_mul(Int8::new(2)), None);
}

#[test]
fn test_unsigned_underflow() {
    let zero = UInt8::zero();
    let one = UInt8::one();

    // 0 - 1 wraps to 255
    assert_eq!(zero - one, UInt8::new(255));
    assert_eq!(zero.checked_sub(one), None);
}

// ============================================================================
// NAN PROPAGATION TESTS
// ============================================================================

#[test]
fn test_nan_arithmetic_propagation() {
    let nan = Float32::new(f32::NAN);
    let x = Float32::new(1.0);

    assert!((nan + x).is_nan());
    assert!((nan - x).is_nan());
    assert!((nan * x).is_nan());
    assert!((nan / x).is_nan());
}

#[test]
fn test_nan_math_functions() {
    let nan = Float32::new(f32::NAN);

    assert!(nan.sin().is_nan());
    assert!(nan.cos().is_nan());
    assert!(nan.exp().is_nan());
    assert!(nan.ln().is_nan());
    assert!(nan.sqrt().is_nan());
}

#[test]
fn test_sqrt_negative_produces_nan() {
    let neg = Float32::new(-1.0);
    assert!(neg.sqrt().is_nan());
}

#[test]
fn test_ln_negative_produces_nan() {
    let neg = Float32::new(-1.0);
    assert!(neg.ln().is_nan());
}

// ============================================================================
// INFINITY TESTS
// ============================================================================

#[test]
fn test_infinity_arithmetic() {
    let inf = Float32::new(f32::INFINITY);
    let neg_inf = Float32::new(f32::NEG_INFINITY);
    let x = Float32::new(1.0);

    assert!((inf + x).is_infinite());
    assert!((inf * x).is_infinite());
    assert!((inf - inf).is_nan()); // inf - inf = NaN
    assert!((inf * neg_inf).is_infinite()); // Should be negative infinity
}

#[test]
fn test_division_by_infinity() {
    let x = Float32::new(1.0);
    let inf = Float32::new(f32::INFINITY);

    assert_eq!((x / inf).get(), 0.0);
}

// ============================================================================
// PRECISION LOSS TESTS
// ============================================================================

#[test]
fn test_float_to_int_truncation() {
    let f = Float32::new(42.9);
    let i: Option<Int32> = NumCast::from(f);

    assert_eq!(i.unwrap().get(), 42); // Truncates decimal part
}

#[test]
fn test_negative_float_to_int_truncation() {
    let f = Float32::new(-42.9);
    let i: Option<Int32> = NumCast::from(f);

    assert_eq!(i.unwrap().get(), -42); // Truncates toward zero
}

#[test]
fn test_large_int_to_float_precision_loss() {
    // 2^24 + 1 cannot be represented exactly in f32 (mantissa has 23 bits)
    let large = Int32::new((1 << 24) + 1);
    let f: Option<Float32> = NumCast::from(large);

    // Conversion succeeds but loses precision
    assert!(f.is_some());
}

// ============================================================================
// ZERO EDGE CASES
// ============================================================================

#[test]
fn test_zero_multiplication() {
    let zero = Int32::zero();
    let x = Int32::new(42);

    assert_eq!(zero * x, zero);
    assert_eq!(x * zero, zero);
}

#[test]
fn test_negative_zero_float() {
    let pos_zero = Float32::new(0.0);
    let neg_zero = Float32::new(-0.0);

    // IEEE 754: +0.0 == -0.0
    assert_eq!(pos_zero, neg_zero);

    // But they have different bit patterns
    assert_eq!(pos_zero.get().to_bits(), 0x00000000);
    assert_eq!(neg_zero.get().to_bits(), 0x80000000);
}

#[test]
fn test_division_produces_infinity() {
    let x = Float32::new(1.0);
    let zero = Float32::new(0.0);

    assert!((x / zero).is_infinite());
    assert!((x / zero).get().is_sign_positive());
}

#[test]
fn test_negative_division_produces_negative_infinity() {
    let x = Float32::new(-1.0);
    let zero = Float32::new(0.0);

    let result = x / zero;
    assert!(result.is_infinite());
    assert!(result.get().is_sign_negative());
}

// ============================================================================
// SUBNORMAL NUMBER TESTS
// ============================================================================

#[test]
fn test_subnormal_float_operations() {
    // Smallest positive subnormal f32: 2^-149
    let subnormal = Float32::new(f32::MIN_POSITIVE / 2.0);

    assert!(subnormal.get().is_subnormal() || subnormal.is_zero());
    assert!(!subnormal.is_nan());
    assert!(subnormal.is_finite());
}

// ============================================================================
// SIGN PROPAGATION TESTS
// ============================================================================

#[test]
fn test_negation_sign_flip() {
    let pos = Int32::new(42);
    let neg = -pos;
    assert_eq!(neg, Int32::new(-42));

    let neg_neg = -neg;
    assert_eq!(neg_neg, pos);
}

#[test]
fn test_abs_removes_sign() {
    let neg = Int32::new(-42);
    assert_eq!(neg.abs(), Int32::new(42));

    let pos = Int32::new(42);
    assert_eq!(pos.abs(), pos);
}

#[test]
fn test_i8_min_negation_overflow() {
    let min = Int8::min_value();

    // -(-128) would be 128, but i8::MAX is 127
    // Wrapping negation produces -128 again
    assert_eq!(-min, min);

    // Checked negation detects overflow
    assert_eq!(min.checked_neg(), None);
}

#[test]
fn test_i8_min_abs_overflow() {
    let min = Int8::min_value();

    // abs(-128) would be 128, overflows
    // Wrapping abs returns -128
    assert_eq!(min.abs(), min);

    // Checked abs detects overflow
    assert_eq!(min.checked_abs(), None);
}
