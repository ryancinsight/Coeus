//! Division safety tests
//!
//! Validates that division operations handle edge cases correctly
//! per SRS-REL-ERR-001 (No panics in public APIs)

use coeus_dtype::int::*;
use coeus_dtype::traits::IntExt;
use num_traits::Zero;

#[test]
fn test_checked_div_by_zero_returns_none() {
    let a = Int32::new(42);
    let zero = Int32::zero();

    // checked_div should return None for division by zero
    assert_eq!(a.checked_div(zero), None);
}

#[test]
fn test_checked_div_normal() {
    let a = Int32::new(42);
    let b = Int32::new(7);

    assert_eq!(a.checked_div(b), Some(Int32::new(6)));
}

#[test]
fn test_checked_div_negative() {
    let a = Int8::new(-10);
    let b = Int8::new(2);

    assert_eq!(a.checked_div(b), Some(Int8::new(-5)));
}

#[test]
fn test_checked_div_overflow() {
    // i8::MIN / -1 overflows (result would be 128, but i8::MAX is 127)
    let min = Int8::min_value();
    let neg_one = Int8::new(-1);

    assert_eq!(min.checked_div(neg_one), None);
}

#[test]
#[should_panic(expected = "attempt to divide by zero")]
fn test_unchecked_div_panics_on_zero() {
    // Standard Div trait WILL panic - this documents current behavior
    // TODO: Refactor to return Result or require explicit checked_div
    let a = Int32::new(42);
    let zero = Int32::zero();

    let _ = a / zero; // This WILL panic
}

#[test]
fn test_checked_rem_by_zero_returns_none() {
    let a = Int32::new(42);
    let zero = Int32::zero();

    assert_eq!(a.checked_rem(zero), None);
}

#[test]
fn test_unsigned_div_by_zero() {
    let a = UInt32::new(42);
    let zero = UInt32::zero();

    assert_eq!(a.checked_div(zero), None);
}
