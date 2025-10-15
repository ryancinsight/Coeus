//! Type promotion edge case tests
//!
//! Validates correct type promotion behavior per SRS § 3.1

use coeus_dtype::{promotion::promote, Dtype};

#[test]
fn test_i64_u64_promotion_to_float() {
    // CRITICAL: i64/u64 cannot promote to i64 (u64::MAX > i64::MAX)
    // Correct behavior: promote to Float64
    assert_eq!(promote(Dtype::Int64, Dtype::UInt64), Dtype::Float64);
    assert_eq!(promote(Dtype::UInt64, Dtype::Int64), Dtype::Float64);
}

#[test]
fn test_smaller_mixed_sign_promotions() {
    assert_eq!(promote(Dtype::Int8, Dtype::UInt8), Dtype::Int16);
    assert_eq!(promote(Dtype::Int16, Dtype::UInt16), Dtype::Int32);
    assert_eq!(promote(Dtype::Int32, Dtype::UInt32), Dtype::Int64);
}

#[test]
fn test_same_sign_promotion() {
    assert_eq!(promote(Dtype::Int8, Dtype::Int32), Dtype::Int32);
    assert_eq!(promote(Dtype::UInt8, Dtype::UInt32), Dtype::UInt32);
}

#[test]
fn test_float_dominates_int() {
    assert_eq!(promote(Dtype::Float32, Dtype::Int64), Dtype::Float32);
    assert_eq!(promote(Dtype::Int32, Dtype::Float64), Dtype::Float64);
}
