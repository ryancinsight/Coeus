//! Type promotion rules for dtype operations
//!
//! Implements NumPy/PyTorch-compatible type promotion for mixed-dtype operations.
//! Follows the principle: operations between different dtypes promote to the
//! least common type that can represent all values.

use crate::Dtype;

/// Promote two dtypes to their common type
///
/// Follows type promotion hierarchy:
/// ```text
/// bool < int8 < int16 < int32 < int64
///      < uint8 < uint16 < uint32 < uint64
///      < float16 < float32 < float64
/// complex types promote to complex with promoted real type
/// ```
///
/// # Examples
///
/// ```
/// use coeus_dtype::{Dtype, promotion::promote};
///
/// assert_eq!(promote(Dtype::Int8, Dtype::Int32), Dtype::Int32);
/// assert_eq!(promote(Dtype::Float32, Dtype::Int32), Dtype::Float32);
/// assert_eq!(promote(Dtype::Float32, Dtype::Float64), Dtype::Float64);
/// ```
#[must_use]
pub fn promote(left: Dtype, right: Dtype) -> Dtype {
    if left == right {
        return left;
    }

    // Floating point always wins over integer
    if left.is_floating_point() && !right.is_floating_point() {
        return left;
    }
    if right.is_floating_point() && !left.is_floating_point() {
        return right;
    }

    // Complex always wins
    if left.is_complex() {
        return left;
    }
    if right.is_complex() {
        return right;
    }

    // Promote to larger size within same category
    if left.is_floating_point() && right.is_floating_point() {
        promote_float(left, right)
    } else if left.is_integer() && right.is_integer() {
        promote_int(left, right)
    } else {
        // Fallback: promote to float64 for mixed categories
        Dtype::Float64
    }
}

/// Promote two floating point types
fn promote_float(left: Dtype, right: Dtype) -> Dtype {
    use Dtype::{BFloat16, Float32, Float64, Half};
    match (left, right) {
        (Float64, _) | (_, Float64) => Float64,
        (Float32, _) | (_, Float32) | (BFloat16, Half) | (Half, BFloat16) => Float32,
        (BFloat16, _) | (_, BFloat16) => BFloat16,
        (Half, _) | (_, Half) => Half,
        _ => Float64, // Fallback
    }
}

/// Promote two integer types
///
/// # Implementation Note
///
/// For mixed signed/unsigned at maximum width (i64/u64), promotion to i64 is
/// **incorrect** as `u64::MAX` cannot fit in `i64`. PyTorch/NumPy handle this by
/// promoting to float64 or erroring. We follow `PyTorch`: promote to `Float64`.
fn promote_int(left: Dtype, right: Dtype) -> Dtype {
    use Dtype::{Float64, Int16, Int32, Int64, Int8, UInt16, UInt32, UInt64, UInt8};

    let left_signed = matches!(left, Int8 | Int16 | Int32 | Int64);
    let right_signed = matches!(right, Int8 | Int16 | Int32 | Int64);

    // Mixed signed/unsigned: promote to next larger signed type
    if left_signed != right_signed {
        let max_bytes = left.size_bytes().max(right.size_bytes());
        return match max_bytes {
            1 => Int16, // i8/u8 mix -> i16 (can hold both ranges)
            2 => Int32, // i16/u16 mix -> i32
            4 => Int64, // i32/u32 mix -> i64
            // CRITICAL: i64/u64 cannot promote to i64 (u64::MAX > i64::MAX)
            // Follow PyTorch: promote to Float64 (53-bit mantissa loses precision
            // but avoids silent overflow)
            _ => Float64,
        };
    }

    // Same signedness: promote to larger
    let max_bytes = left.size_bytes().max(right.size_bytes());
    match (left_signed, max_bytes) {
        (true, 1) => Int8,
        (true, 2) => Int16,
        (true, 4) => Int32,
        (true, _) => Int64,
        (false, 1) => UInt8,
        (false, 2) => UInt16,
        (false, 4) => UInt32,
        (false, _) => UInt64,
    }
}

/// Check if a dtype can be safely cast to another without loss
///
/// # Examples
///
/// ```
/// use coeus_dtype::{Dtype, promotion::can_cast};
///
/// assert!(can_cast(Dtype::Int8, Dtype::Int32));  // Widen: safe
/// assert!(!can_cast(Dtype::Int32, Dtype::Int8)); // Narrow: unsafe
/// assert!(can_cast(Dtype::Int32, Dtype::Float64)); // Int to float: safe if mantissa big enough
/// assert!(!can_cast(Dtype::Float32, Dtype::Int32)); // Float to int: may lose fractional part
/// ```
#[must_use]
pub fn can_cast(from: Dtype, to: Dtype) -> bool {
    if from == to {
        return true;
    }

    // Same category, larger size = safe
    if from.is_floating_point() && to.is_floating_point() {
        return to.size_bytes() >= from.size_bytes();
    }

    if from.is_integer() && to.is_integer() {
        // Check for sign compatibility
        let from_signed = matches!(
            from,
            Dtype::Int8 | Dtype::Int16 | Dtype::Int32 | Dtype::Int64
        );
        let to_signed = matches!(to, Dtype::Int8 | Dtype::Int16 | Dtype::Int32 | Dtype::Int64);

        // Unsigned to signed requires extra bit
        if !from_signed && to_signed {
            return to.size_bytes() > from.size_bytes();
        }

        // Otherwise, same or larger size is safe
        return to.size_bytes() >= from.size_bytes();
    }

    // Integer to float: safe if float mantissa can hold int
    if from.is_integer() && to.is_floating_point() {
        return match (from, to) {
            (Dtype::Int8 | Dtype::UInt8 | Dtype::Int16, Dtype::Float32 | Dtype::Float64)
            | (Dtype::Int32 | _, Dtype::Float64) => true, // f64 has 53-bit mantissa
            _ => false,
        };
    }

    // Float to int: never safe (fractional part loss)
    if from.is_floating_point() && to.is_integer() {
        return false;
    }

    // Complex promotions
    if to.is_complex() && (from.is_floating_point() || from.is_integer()) {
        return true; // Can always promote to complex
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promote_same_type() {
        assert_eq!(promote(Dtype::Float32, Dtype::Float32), Dtype::Float32);
        assert_eq!(promote(Dtype::Int32, Dtype::Int32), Dtype::Int32);
    }

    #[test]
    fn test_promote_float_int() {
        assert_eq!(promote(Dtype::Float32, Dtype::Int32), Dtype::Float32);
        assert_eq!(promote(Dtype::Int32, Dtype::Float32), Dtype::Float32);
        assert_eq!(promote(Dtype::Float64, Dtype::Int64), Dtype::Float64);
    }

    #[test]
    fn test_promote_float_sizes() {
        assert_eq!(promote(Dtype::Float32, Dtype::Float64), Dtype::Float64);
        assert_eq!(promote(Dtype::Float64, Dtype::Float32), Dtype::Float64);
    }

    #[test]
    fn test_promote_int_sizes() {
        assert_eq!(promote(Dtype::Int8, Dtype::Int32), Dtype::Int32);
        assert_eq!(promote(Dtype::Int32, Dtype::Int64), Dtype::Int64);
        assert_eq!(promote(Dtype::UInt8, Dtype::UInt32), Dtype::UInt32);
    }

    #[test]
    fn test_promote_signed_unsigned() {
        // Mixed signed/unsigned promotes to larger signed type
        assert_eq!(promote(Dtype::Int8, Dtype::UInt8), Dtype::Int16);
        assert_eq!(promote(Dtype::Int16, Dtype::UInt16), Dtype::Int32);
        assert_eq!(promote(Dtype::Int32, Dtype::UInt32), Dtype::Int64);
        // CORRECTED: i64/u64 cannot safely promote to i64 (u64::MAX > i64::MAX)
        // Promotes to Float64 per PyTorch semantics
        assert_eq!(promote(Dtype::Int64, Dtype::UInt64), Dtype::Float64);
    }

    #[test]
    fn test_can_cast_safe() {
        assert!(can_cast(Dtype::Int8, Dtype::Int32));
        assert!(can_cast(Dtype::Float32, Dtype::Float64));
        assert!(can_cast(Dtype::Int16, Dtype::Float32));
    }

    #[test]
    fn test_can_cast_unsafe() {
        assert!(!can_cast(Dtype::Int32, Dtype::Int8));
        assert!(!can_cast(Dtype::Float64, Dtype::Float32));
        assert!(!can_cast(Dtype::Float32, Dtype::Int32));
    }

    #[test]
    fn test_can_cast_complex() {
        assert!(can_cast(Dtype::Float32, Dtype::Complex32));
        assert!(can_cast(Dtype::Int32, Dtype::Complex64));
        assert!(!can_cast(Dtype::Complex64, Dtype::Float64));
    }
}
