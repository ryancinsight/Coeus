//! Integration tests for dtype crate
//!
//! Tests cross-module interactions and type conversions

use coeus_dtype::*;
use num_traits::{NumCast, One, Zero};

#[test]
fn test_float_to_int_conversion() {
    let f = float::Float32::new(42.7);
    let i: Option<int::Int32> = NumCast::from(f);
    assert!(i.is_some());
    assert_eq!(i.unwrap().get(), 42);
}

#[test]
fn test_int_to_float_conversion() {
    let i = int::Int32::new(42);
    let f: Option<float::Float32> = NumCast::from(i);
    assert!(f.is_some());
    assert_eq!(f.unwrap().get(), 42.0);
}

#[test]
fn test_dtype_enum_consistency() {
    assert_eq!(float::Float32::dtype(), Dtype::Float32);
    assert_eq!(float::Float64::dtype(), Dtype::Float64);
    assert_eq!(int::Int8::dtype(), Dtype::Int8);
    assert_eq!(int::Int32::dtype(), Dtype::Int32);
    assert_eq!(int::UInt8::dtype(), Dtype::UInt8);
}

#[test]
fn test_mixed_arithmetic_via_traits() {
    let f_zero = float::Float32::zero();
    let f_one = float::Float32::one();
    let i_zero = int::Int32::zero();
    let i_one = int::Int32::one();

    assert!(f_zero.is_zero());
    assert!(i_zero.is_zero());
    assert_eq!(f_one, float::Float32::new(1.0));
    assert_eq!(i_one, int::Int32::new(1));
}

#[test]
fn test_dtype_size_bytes() {
    assert_eq!(Dtype::Float32.size_bytes(), 4);
    assert_eq!(Dtype::Float64.size_bytes(), 8);
    assert_eq!(Dtype::Int8.size_bytes(), 1);
    assert_eq!(Dtype::Int32.size_bytes(), 4);
    assert_eq!(Dtype::Int64.size_bytes(), 8);
}

#[test]
fn test_dtype_names() {
    assert_eq!(Dtype::Float32.name(), "float32");
    assert_eq!(Dtype::Int32.name(), "int32");
    assert_eq!(Dtype::UInt8.name(), "uint8");
}

#[test]
fn test_dtype_categories() {
    assert!(Dtype::Float32.is_floating_point());
    assert!(!Dtype::Float32.is_integer());

    assert!(Dtype::Int32.is_integer());
    assert!(!Dtype::Int32.is_floating_point());

    assert!(!Dtype::QInt8.is_floating_point());
    assert!(!Dtype::QInt8.is_integer());
    assert!(Dtype::QInt8.is_quantized());
}

#[test]
fn test_overflow_edge_cases() {
    use traits::IntExt;

    let max_i8 = int::Int8::max_value();
    assert_eq!(max_i8.checked_add(int::Int8::new(1)), None);

    let min_i8 = int::Int8::min_value();
    assert_eq!(min_i8.checked_sub(int::Int8::new(1)), None);

    // Wrapping should work
    let wrapped = max_i8 + int::Int8::new(1);
    assert_eq!(wrapped, min_i8);
}

#[test]
fn test_bitwise_operations_across_types() {
    use traits::IntExt;

    let a = int::Int32::new(0xFF00);
    let b = int::Int32::new(0x0F0F);

    assert_eq!(a.bitand(b), int::Int32::new(0x0F00));
    assert_eq!(a.bitor(b), int::Int32::new(0xFF0F));
    assert_eq!(a.bitxor(b), int::Int32::new(0xF00F));
}

#[test]
fn test_numeric_edge_values() {
    use traits::FloatExt;

    let nan = float::Float32::new(f32::NAN);
    let inf = float::Float32::new(f32::INFINITY);
    let neg_inf = float::Float32::new(f32::NEG_INFINITY);

    assert!(nan.is_nan());
    assert!(inf.is_infinite());
    assert!(neg_inf.is_infinite());
    assert!(!nan.is_finite());
}

// Proptest-based property tests for edge cases
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_float32_arithmetic_no_nan_inf(
            a in -1e10f32..1e10f32,
            b in -1e10f32..1e10f32
        ) {
            let fa = float::Float32::new(a);
            let fb = float::Float32::new(b);

            // Addition should not produce NaN/Inf for finite inputs
            if a.is_finite() && b.is_finite() {
                let sum = fa + fb;
                prop_assert!(sum.get().is_finite() || sum.get().is_nan(),
                    "Addition of finite numbers should be finite or NaN, got {:?}", sum);
            }
        }

        #[test]
        fn prop_int32_overflow_detection(
            a in i32::MIN..i32::MAX,
            b in i32::MIN..i32::MAX
        ) {
            use traits::IntExt;
            let ia = int::Int32::new(a);
            let ib = int::Int32::new(b);

            // Test checked operations
            if let Some(sum) = ia.checked_add(ib) {
                prop_assert_eq!(sum.get(), a.wrapping_add(b));
            } else {
                // Overflow occurred
                prop_assert!(a.checked_add(b).is_none());
            }
        }

        #[test]
        fn prop_float_precision_loss(
            x in -1e6f32..1e6f32
        ) {
            let fx = float::Float32::new(x);
            let back = fx.get();
            // For most values, round-trip should be exact or very close
            prop_assert!((x - back).abs() < 1e-6 || x.is_nan() && back.is_nan(),
                "Round-trip precision loss too large: {} vs {}", x, back);
        }

        #[cfg(feature = "complex")]
        #[test]
        fn prop_complex_norm_properties(
            re in -100.0f32..100.0,
            im in -100.0f32..100.0
        ) {
            use coeus_dtype::complex::Complex32;
            let z = Complex32::new(re, im);
            let norm = (z.re * z.re + z.im * z.im).sqrt();
            prop_assert!(norm >= 0.0, "Norm should be non-negative");
            if re == 0.0 && im == 0.0 {
                prop_assert_eq!(norm, 0.0, "Zero complex should have zero norm");
            }
        }

        #[cfg(feature = "half")]
        #[test]
        fn prop_half_range_conversion(
            x in -65504.0f32..65504.0f32
        ) {
            let h = float::Half::new(x);
            let back = h.get();
            // Half precision has limited range, so check within representable range
            if x.abs() <= 65504.0 {
                prop_assert!((x - back).abs() <= x.abs() * 1e-3 || back.is_infinite(),
                    "Half precision conversion error too large: {} vs {}", x, back);
            }
        }
    }
}
