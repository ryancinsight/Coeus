//! `Scalar::total_add`/`total_mul` are defined for every `Scalar`
//! implementor and behave correctly at each type's numeric extremes:
//! integers wrap modulo 2^bits (two's-complement), floats follow IEEE 754
//! (overflow saturates to +/-infinity). One test function, one macro
//! invocation over the integer type list and one generic function called
//! per float type -- no per-type test names, so a newly admitted `Scalar`
//! type is covered by adding it to the list here rather than by a hand-added
//! copy of the whole test.

use coeus_core::{Complex, Float, Scalar};
use eunomia::{Bf16, F16};
use std::fmt::Debug;

/// Every integer `Scalar`: `MAX + 1` wraps to `MIN` (true for signed and
/// unsigned widths alike, since two's-complement addition is exactly
/// arithmetic modulo `2^bits`), the product at `MAX` wraps (cross-verified
/// against the independent `wrapping_mul` oracle in `core`), and a normal,
/// non-overflowing pair still computes the exact result.
macro_rules! assert_int_total_ops {
    ($t:ty) => {{
        assert_eq!(
            <$t>::MAX.total_add(1 as $t),
            <$t>::MIN,
            "{}::MAX.total_add(1) must wrap to MIN",
            stringify!($t)
        );
        assert_eq!(
            <$t>::MAX.total_mul(2 as $t),
            <$t>::MAX.wrapping_mul(2 as $t),
            "{}::MAX.total_mul(2) must match the wrapping_mul oracle",
            stringify!($t)
        );
        assert_eq!((3 as $t).total_add(4 as $t), 7 as $t);
        assert_eq!((3 as $t).total_mul(4 as $t), 12 as $t);
    }};
}

/// Every native/half-precision float `Scalar` (anything implementing
/// [`Float`]): an overflowing operation saturates to infinity per IEEE 754
/// (never panics, never wraps), and a normal pair gives the exact IEEE
/// result.
fn assert_float_total_ops<T: Float + Debug>() {
    assert!(
        T::MAX.total_add(T::MAX).is_infinite(),
        "{:?}::MAX.total_add(MAX) must overflow to infinity",
        T::MAX
    );
    assert!(
        T::MAX.total_mul(T::MAX).is_infinite(),
        "{:?}::MAX.total_mul(MAX) must overflow to infinity",
        T::MAX
    );
    let two = T::one().total_add(T::one());
    let three = two.total_add(T::one());
    let five = three.total_add(two);
    assert_eq!(two.total_add(three), five);
    let six = two.total_mul(three);
    assert_eq!(six, three.total_add(three));
}

#[test]
fn total_add_and_mul_are_defined_for_every_scalar_type() {
    assert_int_total_ops!(i8);
    assert_int_total_ops!(i16);
    assert_int_total_ops!(i32);
    assert_int_total_ops!(i64);
    assert_int_total_ops!(u8);
    assert_int_total_ops!(u16);
    assert_int_total_ops!(u32);
    assert_int_total_ops!(u64);

    assert_float_total_ops::<f32>();
    assert_float_total_ops::<f64>();
    assert_float_total_ops::<F16>();
    assert_float_total_ops::<Bf16>();

    // `Complex<T>` implements `Scalar` but not `Float`; its arithmetic is
    // component-wise, so overflow and normal cases follow the same IEEE 754
    // rule at the component level as the real float types above.
    let huge = Complex::<f32>::new(f32::MAX, 0.0);
    let overflowed = huge.total_add(huge);
    assert!(
        overflowed.re.is_infinite(),
        "Complex<f32> total_add must overflow its real part to infinity"
    );
    let one_two = Complex::<f32>::new(1.0, 2.0);
    let three_four = Complex::<f32>::new(3.0, 4.0);
    assert_eq!(one_two.total_add(three_four), Complex::new(4.0, 6.0));
    // (1 + 2i)(3 + 4i) = (1*3 - 2*4) + (1*4 + 2*3)i = -5 + 10i
    assert_eq!(one_two.total_mul(three_four), Complex::new(-5.0, 10.0));
}
