use coeus_core::{Complex, TryFromCount};
use eunomia::{Bf16, FloatElement, F16};

#[test]
fn primitive_scalars_convert_counts_without_f64_detour() {
    assert_eq!(i32::try_from_count(7), Ok(7));
    assert_eq!(u64::try_from_count(11), Ok(11));
    assert_eq!(f32::from_count(13), 13.0);
    assert_eq!(f64::from_count(17), 17.0);
}

#[test]
fn reduced_precision_scalars_convert_counts_to_native_values() {
    assert_eq!(F16::from_count(5), F16::from_f32(5.0));
    assert_eq!(Bf16::from_count(9), Bf16::from_f32(9.0));
}

#[test]
fn complex_scalars_convert_counts_to_real_axis() {
    let value = Complex::<f32>::try_from_count(23).expect("complex embeds every count");

    assert_eq!(value, Complex::new(23.0, 0.0));
}

#[test]
fn integer_scalars_refuse_counts_outside_their_range() {
    assert_eq!(i8::try_from_count(127), Ok(127));
    let err = i8::try_from_count(128).expect_err("128 does not fit i8");
    assert_eq!(err.count(), 128);
    assert_eq!(err.target(), "i8");
}
