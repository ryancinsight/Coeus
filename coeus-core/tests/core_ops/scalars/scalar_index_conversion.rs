use coeus_core::{Complex, Scalar};
use eunomia::{Bf16, F16};

#[test]
fn primitive_scalars_convert_indices_without_f64_detour() {
    assert_eq!(i32::from_usize(7), 7);
    assert_eq!(u64::from_usize(11), 11);
    assert_eq!(f32::from_usize(13), 13.0);
    assert_eq!(f64::from_usize(17), 17.0);
}

#[test]
fn reduced_precision_scalars_convert_indices_to_native_values() {
    assert_eq!(F16::from_usize(5), F16::from_f32(5.0));
    assert_eq!(Bf16::from_usize(9), Bf16::from_f32(9.0));
}

#[test]
fn complex_scalars_convert_indices_to_real_axis() {
    let value = Complex::<f32>::from_usize(23);

    assert_eq!(value, Complex::new(23.0, 0.0));
}
