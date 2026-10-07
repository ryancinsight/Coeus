//! Shared harness for the staged `powi` differential: ULP comparison and
//! per-type check helpers. (The sweep machinery retired with the deleted
//! transcendental tests; see the root module docs.)

pub(crate) use coeus_core::Float as CoeusFloat;
pub(crate) use eunomia::FloatElement as EunomiaFloat;

/// Order-preserving integer map, so ULP distance is a subtraction.
fn ordered_f32(x: f32) -> u32 {
    let bits = x.to_bits();
    if bits & 0x8000_0000 == 0 {
        bits ^ 0x8000_0000
    } else {
        !bits
    }
}

fn ordered_f64(x: f64) -> u64 {
    let bits = x.to_bits();
    if bits & 0x8000_0000_0000_0000 == 0 {
        bits ^ 0x8000_0000_0000_0000
    } else {
        !bits
    }
}

/// Assert class agreement + ULP tolerance; returns the observed ULP distance.
pub(crate) fn check_f32(name: &str, x: f32, coeus: f32, eunomia: f32, max_ulp: u32) -> u32 {
    if coeus.is_nan() || eunomia.is_nan() {
        assert!(
            coeus.is_nan() && eunomia.is_nan(),
            "{name}({x}): NaN-class mismatch: coeus={coeus}, eunomia={eunomia}"
        );
        return 0;
    }
    if coeus.is_infinite() || eunomia.is_infinite() {
        assert_eq!(
            coeus, eunomia,
            "{name}({x}): inf-class mismatch: coeus={coeus}, eunomia={eunomia}"
        );
        return 0;
    }
    let ulp = ordered_f32(coeus).abs_diff(ordered_f32(eunomia));
    assert!(
        ulp <= max_ulp,
        "{name}({x}): {ulp} ulp over tolerance {max_ulp}: coeus={coeus} eunomia={eunomia}"
    );
    ulp
}

pub(crate) fn check_f64(name: &str, x: f64, coeus: f64, eunomia: f64, max_ulp: u64) -> u64 {
    if coeus.is_nan() || eunomia.is_nan() {
        assert!(
            coeus.is_nan() && eunomia.is_nan(),
            "{name}({x}): NaN-class mismatch: coeus={coeus}, eunomia={eunomia}"
        );
        return 0;
    }
    if coeus.is_infinite() || eunomia.is_infinite() {
        assert_eq!(
            coeus, eunomia,
            "{name}({x}): inf-class mismatch: coeus={coeus}, eunomia={eunomia}"
        );
        return 0;
    }
    let ulp = ordered_f64(coeus).abs_diff(ordered_f64(eunomia));
    assert!(
        ulp <= max_ulp,
        "{name}({x}): {ulp} ulp over tolerance {max_ulp}: coeus={coeus} eunomia={eunomia}"
    );
    ulp
}
