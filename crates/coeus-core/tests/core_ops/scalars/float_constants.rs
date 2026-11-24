//! IEEE 754 minimum-positive-normal constants match the canonical provider.

use coeus_core::Float;
use eunomia::{Bf16, F16};

#[test]
fn half_minimum_positive_is_the_first_normal() {
    let minimum = <F16 as Float>::MIN_POSITIVE;
    assert_eq!(minimum, F16::MIN_POSITIVE);
    assert_eq!(minimum.to_bits(), 0x0400);
    assert_eq!(minimum.to_f32(), 2.0_f32.powi(-14));
    // A normal binary16 has a nonzero exponent field. Its predecessor has
    // exponent zero and is the largest subnormal, not another normal value.
    assert_eq!(minimum.to_bits() & 0x7c00, 0x0400);
    assert_eq!(F16::from_bits(minimum.to_bits() - 1).to_bits() & 0x7c00, 0);
    assert_eq!(minimum.to_f32().classify(), std::num::FpCategory::Normal);
}

#[test]
fn brain_half_minimum_positive_remains_normal() {
    let minimum = <Bf16 as Float>::MIN_POSITIVE;
    assert_eq!(minimum, Bf16::MIN_POSITIVE);
    assert_eq!(minimum.to_bits(), 0x0080);
    assert_eq!(minimum.to_f32(), 2.0_f32.powi(-126));
    assert_eq!(minimum.to_f32().classify(), std::num::FpCategory::Normal);
}
