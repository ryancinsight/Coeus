//! CPU math operations for strided storage
//!
//! Provides optimized unary math operations for strided storage.

use crate::DataType;
use dtype::num_traits;

// Negation
crate::unary_strided_primitive!(neg_strided_primitive, |x_val: T| -x_val, core::ops::Neg<Output = T>);

crate::unary_csr_primitive!(neg_csr_primitive, |x_val: T| -x_val, core::ops::Neg<Output = T>);

// Transcendental functions
crate::unary_strided_primitive!(exp_strided_primitive, |x_val: T| {
    let x_f64 = x_val.to_f64().unwrap_or(0.0);
    T::from(x_f64.exp()).unwrap_or(x_val)
});

crate::unary_strided_primitive!(log_strided_primitive, |x_val: T| {
    let x_f64 = x_val.to_f64().unwrap_or(0.0);
    T::from(x_f64.ln()).unwrap_or(x_val)
});

crate::unary_strided_primitive!(abs_strided_primitive, |x_val: T| {
    let x_f64 = x_val.to_f64().unwrap_or(0.0);
    T::from(x_f64.abs()).unwrap_or(x_val)
});

crate::unary_csr_primitive!(abs_csr_primitive, |x_val: T| {
    let x_f64 = x_val.to_f64().unwrap_or(0.0);
    T::from(x_f64.abs()).unwrap_or(x_val)
});

crate::unary_strided_primitive!(sqrt_strided_primitive, |x_val: T| {
    let x_f64 = x_val.to_f64().unwrap_or(0.0);
    T::from(x_f64.sqrt()).unwrap_or(x_val)
});

crate::unary_strided_primitive!(sin_strided_primitive, |x_val: T| {
    let x_f64 = x_val.to_f64().unwrap_or(0.0);
    T::from(x_f64.sin()).unwrap_or(x_val)
});

crate::unary_strided_primitive!(cos_strided_primitive, |x_val: T| {
    let x_f64 = x_val.to_f64().unwrap_or(0.0);
    T::from(x_f64.cos()).unwrap_or(x_val)
});

crate::unary_strided_primitive!(isnan_strided_primitive, |x_val: T| {
    let x_f64 = x_val.to_f64().unwrap_or(0.0);
    if x_f64.is_nan() { T::one() } else { T::zero() }
});

crate::unary_strided_primitive!(isinf_strided_primitive, |x_val: T| {
    let x_f64 = x_val.to_f64().unwrap_or(0.0);
    if x_f64.is_infinite() { T::one() } else { T::zero() }
});

crate::unary_strided_primitive!(isfinite_strided_primitive, |x_val: T| {
    let x_f64 = x_val.to_f64().unwrap_or(0.0);
    if x_f64.is_finite() { T::one() } else { T::zero() }
});
