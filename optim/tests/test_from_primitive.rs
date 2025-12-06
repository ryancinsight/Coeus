//! Test that FromPrimitive trait works correctly for optimizer use cases

use dtype::float::{Float32, Float64};
use dtype::traits::FloatExt;
use num_traits::{Float, FromPrimitive};

#[test]
fn test_float32_from_f64_learning_rate() {
    // This is the critical use case for optimizers: converting f64 learning rates to Float32
    let lr = 0.001_f64;
    let lr_scalar = Float32::from_f64(lr).unwrap();

    assert!((lr_scalar.get() - 0.001).abs() < 1e-6);
}

#[test]
fn test_float64_from_f64_learning_rate() {
    // This is the critical use case for optimizers: converting f64 learning rates to Float64
    let lr = 0.001_f64;
    let lr_scalar = Float64::from_f64(lr).unwrap();

    assert!((lr_scalar.get() - 0.001).abs() < 1e-12);
}

#[test]
fn test_optimizer_scalar_conversions() {
    // Test all the scalar conversions needed by optimizers

    // Learning rate
    let lr = Float32::from_f64(0.01).unwrap();
    assert_eq!(lr.get(), 0.01);

    // Momentum
    let momentum = Float32::from_f64(0.9).unwrap();
    assert_eq!(momentum.get(), 0.9);

    // Weight decay
    let weight_decay = Float32::from_f64(0.0001).unwrap();
    assert!((weight_decay.get() - 0.0001).abs() < 1e-6);

    // Dampening
    let dampening = Float32::from_f64(0.1).unwrap();
    assert!((dampening.get() - 0.1).abs() < 1e-6);

    // Epsilon (for Adam)
    let epsilon = Float32::from_f64(1e-8).unwrap();
    assert!((epsilon.get() - 1e-8).abs() < 1e-12);
}

#[test]
fn test_from_primitive_unwrap_pattern() {
    // Verify that the T::from_f64(lr).unwrap() pattern works
    // This is the exact pattern used in the optimizer implementation

    let lr = 0.001_f64;

    // Float32
    let lr_f32 = Float32::from_f64(lr).unwrap();
    assert!((lr_f32.get() - lr as f32).abs() < 1e-9);

    // Float64
    let lr_f64 = Float64::from_f64(lr).unwrap();
    assert_eq!(lr_f64.get(), lr);
}

#[test]
fn test_from_primitive_special_values() {
    // Verify special values work correctly

    // Infinity
    let inf_f32 = Float32::from_f64(f64::INFINITY).unwrap();
    assert!(inf_f32.is_infinite());

    let inf_f64 = Float64::from_f64(f64::INFINITY).unwrap();
    assert!(inf_f64.is_infinite());

    // NaN
    let nan_f32 = Float32::from_f64(f64::NAN).unwrap();
    assert!(nan_f32.is_nan());

    let nan_f64 = Float64::from_f64(f64::NAN).unwrap();
    assert!(nan_f64.is_nan());

    // Zero
    let zero_f32 = Float32::from_f64(0.0).unwrap();
    assert_eq!(zero_f32.get(), 0.0);

    let zero_f64 = Float64::from_f64(0.0).unwrap();
    assert_eq!(zero_f64.get(), 0.0);
}
