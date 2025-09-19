//! Numerical stability utilities for automatic differentiation
//!
//! This module provides mathematically rigorous handling of edge cases,
//! numerical stability, and gradient bounds checking for production ML workflows.

use std::f64;

/// Numerical constants for stability checks
pub mod constants {
    /// Machine epsilon for f64
    pub const EPSILON: f64 = f64::EPSILON;

    /// Threshold for considering a value as zero
    pub const ZERO_THRESHOLD: f64 = 1e-15;

    /// Maximum finite value before considering overflow
    pub const MAX_FINITE: f64 = f64::MAX / 2.0;

    /// Minimum positive value before considering underflow
    pub const MIN_POSITIVE: f64 = f64::MIN_POSITIVE * 2.0;

    /// Gradient clipping threshold to prevent explosion
    pub const GRADIENT_CLIP_THRESHOLD: f64 = 1e10;

    /// Step size for numerical differentiation
    pub const NUMERICAL_DIFF_STEP: f64 = 1e-8;
}

/// Numerical stability utilities
pub struct NumericalStability;

impl NumericalStability {
    /// Check if a value is effectively zero
    pub fn is_effectively_zero(value: f64) -> bool {
        value.abs() < constants::ZERO_THRESHOLD
    }

    /// Check if a value will cause overflow in exponential
    pub fn will_exp_overflow(value: f64) -> bool {
        value > 700.0 // exp(700) ≈ 1e304, near f64::MAX
    }

    /// Check if a value will cause underflow in exponential
    pub fn will_exp_underflow(value: f64) -> bool {
        value < -700.0 // exp(-700) ≈ 1e-304, near 0
    }

    /// Safe division with infinity handling
    pub fn safe_divide(numerator: f64, denominator: f64) -> f64 {
        if Self::is_effectively_zero(denominator) {
            if Self::is_effectively_zero(numerator) {
                f64::NAN // 0/0 is undefined
            } else if numerator > 0.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            }
        } else {
            numerator / denominator
        }
    }

    /// Safe square root with complex number handling
    pub fn safe_sqrt(value: f64) -> f64 {
        if value < 0.0 {
            f64::NAN // sqrt of negative is complex
        } else if Self::is_effectively_zero(value) {
            0.0
        } else {
            value.sqrt()
        }
    }

    /// Safe logarithm with domain checking
    pub fn safe_log(value: f64) -> f64 {
        if value <= 0.0 {
            if Self::is_effectively_zero(value) {
                f64::NEG_INFINITY // log(0) = -∞
            } else {
                f64::NAN // log(negative) is complex
            }
        } else {
            value.ln()
        }
    }

    /// Safe power operation with domain checking
    pub fn safe_pow(base: f64, exponent: f64) -> f64 {
        if base == 0.0 {
            if exponent > 0.0 {
                0.0
            } else if exponent == 0.0 {
                1.0 // 0^0 = 1 by convention
            } else {
                f64::INFINITY // 0^(-n) = ∞
            }
        } else if base < 0.0 && exponent.fract() != 0.0 {
            f64::NAN // negative^fractional is complex
        } else {
            base.powf(exponent)
        }
    }

    /// Clip gradient to prevent explosion (preserves mathematical correctness)
    pub fn clip_gradient(gradient: f64) -> f64 {
        if gradient.is_nan() {
            f64::NAN // Preserve NaN for mathematical correctness
        } else if gradient.is_infinite() {
            gradient // Preserve infinity for mathematical correctness
        } else if gradient.abs() > constants::GRADIENT_CLIP_THRESHOLD {
            if gradient > 0.0 {
                constants::GRADIENT_CLIP_THRESHOLD
            } else {
                -constants::GRADIENT_CLIP_THRESHOLD
            }
        } else {
            gradient
        }
    }

    /// Check if gradient is numerically stable
    pub fn is_gradient_stable(gradient: f64) -> bool {
        gradient.is_finite() && gradient.abs() < constants::GRADIENT_CLIP_THRESHOLD
    }

    /// Safe reciprocal with infinity handling
    pub fn safe_reciprocal(value: f64) -> f64 {
        Self::safe_divide(1.0, value)
    }

    /// Safe derivative of sqrt: 1/(2*sqrt(x))
    pub fn safe_sqrt_derivative(value: f64) -> f64 {
        if value < 0.0 {
            f64::NAN // sqrt of negative is complex
        } else if Self::is_effectively_zero(value) {
            f64::INFINITY // derivative approaches infinity at x=0
        } else {
            0.5 / value.sqrt()
        }
    }

    /// Safe derivative of log: 1/x
    pub fn safe_log_derivative(value: f64) -> f64 {
        Self::safe_reciprocal(value)
    }

    /// Safe derivative of reciprocal: -1/x^2
    pub fn safe_reciprocal_derivative(value: f64) -> f64 {
        if Self::is_effectively_zero(value) {
            f64::NEG_INFINITY // derivative approaches -∞ at x=0
        } else {
            -1.0 / (value * value)
        }
    }

    /// Safe derivative of power: n*x^(n-1)
    pub fn safe_power_derivative(base: f64, exponent: f64) -> f64 {
        if exponent == 0.0 {
            0.0 // derivative of constant is 0
        } else if exponent == 1.0 {
            1.0 // derivative of x is 1
        } else if Self::is_effectively_zero(base) {
            if exponent > 1.0 {
                0.0 // x^n where n>1 has derivative 0 at x=0
            } else if exponent == 1.0 {
                1.0 // x^1 has derivative 1
            } else {
                f64::INFINITY // x^n where n<1 has infinite derivative at x=0
            }
        } else if base < 0.0 && exponent.fract() != 0.0 {
            f64::NAN // negative^fractional is complex
        } else {
            exponent * base.powf(exponent - 1.0)
        }
    }

    /// Validate and sanitize gradient vector (preserves mathematical correctness)
    pub fn sanitize_gradients(gradients: &mut [f64]) {
        for grad in gradients.iter_mut() {
            // Only clip finite gradients that are too large
            // Preserve NaN and infinity for mathematical correctness
            if grad.is_finite() && grad.abs() > constants::GRADIENT_CLIP_THRESHOLD {
                *grad = Self::clip_gradient(*grad);
            }
        }
    }

    /// Check if any gradient in vector is unstable
    pub fn has_unstable_gradients(gradients: &[f64]) -> bool {
        gradients.iter().any(|&g| !Self::is_gradient_stable(g))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_is_effectively_zero() {
        assert!(NumericalStability::is_effectively_zero(0.0));
        assert!(NumericalStability::is_effectively_zero(1e-16));
        assert!(!NumericalStability::is_effectively_zero(1e-10));
    }

    #[test]
    fn test_safe_divide() {
        // Normal division
        assert_relative_eq!(NumericalStability::safe_divide(6.0, 2.0), 3.0);

        // Division by zero
        assert_eq!(NumericalStability::safe_divide(1.0, 0.0), f64::INFINITY);
        assert_eq!(
            NumericalStability::safe_divide(-1.0, 0.0),
            f64::NEG_INFINITY
        );

        // 0/0 case
        assert!(NumericalStability::safe_divide(0.0, 0.0).is_nan());
    }

    #[test]
    fn test_safe_sqrt() {
        // Normal sqrt
        assert_relative_eq!(NumericalStability::safe_sqrt(4.0), 2.0);

        // sqrt(0)
        assert_eq!(NumericalStability::safe_sqrt(0.0), 0.0);

        // sqrt(negative)
        assert!(NumericalStability::safe_sqrt(-1.0).is_nan());
    }

    #[test]
    fn test_safe_log() {
        // Normal log
        assert_relative_eq!(NumericalStability::safe_log(f64::consts::E), 1.0);

        // log(0)
        assert_eq!(NumericalStability::safe_log(0.0), f64::NEG_INFINITY);

        // log(negative)
        assert!(NumericalStability::safe_log(-1.0).is_nan());
    }

    #[test]
    fn test_safe_pow() {
        // Normal power
        assert_relative_eq!(NumericalStability::safe_pow(2.0, 3.0), 8.0);

        // 0^positive
        assert_eq!(NumericalStability::safe_pow(0.0, 2.0), 0.0);

        // 0^0
        assert_eq!(NumericalStability::safe_pow(0.0, 0.0), 1.0);

        // 0^negative
        assert_eq!(NumericalStability::safe_pow(0.0, -1.0), f64::INFINITY);

        // negative^fractional
        assert!(NumericalStability::safe_pow(-1.0, 0.5).is_nan());
    }

    #[test]
    fn test_clip_gradient() {
        // Normal gradient
        assert_relative_eq!(NumericalStability::clip_gradient(1.0), 1.0);

        // Large positive gradient
        assert_eq!(
            NumericalStability::clip_gradient(1e15),
            constants::GRADIENT_CLIP_THRESHOLD
        );

        // Large negative gradient
        assert_eq!(
            NumericalStability::clip_gradient(-1e15),
            -constants::GRADIENT_CLIP_THRESHOLD
        );

        // Infinite gradient (preserved for mathematical correctness)
        assert_eq!(
            NumericalStability::clip_gradient(f64::INFINITY),
            f64::INFINITY
        );
        assert_eq!(
            NumericalStability::clip_gradient(f64::NEG_INFINITY),
            f64::NEG_INFINITY
        );

        // NaN gradient (preserved)
        assert!(NumericalStability::clip_gradient(f64::NAN).is_nan());
    }

    #[test]
    fn test_safe_derivatives() {
        // sqrt derivative
        assert_relative_eq!(NumericalStability::safe_sqrt_derivative(4.0), 0.25);
        assert_eq!(NumericalStability::safe_sqrt_derivative(0.0), f64::INFINITY);
        assert!(NumericalStability::safe_sqrt_derivative(-1.0).is_nan());

        // log derivative
        assert_relative_eq!(NumericalStability::safe_log_derivative(2.0), 0.5);
        assert_eq!(NumericalStability::safe_log_derivative(0.0), f64::INFINITY);

        // reciprocal derivative
        assert_relative_eq!(NumericalStability::safe_reciprocal_derivative(2.0), -0.25);
        assert_eq!(
            NumericalStability::safe_reciprocal_derivative(0.0),
            f64::NEG_INFINITY
        );

        // power derivative
        assert_relative_eq!(NumericalStability::safe_power_derivative(2.0, 3.0), 12.0); // 3*2^2
        assert_eq!(
            NumericalStability::safe_power_derivative(0.0, 0.5),
            f64::INFINITY
        );
        assert_eq!(NumericalStability::safe_power_derivative(0.0, 2.0), 0.0);
    }

    #[test]
    fn test_sanitize_gradients() {
        let mut gradients = vec![1.0, f64::INFINITY, -1e15, f64::NAN, 0.0];
        NumericalStability::sanitize_gradients(&mut gradients);

        assert_relative_eq!(gradients[0], 1.0); // Normal gradient unchanged
        assert_eq!(gradients[1], f64::INFINITY); // Infinity preserved
        assert_eq!(gradients[2], -constants::GRADIENT_CLIP_THRESHOLD); // Large negative clipped
        assert!(gradients[3].is_nan()); // NaN preserved
        assert_relative_eq!(gradients[4], 0.0); // Zero unchanged
    }
}
