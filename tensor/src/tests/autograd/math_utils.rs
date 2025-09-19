/// Mathematical utilities and validation functions
/// This module contains utility functions for mathematical validation
use approx::assert_relative_eq;

/// Numerical gradient verification using central difference method
/// Validates computed gradients against analytical derivatives with high precision
pub fn verify_numerical_gradient<F>(f: F, x: f64, h: f64, expected_grad: f64, tolerance: f64)
where
    F: Fn(f64) -> f64,
{
    // Central difference: (f(x+h) - f(x-h)) / (2h)
    let numerical_grad = (f(x + h) - f(x - h)) / (2.0 * h);

    // Validate against expected analytical gradient
    assert_relative_eq!(numerical_grad, expected_grad, epsilon = tolerance);

    // Test with different step sizes to ensure stability
    let h_small = h / 4.0; // Less aggressive reduction to avoid precision issues
    let numerical_grad_small = (f(x + h_small) - f(x - h_small)) / (2.0 * h_small);

    // Both should be reasonably close to expected
    let error_large = (numerical_grad - expected_grad).abs();
    let error_small = (numerical_grad_small - expected_grad).abs();

    // Allow for some numerical precision variation but ensure both are within tolerance
    assert!(error_large < tolerance * 2.0,
        "Large step size error too high: {} > {}", error_large, tolerance * 2.0);
    assert!(error_small < tolerance * 2.0,
        "Small step size error too high: {} > {}", error_small, tolerance * 2.0);
}


/// Test mathematical identities and properties
#[cfg(test)]
mod math_validation_tests {
    use super::*;

    #[test]
    fn test_exponential_properties() {
        // Test that exp(0) = 1
        assert_relative_eq!(0.0f64.exp(), 1.0, epsilon = 1e-10);

        // Test exp(-∞) = 0 (mathematically correct)
        assert_relative_eq!(f64::NEG_INFINITY.exp(), 0.0, epsilon = 1e-10);

        // Test exp(∞) = ∞
        assert_eq!(f64::INFINITY.exp(), f64::INFINITY);

        // Test exp(NaN) = NaN
        assert!(f64::NAN.exp().is_nan());

        // Test exp(x) > 0 for all finite x
        let test_values = [-10.0f64, -1.0f64, 0.0f64, 1.0f64, 10.0f64];
        for &x in &test_values {
            let result = x.exp();
            assert!(result > 0.0, "exp({}) = {} should be positive", x, result);
        }
    }

    #[test]
    fn test_logarithmic_properties() {
        // Test that log(1) = 0
        let log_1 = 1.0f64.ln();
        assert_relative_eq!(log_1, 0.0, epsilon = 1e-10);

        // Test log(exp(x)) = x for reasonable x
        let test_values = [-5.0f64, -1.0f64, 0.0f64, 1.0f64, 5.0f64];
        for &x in &test_values {
            let log_exp = x.exp().ln();
            assert_relative_eq!(log_exp, x, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_trigonometric_identities() {
        // Test sin²(x) + cos²(x) = 1
        let test_values = [0.0, std::f64::consts::PI / 4.0, std::f64::consts::PI / 2.0, std::f64::consts::PI];
        for &x in &test_values {
            let sin_sq = x.sin().powi(2);
            let cos_sq = x.cos().powi(2);
            let sum = sin_sq + cos_sq;
            assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_numerical_gradient_accuracy() {
        // Test that numerical gradient computation is accurate
        let f = |x: f64| x * x + 2.0 * x + 1.0; // f(x) = x² + 2x + 1
        let df_dx = |x: f64| 2.0 * x + 2.0; // f'(x) = 2x + 2

        // Test at x = 1.0, f'(1) = 4.0
        verify_numerical_gradient(f, 1.0, 1e-5, df_dx(1.0), 1e-6);

        // Test at x = -1.0, f'(-1) = 0.0
        verify_numerical_gradient(f, -1.0, 1e-5, df_dx(-1.0), 1e-6);

        // Test at x = 0.0, f'(0) = 2.0
        verify_numerical_gradient(f, 0.0, 1e-5, df_dx(0.0), 1e-6);
    }
}
