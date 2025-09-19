//! Property-based tests for mathematical correctness
//!
//! This module implements property-based testing using the proptest crate
//! to validate mathematical properties across a wide range of inputs,
//! ensuring robustness and correctness under diverse conditions.

use proptest::prelude::*;
use approx::assert_relative_eq;
use crate::Tensor;

/// Strategy for generating valid f64 values for tensor operations
fn finite_f64() -> impl Strategy<Value = f64> {
    // Generate finite f64 values, avoiding extreme values that might cause issues
    (-1e10..1e10).prop_filter("Must be finite", |x| x.is_finite())
}

/// Strategy for generating small positive values to avoid overflow
fn small_positive_f64() -> impl Strategy<Value = f64> {
    (1e-10..1e10).prop_filter("Must be small positive", |x| x.is_finite() && *x > 0.0)
}

/// Property: Addition is commutative
/// ∀a,b ∈ ℝ: a + b = b + a
proptest! {
    #[test]
    fn test_addition_commutative(a in finite_f64(), b in finite_f64()) {
        let tensor_a = Tensor::scalar(a);
        let tensor_b = Tensor::scalar(b);

        let result_ab = (&tensor_a + &tensor_b).unwrap();
        let result_ba = (&tensor_b + &tensor_a).unwrap();

        let val_ab = result_ab.as_scalar().unwrap();
        let val_ba = result_ba.as_scalar().unwrap();

        // Allow for small floating-point errors
        assert!((val_ab - val_ba).abs() < 1e-12,
                "Addition should be commutative: {} + {} = {} + {}", a, b, a, b);
    }
}

/// Property: Addition is associative
/// ∀a,b,c ∈ ℝ: (a + b) + c = a + (b + c)
proptest! {
    #[test]
    fn test_addition_associative(a in finite_f64(), b in finite_f64(), c in finite_f64()) {
        let tensor_a = Tensor::scalar(a);
        let tensor_b = Tensor::scalar(b);
        let tensor_c = Tensor::scalar(c);

        let result1 = ((&tensor_a + &tensor_b).unwrap() + &tensor_c).unwrap();
        let result2 = (&tensor_a + (&tensor_b + &tensor_c).unwrap()).unwrap();

        let val1 = result1.as_scalar().unwrap();
        let val2 = result2.as_scalar().unwrap();

        assert!((val1 - val2).abs() < 1e-12,
                "Addition should be associative: ({:?} + {:?}) + {:?} = {:?} + ({:?} + {:?})",
                a, b, c, a, b, c);
    }
}

/// Property: Multiplication is commutative
/// ∀a,b ∈ ℝ: a × b = b × a
proptest! {
    #[test]
    fn test_multiplication_commutative(a in finite_f64(), b in finite_f64()) {
        let tensor_a = Tensor::scalar(a);
        let tensor_b = Tensor::scalar(b);

        let result_ab = (&tensor_a * &tensor_b).unwrap();
        let result_ba = (&tensor_b * &tensor_a).unwrap();

        let val_ab = result_ab.as_scalar().unwrap();
        let val_ba = result_ba.as_scalar().unwrap();

        assert!((val_ab - val_ba).abs() < 1e-12,
                "Multiplication should be commutative: {} × {} = {} × {}", a, b, a, b);
    }
}

/// Property: Multiplication is associative
/// ∀a,b,c ∈ ℝ: (a × b) × c = a × (b × c)
proptest! {
    #[test]
    fn test_multiplication_associative(a in finite_f64(), b in finite_f64(), c in finite_f64()) {
        let tensor_a = Tensor::scalar(a);
        let tensor_b = Tensor::scalar(b);
        let tensor_c = Tensor::scalar(c);

        let result1 = ((&tensor_a * &tensor_b).unwrap() * &tensor_c).unwrap();
        let result2 = (&tensor_a * (&tensor_b * &tensor_c).unwrap()).unwrap();

        let val1 = result1.as_scalar().unwrap();
        let val2 = result2.as_scalar().unwrap();

        assert!((val1 - val2).abs() < 1e-12,
                "Multiplication should be associative");
    }
}

/// Property: Distributive law
/// ∀a,b,c ∈ ℝ: a × (b + c) = a × b + a × c
proptest! {
    #[test]
    fn test_distributive_law(a in finite_f64(), b in finite_f64(), c in finite_f64()) {
        let tensor_a = Tensor::scalar(a);
        let tensor_b = Tensor::scalar(b);
        let tensor_c = Tensor::scalar(c);

        let result1 = (&tensor_a * (&tensor_b + &tensor_c).unwrap()).unwrap();
        let result2 = ((&tensor_a * &tensor_b).unwrap() + (&tensor_a * &tensor_c).unwrap()).unwrap();

        let val1 = result1.as_scalar().unwrap();
        let val2 = result2.as_scalar().unwrap();

        assert!((val1 - val2).abs() < 1e-12,
                "Distributive law should hold: {} × ({} + {}) = {} × {} + {} × {}",
                a, b, c, a, b, a, c);
    }
}

/// Property: Gradient of identity function
/// ∀x ∈ ℝ: d/dx(x) = 1
proptest! {
    #[test]
    fn test_identity_gradient(x in finite_f64()) {
        let mut tensor_x = Tensor::scalar(x);
        tensor_x.set_requires_grad(true);

        let y = &tensor_x; // Identity function
        y.backward().unwrap();

        let grad = tensor_x.grad().unwrap().as_scalar().unwrap();

        assert!((grad - 1.0).abs() < 1e-12,
                "Gradient of identity function should be 1, got {} for x = {}", grad, x);
    }
}

/// Property: Gradient of constant
/// ∀c ∈ ℝ: d/dc(c) = 0
proptest! {
    #[test]
    fn test_constant_gradient(c in finite_f64()) {
        let mut tensor_x = Tensor::scalar(1.0); // Dummy variable
        tensor_x.set_requires_grad(true);

        let y = Tensor::scalar(c); // Constant function
        y.backward().unwrap();

        let grad = tensor_x.grad().unwrap().as_scalar().unwrap();

        // Since y doesn't depend on x, gradient should be 0
        assert!(grad.abs() < 1e-12,
                "Gradient of constant should be 0, got {} for constant = {}", grad, c);
    }
}

/// Property: Power rule for derivatives
/// ∀x ∈ ℝ⁺: d/dx(x^n) = n × x^(n-1)
proptest! {
    #[test]
    fn test_power_rule(x in small_positive_f64(), n in 1..10) {
        let mut tensor_x = Tensor::scalar(x);
        tensor_x.set_requires_grad(true);

        // Compute x^n using repeated multiplication
        let mut result = Tensor::scalar(1.0);
        for _ in 0..n {
            result = (&result * &tensor_x).unwrap();
        }

        result.backward().unwrap();

        let grad = tensor_x.grad().unwrap().as_scalar().unwrap();
        let expected = n as f64 * x.powf((n - 1) as f64);

        assert!((grad - expected).abs() < 1e-10,
                "Power rule failed: d/dx(x^{}) = {} × x^{} = {}, got {}",
                n, n, n-1, expected, grad);
    }
}

/// Property: Chain rule
/// ∀f,g: d/dx(f(g(x))) = f'(g(x)) × g'(x)
proptest! {
    #[test]
    fn test_chain_rule(x in finite_f64()) {
        let mut tensor_x = Tensor::scalar(x);
        tensor_x.set_requires_grad(true);

        // f(g(x)) = sin(x²)
        // f'(g) = cos(g), g'(x) = 2x
        // So f'(g(x)) × g'(x) = cos(x²) × 2x
        let x_squared = (&tensor_x * &tensor_x).unwrap();
        let sin_x_squared = x_squared.sin();

        sin_x_squared.backward().unwrap();

        let grad = tensor_x.grad().unwrap().as_scalar().unwrap();
        let expected = (x * x).cos() * 2.0 * x;

        assert!((grad - expected).abs() < 1e-12,
                "Chain rule failed: d/dx(sin(x²)) = cos(x²) × 2x = {}, got {}",
                expected, grad);
    }
}

/// Property: Linearity of differentiation
/// ∀f,g: d/dx(af(x) + bg(x)) = a × f'(x) + b × g'(x)
proptest! {
    #[test]
    fn test_differentiation_linearity(x in finite_f64(), a in finite_f64(), b in finite_f64()) {
        let mut tensor_x = Tensor::scalar(x);
        tensor_x.set_requires_grad(true);

        // f(x) = x, g(x) = x²
        let f_x = &tensor_x;
        let g_x = (&tensor_x * &tensor_x).unwrap();

        // h(x) = a × f(x) + b × g(x) = a × x + b × x²
        let h_x = ((Tensor::scalar(a) * f_x).unwrap() + (Tensor::scalar(b) * &g_x).unwrap()).unwrap();

        h_x.backward().unwrap();

        let grad = tensor_x.grad().unwrap().as_scalar().unwrap();
        let expected = a + b * 2.0 * x; // a × 1 + b × 2x

        assert!((grad - expected).abs() < 1e-12,
                "Differentiation linearity failed: d/dx({}x + {}x²) = {} + {}×2x = {}, got {}",
                a, b, a, b, expected, grad);
    }
}

/// Property: Gradient accumulation
/// If a tensor is used multiple times in computation, gradients should accumulate
proptest! {
    #[test]
    fn test_gradient_accumulation(x in finite_f64()) {
        let mut tensor_x = Tensor::scalar(x);
        tensor_x.set_requires_grad(true);

        // y = x + x = 2x
        let y = (&tensor_x + &tensor_x).unwrap();
        y.backward().unwrap();

        let grad = tensor_x.grad().unwrap().as_scalar().unwrap();
        let expected = 2.0; // ∂(x + x)/∂x = 1 + 1 = 2

        assert!((grad - expected).abs() < 1e-12,
                "Gradient accumulation failed: d/dx(x + x) = 2, got {}", grad);
    }
}

/// Property: Broadcasting preserves mathematical correctness
proptest! {
    #[test]
    fn test_broadcasting_mathematical_correctness(scalar in finite_f64(), vector_len in 1..100) {
        let scalar_tensor = Tensor::scalar(scalar);

        // Create a vector of ones
        let vector_data = vec![1.0; vector_len];
        let vector_tensor = Tensor::from_vec(vector_data, vec![vector_len]);

        // scalar + vector should broadcast correctly
        let result = (&scalar_tensor + &vector_tensor).unwrap();

        // Each element should be scalar + 1.0
        let result_data = result.data();
        for &val in result_data {
            assert!((val - (scalar + 1.0)).abs() < 1e-12,
                    "Broadcasting failed: expected {}, got {}", scalar + 1.0, val);
        }
    }
}

/// Property: Matrix multiplication shape constraints
proptest! {
    #[test]
    fn test_matrix_multiplication_shapes(m in 1..20, n in 1..20, p in 1..20) {
        // Create matrices A (m×n) and B (n×p), result should be (m×p)
        let a_data = vec![1.0; m * n];
        let b_data = vec![2.0; n * p];

        let a = Tensor::from_vec(a_data, vec![m, n]);
        let b = Tensor::from_vec(b_data, vec![n, p]);

        let result = (a @ b).unwrap();

        assert_eq!(result.shape(), &[m, p],
                   "Matrix multiplication shape error: ({},{}) @ ({},{}) should give ({},{})",
                   m, n, n, p, m, p);
    }
}

/// Property: Transpose is involutive
/// ∀A: (A^T)^T = A
proptest! {
    #[test]
    fn test_transpose_involutive(rows in 1..10, cols in 1..10) {
        let data = vec![1.0; rows * cols];
        let tensor = Tensor::from_vec(data, vec![rows, cols]);

        let transposed_once = tensor.t().unwrap();
        let transposed_twice = transposed_once.t().unwrap();

        // Should be back to original shape
        assert_eq!(transposed_twice.shape(), &[rows, cols],
                   "Double transpose should restore original shape");

        // Elements should be identical
        let original_data = tensor.data();
        let double_transposed_data = transposed_twice.data();

        for (orig, double_t) in original_data.iter().zip(double_transposed_data.iter()) {
            assert_eq!(orig, double_t, "Double transpose should preserve elements");
        }
    }
}
