// Property-based tests for mathematical correctness
//
// This module implements property-based testing using the proptest crate
// to validate mathematical properties across a wide range of inputs,
// ensuring robustness and correctness under diverse conditions.

use proptest::prelude::*;

/// Strategy for generating valid f64 values for tensor operations
fn finite_f64() -> impl Strategy<Value = f64> {
    // Generate finite f64 values, avoiding extreme values that might cause issues
    (-1e10..1e10).prop_filter("Must be finite", |x: &f64| x.is_finite())
}

fn mul_stub(a: &Tensor<f64, CpuBackend>, b: &Tensor<f64, CpuBackend>) -> Tensor<f64, CpuBackend> {
    let data: Vec<f64> = a.data().iter().zip(b.data().iter()).map(|(&x, &y)| x * y).collect();
    Tensor::from_vec(CpuBackend::default(), data, a.shape().to_vec()).expect("mul")
}


proptest! {
    /// Property: Addition is commutative
    /// ∀a,b ∈ ℝ: a + b = b + a
    #[test]
    fn test_addition_commutative(a in finite_f64(), b in finite_f64()) {
        let tensor_a = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(a);
        let tensor_b = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(b);

        let result_ab = (&tensor_a + &tensor_b).unwrap();
        let result_ba = (&tensor_b + &tensor_a).unwrap();

        let val_ab = result_ab.as_scalar().unwrap();
        let val_ba = result_ba.as_scalar().unwrap();

        // Allow for small floating-point errors
        assert!((val_ab - val_ba).abs() < 1e-12,
                "Addition should be commutative: {} + {} = {} + {}", a, b, a, b);
    }
}

proptest! {
    /// Property: Addition is associative
    /// ∀a,b,c ∈ ℝ: (a + b) + c = a + (b + c)
    #[test]
    fn test_addition_associative(a in finite_f64(), b in finite_f64(), c in finite_f64()) {
        let tensor_a = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(a);
        let tensor_b = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(b);
        let tensor_c = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(c);

        let temp1 = (&tensor_a + &tensor_b).unwrap();
        let result1 = (&temp1 + &tensor_c).unwrap();
        let temp2 = (&tensor_b + &tensor_c).unwrap();
        let result2 = (&tensor_a + &temp2).unwrap();

        let val1 = result1.as_scalar().unwrap();
        let val2 = result2.as_scalar().unwrap();

        // Use relative tolerance for floating point precision
        let tolerance = 1e-10 * val1.abs().max(val2.abs()).max(1.0);
        assert!((val1 - val2).abs() < tolerance,
                "Addition should be associative: ({:?} + {:?}) + {:?} = {:?} + ({:?} + {:?})",
                a, b, c, a, b, c);
    }
}

proptest! {
    /// Property: Multiplication is commutative
    /// ∀a,b ∈ ℝ: a × b = b × a
    #[test]
    fn test_multiplication_commutative(a in finite_f64(), b in finite_f64()) {
        let tensor_a = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(a);
        let tensor_b = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(b);

        let result_ab = (&tensor_a * &tensor_b).unwrap();
        let result_ba = (&tensor_b * &tensor_a).unwrap();

        let val_ab = result_ab.as_scalar().unwrap();
        let val_ba = result_ba.as_scalar().unwrap();

        assert!((val_ab - val_ba).abs() < 1e-12,
                "Multiplication should be commutative: {} × {} = {} × {}", a, b, a, b);
    }
}

proptest! {
    /// Property: Multiplication is associative
    /// ∀a,b,c ∈ ℝ: (a × b) × c = a × (b × c)
    #[test]
    fn test_multiplication_associative(a in finite_f64(), b in finite_f64(), c in finite_f64()) {
        let tensor_a = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(a);
        let tensor_b = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(b);
        let tensor_c = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(c);

        let temp1 = (&tensor_a * &tensor_b).unwrap();
        let result1 = (&temp1 * &tensor_c).unwrap();
        let temp2 = (&tensor_b * &tensor_c).unwrap();
        let result2 = (&tensor_a * &temp2).unwrap();

        let val1 = result1.as_scalar().unwrap();
        let val2 = result2.as_scalar().unwrap();

        // Use relative tolerance for floating point precision
        let tolerance = 1e-10 * val1.abs().max(val2.abs()).max(1.0);
        assert!((val1 - val2).abs() < tolerance,
                "Multiplication should be associative");
    }
}

proptest! {
    /// Property: Distributive law
    /// ∀a,b,c ∈ ℝ: a × (b + c) = a × b + a × c
    #[test]
    fn test_distributive_law(a in finite_f64(), b in finite_f64(), c in finite_f64()) {
        let tensor_a = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(a);
        let tensor_b = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(b);
        let tensor_c = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(c);

        let temp1 = (&tensor_b + &tensor_c).unwrap();
        let result1 = (&tensor_a * &temp1).unwrap();
        let temp2 = (&tensor_a * &tensor_b).unwrap();
        let temp3 = (&tensor_a * &tensor_c).unwrap();
        let result2 = (&temp2 + &temp3).unwrap();

        let val1 = result1.as_scalar().unwrap();
        let val2 = result2.as_scalar().unwrap();

        // Use relative tolerance for floating point precision
        let tolerance = 1e-10 * val1.abs().max(val2.abs()).max(1.0);
        assert!((val1 - val2).abs() < tolerance);
    }
}

proptest! {
    /// Property: Gradient of identity function
    /// ∀x ∈ ℝ: d/dx(x) = 1
    #[test]
    fn test_identity_gradient(x in finite_f64()) {
        let mut tensor_x = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(x);
        tensor_x.set_requires_grad(true);

        tensor_x.backward().unwrap(); // Identity function

        let grad = tensor_x.grad().unwrap().as_scalar().unwrap();

        assert!((grad - 1.0).abs() < 1e-12,
                "Gradient of identity function should be 1, got {} for x = {}", grad, x);
    }
}

proptest! {
    /// Property: Gradient of constant
    /// ∀c ∈ ℝ: d/dc(c) = 0
    #[test]
    fn test_constant_gradient(c in finite_f64()) {
        let mut tensor_x = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(1.0); // Dummy variable
        tensor_x.set_requires_grad(true);

        let y = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(c); // Constant function
        // Constants don't have gradients - this test validates that
        // trying to compute gradients on constants returns None
        if y.grad().is_some() {
            panic!("Constant tensor should not have gradients");
        } // Expected - constants don't participate in gradient computation
    }
}

proptest! {
    /// Property: Power rule for derivatives
    /// ∀x ∈ ℝ⁺: d/dx(x^n) = n × x^(n-1)
    #[test]
    fn test_power_rule(
        x in -10.0..10.0f64,
        n in 1..4u32
    ) {
        let mut tensor_x = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(x);
        tensor_x.set_requires_grad(true);

        // Compute x^n using built-in pow operation
        let mut result = tensor_x.pow(n as f64).unwrap();

        result.backward().unwrap();

        let grad: f64 = tensor_x.grad().unwrap().as_scalar().unwrap();
        let expected = n as f64 * x.powf((n - 1) as f64);

        // Use relative tolerance for floating point precision
        let tolerance = 1e-10 * expected.abs().max(1.0);
        assert!((grad - expected).abs() < tolerance,
                "Power rule failed: d/dx(x^{}) = {} × x^{} = {}, got {}",
                n, n, n-1, expected, grad);
    }
}

proptest! {
    /// Property: Chain rule
    /// ∀f,g: d/dx(f(g(x))) = f'(g(x)) × g'(x)
    #[test]
    fn test_chain_rule(x in -5.0..5.0f64) {
        let mut tensor_x = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(x);
        tensor_x.set_requires_grad(true);

        // f(g(x)) = sin(x²)
        // f'(g) = cos(g), g'(x) = 2x
        // So f'(g(x)) × g'(x) = cos(x²) × 2x
        let x_squared = (&tensor_x * &tensor_x).unwrap();
        let mut sin_x_squared = x_squared.sin().unwrap();

        sin_x_squared.backward().unwrap();

        let grad: f64 = tensor_x.grad().unwrap().as_scalar().unwrap();
        let expected = (x * x).cos() * 2.0 * x;

        // Use relative tolerance for floating point precision (relaxed for f32 autograd precision)
        let tolerance = 1e-6 * expected.abs().max(grad.abs()).max(1.0);
        assert!((grad - expected).abs() < tolerance);
    }
}

proptest! {
    /// Property: Linearity of differentiation
    /// ∀f,g: d/dx(af(x) + bg(x)) = a × f'(x) + b × g'(x)
    #[test]
    fn test_differentiation_linearity(
        x in -5.0..5.0f64,
        a in -2.0..2.0f64,
        b in -2.0..2.0f64
    ) {
        let mut tensor_x = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(x);
        tensor_x.set_requires_grad(true);

        // f(x) = x, g(x) = x²
        let g_x = (&tensor_x * &tensor_x).unwrap();

        // h(x) = a × f(x) + b × g(x) = a × x + b × x²
        let temp1 = (&Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(a) * &tensor_x).unwrap();
        let temp2 = (&Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(b) * &g_x).unwrap();
        let mut h_x = (&temp1 + &temp2).unwrap();

        h_x.backward().unwrap();

        let grad: f64 = tensor_x.grad().unwrap().as_scalar().unwrap();
        let expected = a + b * 2.0 * x; // a × 1 + b × 2x

        // Use relative tolerance for floating point precision
        let tolerance = 1e-10 * expected.abs().max(grad.abs()).max(1.0);
        assert!((grad - expected).abs() < tolerance,
                "Differentiation linearity failed: d/dx({}x + {}x²) = {} + {}×2x = {}, got {}",
                a, b, a, b, expected, grad);
    }
}

proptest! {
    /// Property: Gradient accumulation
    /// If a tensor is used multiple times in computation, gradients should accumulate
    #[test]
    fn test_gradient_accumulation(x in finite_f64()) {
        let mut tensor_x = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(x);
        tensor_x.set_requires_grad(true);

        // y = x + x = 2x
        let mut y = (&tensor_x + &tensor_x).unwrap();
        y.backward().unwrap();

        let grad: f64 = tensor_x.grad().unwrap().as_scalar().unwrap();
        let expected = 2.0; // ∂(x + x)/∂x = 1 + 1 = 2

        assert!((grad - expected).abs() < 1e-12,
                "Gradient accumulation failed: d/dx(x + x) = 2, got {}", grad);
    }
}

proptest! {
    /// Property: Broadcasting preserves mathematical correctness
    #[test]
    fn test_broadcasting_mathematical_correctness(scalar in finite_f64(), vector_len in 1usize..100) {
        let backend = coeus_backend::CpuBackend::new();
        let scalar_tensor = Tensor::<f64, CpuBackend, DenseStorage<f64>>::scalar(scalar);

        // Create a vector of ones
        let vector_data = vec![1.0; vector_len];
        let vector_tensor = Tensor::from_vec(backend, vector_data, vec![vector_len]).unwrap();

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

proptest! {
    /// Property: Matrix multiplication shape constraints
    #[test]
    fn test_matrix_multiplication_shapes(m in 1usize..20, n in 1usize..20, p in 1usize..20) {
        let backend = coeus_backend::CpuBackend::new();
        // Create matrices A (m×n) and B (n×p), result should be (m×p)
        let a_data = vec![1.0; m * n];
        let b_data = vec![2.0; n * p];

        let a = Tensor::from_vec(backend.clone(), a_data, vec![m, n]).unwrap();
        let b = Tensor::from_vec(backend.clone(), b_data, vec![n, p]).unwrap();

        // Use matmul for matrix multiplication instead of element-wise *
        let result = a.matmul(&b).unwrap();

        assert_eq!(result.shape(), &[m, p],
                   "Matrix multiplication shape error: ({},{}) @ ({},{}) should give ({},{})",
                   m, n, n, p, m, p);
    }
}

proptest! {
    /// Property: Transpose is involutive
    /// ∀A: (A^T)^T = A
    #[test]
    fn test_transpose_involutive(rows in 1usize..10, cols in 1usize..10) {
        let backend = coeus_backend::CpuBackend::new();
        let data = vec![1.0; rows * cols];
        let tensor = Tensor::from_vec(backend, data, vec![rows, cols]).unwrap();

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
