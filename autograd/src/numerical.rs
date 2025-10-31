//! Numerical gradient validation utilities
//!
//! This module provides tools for validating analytical gradients against
//! numerical gradients computed via finite differences.
//! Temporarily disabled during node-based autograd refactoring.

// Temporarily disabled during node-based autograd refactoring
// use crate::error::Result;
// use crate::variable::Variable;

/*
// Temporarily disabled - will be reimplemented with node-based autograd
/// Compute numerical gradient using finite differences
///
/// # Mathematical Definition
///
/// For a scalar function f(x), the numerical gradient is approximated as:
/// ```text
/// ∂f/∂x ≈ (f(x + ε) - f(x - ε)) / (2ε)
/// ```
///
/// This is the central difference formula, which has O(ε²) error compared to
/// O(ε) for forward differences.
///
/// # Arguments
/// * `f` - Function that takes a variable and returns a scalar variable
/// * `x` - Input variable at which to compute gradient
/// * `epsilon` - Step size for finite differences (default: 1e-5 for f32)
///
/// # Returns
/// Tensor containing numerical gradient with same shape as input
///
/// # Errors
/// Returns error if tensor operations fail
pub fn numerical_gradient<T, F>(
    f: F,
    x: &Variable<T>,
    epsilon: T,
) -> Result<Tensor<CpuBackend<Data = T>, DenseStorage<T>, T>>
where
    T: DataType + FloatExt,
    F: Fn(&Variable<T>) -> Variable<T>,
{
    let x_data = x.data();
    let shape = x_data.shape().dims();
    let n = x_data.len();

    let mut grad_data = Vec::with_capacity(n);

    // Compute gradient for each element using central differences
    for i in 0..n {
        // Create x + ε (perturb element i)
        let mut x_plus = x_data.as_slice().to_vec();
        x_plus[i] = x_plus[i] + epsilon;
        let x_plus_tensor = Tensor::from_vec(x_plus, shape)?;
        let x_plus_var = Variable::new(x_plus_tensor);

        // Create x - ε (perturb element i)
        let mut x_minus = x_data.as_slice().to_vec();
        x_minus[i] = x_minus[i] - epsilon;
        let x_minus_tensor = Tensor::from_vec(x_minus, shape)?;
        let x_minus_var = Variable::new(x_minus_tensor);

        // Compute f(x + ε) and f(x - ε)
        let f_plus = f(&x_plus_var);
        let f_minus = f(&x_minus_var);

        // Sum all output elements (handles both scalar and vector outputs)
        // For scalar output: sum([y]) = y
        // For vector output: sum([y1, y2, ...]) = y1 + y2 + ...
        // This matches backward() behavior which implicitly uses grad_output = ones_like(output)
        let f_plus_val: T = f_plus
            .data()
            .as_slice()
            .iter()
            .copied()
            .fold(T::zero(), |acc, x| acc + x);
        let f_minus_val: T = f_minus
            .data()
            .as_slice()
            .iter()
            .copied()
            .fold(T::zero(), |acc, x| acc + x);

        // Central difference: (f(x + ε) - f(x - ε)) / (2ε)
        let two = T::one() + T::one();
        let grad_i = (f_plus_val - f_minus_val) / (two * epsilon);

        grad_data.push(grad_i);
    }

    Ok(Tensor::from_vec(grad_data, shape)?)
}

/// Check if analytical and numerical gradients match within tolerance
///
/// # Arguments
/// * `analytical` - Gradient computed via backpropagation
/// * `numerical` - Gradient computed via finite differences
/// * `rtol` - Relative tolerance (default: 1e-5)
/// * `atol` - Absolute tolerance (default: 1e-8)
///
/// # Returns
/// True if gradients match within tolerance, false otherwise
///
/// # Tolerance Formula
///
/// For each element, we check:
/// ```text
/// |analytical - numerical| ≤ atol + rtol * |numerical|
/// ```
#[must_use]
pub fn gradients_close<T>(
    analytical: &Tensor<CpuBackend<Data = T>, DenseStorage<T>, T>,
    numerical: &Tensor<CpuBackend<Data = T>, DenseStorage<T>, T>,
    rtol: T,
    atol: T,
) -> bool
where
    T: DataType + FloatExt + PartialOrd,
{
    if analytical.shape().dims() != numerical.shape().dims() {
        return false;
    }

    let analytical_data = analytical.as_slice();
    let numerical_data = numerical.as_slice();

    analytical_data
        .iter()
        .zip(numerical_data.iter())
        .all(|(&a, &n)| {
            let diff = if a > n { a - n } else { n - a }; // abs(a - n)
            let threshold = atol + rtol * (if n > T::zero() { n } else { T::zero() - n }); // atol + rtol * |n|
            diff <= threshold
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dtype::float::Float32;

    /// Test numerical gradient for simple quadratic function f(x) = x²
    #[test]
    fn test_numerical_gradient_quadratic() {
        // f(x) = x², so df/dx = 2x
        let x_data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(3.0)],
            &[1],
        )
        .unwrap();
        let x = Variable::new(x_data);

        // Function that returns scalar: f(x) = x²
        let f = |v: &Variable<Float32>| {
            let x_val = v.data().as_slice()[0];
            let result = x_val * x_val;
            let result_tensor = Tensor::from_vec(vec![result], &[1]).unwrap();
            Variable::new(result_tensor)
        };

        let epsilon = Float32::new(1e-5);
        let numerical_grad = numerical_gradient(f, &x, epsilon).unwrap();

        // Analytical gradient: df/dx = 2x = 2 * 3 = 6
        let expected = Float32::new(6.0);
        let actual = numerical_grad.as_slice()[0];

        // Check within tolerance (1e-2 for f32 with epsilon=1e-5)
        // Finite differences have O(epsilon²) truncation error + floating point errors
        let diff = if actual.get() > expected.get() {
            actual.get() - expected.get()
        } else {
            expected.get() - actual.get()
        };
        assert!(
            diff < 1e-2,
            "Expected {}, got {}, diff {}",
            expected.get(),
            actual.get(),
            diff
        );
    }

    /// Test `gradients_close` function with matching gradients
    #[test]
    fn test_gradients_close_match() {
        let analytical = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let numerical = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![
                Float32::new(1.00001),
                Float32::new(2.00001),
                Float32::new(3.00001),
            ],
            &[3],
        )
        .unwrap();

        let rtol = Float32::new(1e-4);
        let atol = Float32::new(1e-6);

        assert!(gradients_close(&analytical, &numerical, rtol, atol));
    }

    /// Test `gradients_close` function with mismatched gradients
    #[test]
    fn test_gradients_close_mismatch() {
        let analytical = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let numerical = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.1), Float32::new(2.1), Float32::new(3.1)],
            &[3],
        )
        .unwrap();

        let rtol = Float32::new(1e-4);
        let atol = Float32::new(1e-6);

        assert!(!gradients_close(&analytical, &numerical, rtol, atol));
    }

    /// Test `gradients_close` with different shapes
    #[test]
    fn test_gradients_close_different_shapes() {
        let analytical = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0)],
            &[2],
        )
        .unwrap();

        let numerical = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();

        let rtol = Float32::new(1e-4);
        let atol = Float32::new(1e-6);

        assert!(!gradients_close(&analytical, &numerical, rtol, atol));
    }

    /// Test numerical gradient validation for simple sum operation
    #[test]
    fn test_numerical_gradient_sum() {
        use crate::graph::backward;

        let x_data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(2.0), Float32::new(3.0)],
            &[2],
        )
        .unwrap();
        let x = Variable::new(x_data);

        // Forward pass: loss = sum(x)
        let loss = x.sum();

        // Backward pass
        backward(&[&loss], &[]).unwrap();

        // Analytical gradient
        let analytical_grad = x.grad().unwrap();

        // Numerical gradient
        let f = |v: &Variable<Float32>| v.sum();
        let epsilon = Float32::new(1e-5);
        let numerical_grad = numerical_gradient(f, &x, epsilon).unwrap();

        // Debug output
        println!("Sum gradient check:");
        println!(
            "  Analytical: {:?}",
            analytical_grad
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );
        println!(
            "  Numerical:  {:?}",
            numerical_grad
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );

        // Check gradients match
        let rtol = Float32::new(1e-2);
        let atol = Float32::new(1e-4);
        assert!(
            gradients_close(&analytical_grad, &numerical_grad, rtol, atol),
            "Analytical and numerical gradients don't match for sum"
        );
    }

    /// Test numerical gradient validation for Pow operation
    /// Validates fix for approximation bug (x^(y-1) ≈ x*x)
    #[test]
    fn test_numerical_gradient_pow() {
        use crate::graph::backward;

        // Test case: x^3 where x=2.0
        // Analytical: d/dx(x^3) = 3*x^2 = 3*4 = 12.0
        let x_data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(2.0)],
            &[1],
        )
        .unwrap();
        let x = Variable::new(x_data);

        let exp_data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(3.0)],
            &[1],
        )
        .unwrap();
        let exp_var = Variable::new(exp_data);

        // Forward pass: loss = x^3
        let loss = x.pow(&exp_var);

        // Backward pass
        backward(&[&loss], &[]).unwrap();

        // Analytical gradient
        let analytical_grad = x.grad().unwrap();

        // Numerical gradient
        let f = |v: &Variable<Float32>| {
            let exp_data_inner = Tensor::from_vec(vec![Float32::new(3.0)], &[1]).unwrap();
            let exp_var_inner = Variable::new(exp_data_inner);
            v.pow(&exp_var_inner)
        };
        let epsilon = Float32::new(1e-5);
        let numerical_grad = numerical_gradient(f, &x, epsilon).unwrap();

        println!("Pow gradient check (x^3 at x=2):");
        println!(
            "  Analytical: {:?}",
            analytical_grad
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );
        println!(
            "  Numerical:  {:?}",
            numerical_grad
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );

        // Check gradients match (expected: 12.0)
        let rtol = Float32::new(1e-2);
        let atol = Float32::new(1e-4);
        assert!(
            gradients_close(&analytical_grad, &numerical_grad, rtol, atol),
            "Analytical and numerical gradients don't match for pow"
        );
    }

    /// Test numerical gradient validation for Exp operation
    #[test]
    fn test_numerical_gradient_exp() {
        use crate::graph::backward;

        let x_data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0)],
            &[1],
        )
        .unwrap();
        let x = Variable::new(x_data);

        // Forward pass: loss = exp(x)
        let loss = x.exp();

        // Backward pass
        backward(&[&loss], &[]).unwrap();

        // Analytical gradient
        let analytical_grad = x.grad().unwrap();

        // Numerical gradient
        let f = |v: &Variable<Float32>| v.exp();
        let epsilon = Float32::new(1e-5);
        let numerical_grad = numerical_gradient(f, &x, epsilon).unwrap();

        println!("Exp gradient check:");
        println!(
            "  Analytical: {:?}",
            analytical_grad
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );
        println!(
            "  Numerical:  {:?}",
            numerical_grad
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );

        let rtol = Float32::new(1e-2);
        let atol = Float32::new(1e-4);
        assert!(
            gradients_close(&analytical_grad, &numerical_grad, rtol, atol),
            "Analytical and numerical gradients don't match for exp"
        );
    }

    /// Test numerical gradient validation for Log operation
    #[test]
    fn test_numerical_gradient_log() {
        use crate::graph::backward;

        let x_data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(2.0)],
            &[1],
        )
        .unwrap();
        let x = Variable::new(x_data);

        // Forward pass: loss = log(x)
        let loss = x.log();

        // Backward pass
        backward(&[&loss], &[]).unwrap();

        // Analytical gradient
        let analytical_grad = x.grad().unwrap();

        // Numerical gradient
        let f = |v: &Variable<Float32>| v.log();
        let epsilon = Float32::new(1e-5);
        let numerical_grad = numerical_gradient(f, &x, epsilon).unwrap();

        println!("Log gradient check:");
        println!(
            "  Analytical: {:?}",
            analytical_grad
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );
        println!(
            "  Numerical:  {:?}",
            numerical_grad
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );

        let rtol = Float32::new(1e-2);
        let atol = Float32::new(1e-4);
        assert!(
            gradients_close(&analytical_grad, &numerical_grad, rtol, atol),
            "Analytical and numerical gradients don't match for log"
        );
    }

    /// Test numerical gradient validation for Sin operation
    #[test]
    fn test_numerical_gradient_sin() {
        use crate::graph::backward;

        let x_data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.5)],
            &[1],
        )
        .unwrap();
        let x = Variable::new(x_data);

        // Forward pass: loss = sin(x)
        let loss = x.sin();

        // Backward pass
        backward(&[&loss], &[]).unwrap();

        // Analytical gradient
        let analytical_grad = x.grad().unwrap();

        // Numerical gradient
        let f = |v: &Variable<Float32>| v.sin();
        let epsilon = Float32::new(1e-5);
        let numerical_grad = numerical_gradient(f, &x, epsilon).unwrap();

        println!("Sin gradient check:");
        println!(
            "  Analytical: {:?}",
            analytical_grad
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );
        println!(
            "  Numerical:  {:?}",
            numerical_grad
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );

        let rtol = Float32::new(1e-2);
        let atol = Float32::new(1e-4);
        assert!(
            gradients_close(&analytical_grad, &numerical_grad, rtol, atol),
            "Analytical and numerical gradients don't match for sin"
        );
    }

    /// Test numerical gradient validation for Cos operation
    #[test]
    fn test_numerical_gradient_cos() {
        use crate::graph::backward;

        let x_data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.5)],
            &[1],
        )
        .unwrap();
        let x = Variable::new(x_data);

        // Forward pass: loss = cos(x)
        let loss = x.cos();

        // Backward pass
        backward(&[&loss], &[]).unwrap();

        // Analytical gradient
        let analytical_grad = x.grad().unwrap();

        // Numerical gradient
        let f = |v: &Variable<Float32>| v.cos();
        let epsilon = Float32::new(1e-5);
        let numerical_grad = numerical_gradient(f, &x, epsilon).unwrap();

        println!("Cos gradient check:");
        println!(
            "  Analytical: {:?}",
            analytical_grad
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );
        println!(
            "  Numerical:  {:?}",
            numerical_grad
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );

        let rtol = Float32::new(1e-2);
        let atol = Float32::new(1e-4);
        assert!(
            gradients_close(&analytical_grad, &numerical_grad, rtol, atol),
            "Analytical and numerical gradients don't match for cos"
        );
    }

    /// Test numerical gradient validation for Mean operation
    #[test]
    fn test_numerical_gradient_mean() {
        use crate::graph::backward;

        let x_data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(2.0), Float32::new(4.0)],
            &[2],
        )
        .unwrap();
        let x = Variable::new(x_data);

        // Forward pass: loss = mean(x)
        let loss = x.mean();

        // Backward pass
        backward(&[&loss], &[]).unwrap();

        // Analytical gradient
        let analytical_grad = x.grad().unwrap();

        // Numerical gradient
        let f = |v: &Variable<Float32>| v.mean();
        let epsilon = Float32::new(1e-5);
        let numerical_grad = numerical_gradient(f, &x, epsilon).unwrap();

        println!("Mean gradient check:");
        println!(
            "  Analytical: {:?}",
            analytical_grad
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );
        println!(
            "  Numerical:  {:?}",
            numerical_grad
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );

        let rtol = Float32::new(1e-2);
        let atol = Float32::new(1e-4);
        assert!(
            gradients_close(&analytical_grad, &numerical_grad, rtol, atol),
            "Analytical and numerical gradients don't match for mean"
        );
    }

    /// Test numerical gradient validation for scalar broadcasting.
    ///
    /// Validates Sprint 3.7 Phase 2: Tensor-level broadcasting + autograd integration.
    /// Tests that gradients are correctly reduced back to original input shapes after broadcasting.
    #[test]
    fn test_numerical_gradient_broadcast_scalar_add() {
        use crate::graph::backward;

        // Test case: scalar + vector broadcasting
        // x (scalar [1]) + y (vector [3]) → z (vector [3])
        // Analytical: d/dx(x+y) = sum([1,1,1]) = 3 (reduce along broadcast dim)
        //             d/dy(x+y) = [1,1,1] (no broadcasting)
        let x_data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(2.0)],
            &[1],
        )
        .unwrap();
        let x = Variable::new(x_data);

        let y_data = Tensor::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();
        let y = Variable::new(y_data);

        // Forward pass: z = x + y (broadcasts to [3])
        let z = &x + &y;

        // Backward pass
        backward(&[&z], &[]).unwrap();

        // Analytical gradient for x (should be sum of grad_output = 3.0)
        let analytical_grad_x = x.grad().unwrap();

        // Numerical gradient for x
        let f_x = |v: &Variable<Float32>| {
            let y_inner = Variable::new(
                Tensor::from_vec(
                    vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
                    &[3],
                )
                .unwrap(),
            );
            v + &y_inner
        };
        let epsilon = Float32::new(1e-4); // Larger epsilon for better numerical stability with broadcasting
        let numerical_grad_x = numerical_gradient(f_x, &x, epsilon).unwrap();

        println!("Scalar broadcasting gradient check (x + y where x:[1], y:[3]):");
        println!(
            "  Analytical (x): {:?}",
            analytical_grad_x
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );
        println!(
            "  Numerical (x):  {:?}",
            numerical_grad_x
                .as_slice()
                .iter()
                .map(|x| x.get())
                .collect::<Vec<f32>>()
        );

        // Check gradients match (expected: 3.0 for scalar)
        // Relaxed tolerance for broadcasting tests due to summing multiple outputs
        let rtol = Float32::new(5e-2); // 5% relative tolerance
        let atol = Float32::new(1e-3); // 0.001 absolute tolerance
        assert!(
            gradients_close(&analytical_grad_x, &numerical_grad_x, rtol, atol),
            "Analytical and numerical gradients don't match for scalar broadcasting"
        );
    }
}
*/
