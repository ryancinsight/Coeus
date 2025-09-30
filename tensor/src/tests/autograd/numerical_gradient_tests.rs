/// Numerical gradient checking tests
/// This module provides numerical gradient validation against finite differences
use approx::assert_relative_eq;

/// Numerical gradient checker using finite differences
pub struct NumericalGradientChecker {
    epsilon: f64,
}

impl NumericalGradientChecker {
    /// Create a new checker with default epsilon
    pub fn new() -> Self {
        Self { epsilon: 1e-7 }
    }

    /// Create a checker with custom epsilon
    pub fn with_epsilon(epsilon: f64) -> Self {
        Self { epsilon }
    }

    /// Check gradients of a function using central difference approximation
    /// f: function to differentiate
    /// x: input tensor (will be cloned and perturbed)
    /// expected_grad: analytical gradient from autograd
    /// tolerance: relative tolerance for comparison
    pub fn check_gradients<F>(
        &self,
        f: F,
        x: &Tensor<f64, CpuBackend, DenseStorage<f64>>,
        expected_grad: &Tensor<f64, CpuBackend, DenseStorage<f64>>,
        tolerance: f64,
    ) -> std::result::Result<(), Box<dyn std::error::Error>>
    where
        F: Fn(&Tensor<f64, CpuBackend, DenseStorage<f64>>) -> Tensor<f64, CpuBackend, DenseStorage<f64>>,
    {
        let backend = CpuBackend::default();

        // Compute numerical gradients using central difference
        let num_grad = self.compute_numerical_gradient(&f, x)?;

        // Compare shapes
        if expected_grad.shape() != num_grad.shape() {
            return Err(format!(
                "Shape mismatch: expected {:?}, got {:?}",
                expected_grad.shape(),
                num_grad.shape()
            ).into());
        }

        // Compare each element
        let expected_data = expected_grad.data();
        let num_data = num_grad.data();

        for (i, (&expected, &numerical)) in expected_data.iter().zip(num_data.iter()).enumerate() {
            let diff = (expected - numerical).abs();
            let rel_error = if expected.abs() > 0.0 {
                diff / expected.abs()
            } else {
                diff
            };

            assert!(
                rel_error <= tolerance,
                "Gradient mismatch at index {}: analytical={:.8e}, numerical={:.8e}, relative error={:.8e}",
                i,
                expected,
                numerical,
                rel_error
            );
        }

        Ok(())
    }

    /// Compute numerical gradient using central difference
    fn compute_numerical_gradient<F>(
        &self,
        f: &F,
        x: &Tensor<f64, CpuBackend, DenseStorage<f64>>,
    ) -> std::result::Result<Tensor<f64, CpuBackend, DenseStorage<f64>>, Box<dyn std::error::Error>>
    where
        F: Fn(&Tensor<f64, CpuBackend, DenseStorage<f64>>) -> Tensor<f64, CpuBackend, DenseStorage<f64>>,
    {
        let backend = CpuBackend::default();
        let x_data = x.data();
        let mut grad_data = vec![0.0; x_data.len()];

        // For each element in the input tensor
        for i in 0..x_data.len() {
            // Create perturbed tensors: x+h and x-h
            let mut x_plus_data = x_data.to_vec();
            let mut x_minus_data = x_data.to_vec();

            x_plus_data[i] += self.epsilon;
            x_minus_data[i] -= self.epsilon;

            let x_plus = Tensor::from_vec(backend.clone(), x_plus_data, x.shape().to_vec()).unwrap();
            let x_minus = Tensor::from_vec(backend.clone(), x_minus_data, x.shape().to_vec()).unwrap();

            // Compute function values
            let f_plus = f(&x_plus);
            let f_minus = f(&x_minus);

            // Central difference approximation
            // ∂f/∂x_i ≈ (f(x+h) - f(x-h)) / (2h)
            let f_plus_val = f_plus.as_scalar().unwrap();
            let f_minus_val = f_minus.as_scalar().unwrap();
            grad_data[i] = (f_plus_val - f_minus_val) / (2.0 * self.epsilon);
        }

        Ok(Tensor::from_vec(backend, grad_data, x.shape().to_vec()).unwrap())
    }
}

impl Default for NumericalGradientChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Test numerical gradient checking for simple functions
#[test]
fn test_numerical_gradient_checker() {
    let checker = NumericalGradientChecker::new();

    // Test: f(x) = x^2, f'(x) = 2x
    // At x = 3.0, f'(x) = 6.0
    let backend = CpuBackend::default();
    let mut x = Tensor::scalar(3.0);
    x.set_requires_grad(true);

    let mut y = x.pow(2.0).unwrap();
    y.backward().unwrap();

    let analytical_grad = x.grad().unwrap();

    // Define the function for numerical checking
    let f = |t: &Tensor<f64, CpuBackend, DenseStorage<f64>>| {
        t.pow(2.0).unwrap()
    };

    checker.check_gradients(f, &x, &analytical_grad, 1e-6).unwrap();
}

/// Test numerical gradient checking for multi-element tensors
#[test]
fn test_numerical_gradient_vector_input() {
    let checker = NumericalGradientChecker::new();

    // Test: f(x,y) = x + y, ∇f = [1, 1]
    // At x=2.0, y=3.0: ∇f = [1.0, 1.0]
    let backend = CpuBackend::default();
    let mut x = Tensor::from_vec(backend.clone(), vec![2.0, 3.0], vec![2]).unwrap();
    x.set_requires_grad(true);

    let mut y = x.clone(); // Identity function for simplicity
    y.backward().unwrap();

    let analytical_grad = x.grad().unwrap();

    // Define the function for numerical checking
    let f = |t: &Tensor<f64, CpuBackend, DenseStorage<f64>>| {
        t.clone() // Identity function
    };

    checker.check_gradients(f, &x, &analytical_grad, 1e-6).unwrap();
}

/// Test numerical gradient checking for activation functions
#[test]
fn test_activation_function_gradients() {
    let checker = NumericalGradientChecker::new();

    // Test exp function: f(x) = e^x, f'(x) = e^x
    let mut x: Tensor<f64, CpuBackend, DenseStorage<f64>> = Tensor::scalar(1.0);
    x.set_requires_grad(true);
    let mut y = x.exp().unwrap();
    y.backward().unwrap();
    let analytical_grad = x.grad().unwrap();

    let f_exp = |t: &Tensor<f64, CpuBackend, DenseStorage<f64>>| t.exp().unwrap();
    checker.check_gradients(f_exp, &x, &analytical_grad, 1e-6).unwrap();

    // Test sin function: f(x) = sin(x), f'(x) = cos(x)
    let mut x_sin: Tensor<f64, CpuBackend, DenseStorage<f64>> = Tensor::scalar(std::f64::consts::PI / 4.0);
    x_sin.set_requires_grad(true);
    let mut y_sin = x_sin.sin().unwrap();
    y_sin.backward().unwrap();
    let analytical_grad_sin = x_sin.grad().unwrap();

    let f_sin = |t: &Tensor<f64, CpuBackend, DenseStorage<f64>>| t.sin().unwrap();
    checker.check_gradients(f_sin, &x_sin, &analytical_grad_sin, 1e-6).unwrap();
}

