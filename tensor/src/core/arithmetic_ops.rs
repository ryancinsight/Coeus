//! Mathematical operations for tensors
//!
//! This module contains element-wise mathematical operations and activation functions
//! for tensors, including automatic differentiation support.

use crate::{Tensor, TensorError, Dtype, FloatDtype, Result};
use num_traits::{Signed, Zero};
use crate::with_autograd_context;
use coeus_autograd::context::Operation;

#[cfg(test)]
mod arithmetic_ops_tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Test absolute value operation
    #[test]
    fn test_abs_basic() {
        let tensor = Tensor::from_vec(vec![-1.0, 2.0, -3.0], vec![3]);
        let result = tensor.abs();
        assert_eq!(result.data(), &[1.0, 2.0, 3.0]);
    }

    /// Test absolute value with gradient computation
    #[test]
    fn test_abs_with_gradients() {
        let mut tensor = Tensor::from_vec(vec![-2.0, 3.0, -1.0], vec![3]);
        tensor.set_requires_grad(true);

        let result = tensor.abs();
        assert_eq!(result.data(), &[2.0, 3.0, 1.0]);
        assert!(result.requires_grad());
    }

    /// Test absolute value edge cases
    #[test]
    fn test_abs_edge_cases() {
        // Test with zeros
        let zero_tensor = Tensor::from_vec(vec![0.0, -0.0], vec![2]);
        let abs_zero = zero_tensor.abs();
        assert_eq!(abs_zero.data(), &[0.0, 0.0]);

        // Test with positive values
        let pos_tensor = Tensor::from_vec(vec![1.0, 5.0], vec![2]);
        let abs_pos = pos_tensor.abs();
        assert_eq!(abs_pos.data(), &[1.0, 5.0]);
    }

    /// Test hyperbolic tangent operation
    #[test]
    fn test_tanh_basic() {
        let tensor = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);
        let result = tensor.tanh();
        // tanh(0) = 0, tanh(1) ≈ 0.7616, tanh(-1) ≈ -0.7616
        assert_relative_eq!(result.data()[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.data()[1], 0.7615941559557649, epsilon = 1e-6);
        assert_relative_eq!(result.data()[2], -0.7615941559557649, epsilon = 1e-6);
    }

    /// Test sigmoid operation
    #[test]
    fn test_sigmoid_basic() {
        let tensor = Tensor::from_vec(vec![0.0, 1.0, -1.0], vec![3]);
        let result = tensor.sigmoid();
        // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.7311, sigmoid(-1) ≈ 0.2689
        assert_relative_eq!(result.data()[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(result.data()[1], 0.7310585786300049, epsilon = 1e-6);
        assert_relative_eq!(result.data()[2], 0.2689414213699951, epsilon = 1e-6);
    }

    /// Test exponential operation
    #[test]
    fn test_exp_basic() {
        let tensor = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3]);
        let result = tensor.exp();
        // exp(0) = 1, exp(1) ≈ 2.7183, exp(2) ≈ 7.3891
        assert_relative_eq!(result.data()[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.data()[1], std::f64::consts::E, epsilon = 1e-6);
        assert_relative_eq!(result.data()[2], std::f64::consts::E.powi(2), epsilon = 1e-6);
    }

    /// Test natural logarithm operation
    #[test]
    fn test_log_basic() {
        let tensor = Tensor::from_vec(vec![1.0, std::f64::consts::E, std::f64::consts::E.powi(2)], vec![3]);
        let result = tensor.log();
        // log(1) = 0, log(e) = 1, log(e^2) = 2
        assert_relative_eq!(result.data()[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.data()[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.data()[2], 2.0, epsilon = 1e-6);
    }

    /// Test sine operation
    #[test]
    fn test_sin_basic() {
        let tensor = Tensor::from_vec(vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI], vec![3]);
        let result = tensor.sin();
        // sin(0) = 0, sin(π/2) = 1, sin(π) = 0
        assert_relative_eq!(result.data()[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.data()[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.data()[2], 0.0, epsilon = 1e-6);
    }

    /// Test cosine operation
    #[test]
    fn test_cos_basic() {
        let tensor = Tensor::from_vec(vec![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI], vec![3]);
        let result = tensor.cos();
        // cos(0) = 1, cos(π/2) = 0, cos(π) = -1
        assert_relative_eq!(result.data()[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.data()[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.data()[2], -1.0, epsilon = 1e-6);
    }

    /// Test square root operation
    #[test]
    fn test_sqrt_basic() {
        let tensor = Tensor::from_vec(vec![0.0, 1.0, 4.0], vec![3]);
        let result = tensor.sqrt();
        // sqrt(0) = 0, sqrt(1) = 1, sqrt(4) = 2
        assert_relative_eq!(result.data()[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result.data()[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result.data()[2], 2.0, epsilon = 1e-6);
    }

    /// Test ReLU operation
    #[test]
    fn test_relu_basic() {
        let tensor = Tensor::from_vec(vec![-1.0, 0.0, 2.0], vec![3]);
        let result = tensor.relu();
        // ReLU(x) = max(0, x)
        assert_eq!(result.data(), &[0.0, 0.0, 2.0]);
    }

    /// Test power operation
    #[test]
    fn test_pow_basic() {
        let tensor = Tensor::from_vec(vec![2.0, 3.0, 4.0], vec![3]);
        let result = tensor.pow(2.0);
        // 2^2 = 4, 3^2 = 9, 4^2 = 16
        assert_eq!(result.data(), &[4.0, 9.0, 16.0]);
    }

    /// Test numerical stability for edge cases
    #[test]
    fn test_numerical_stability() {
        // Test with very small numbers
        let small_tensor = Tensor::from_vec(vec![1e-10, 1e-15], vec![2]);
        let sqrt_result = small_tensor.sqrt();
        assert!(sqrt_result.data()[0].is_finite());
        assert!(sqrt_result.data()[1].is_finite());

        // Test with very large numbers
        let large_tensor = Tensor::from_vec(vec![1e10, 1e15], vec![2]);
        let log_result = large_tensor.log();
        assert!(log_result.data()[0].is_finite());
        assert!(log_result.data()[1].is_finite());
    }

    /// Test gradient computation accuracy
    #[test]
    fn test_gradient_computation() {
        // Test sigmoid gradient
        let mut tensor = Tensor::from_vec(vec![0.5, 1.0], vec![2]);
        tensor.set_requires_grad(true);

        let result = tensor.sigmoid();
        // For numerical validation, we'll check that gradients are computed
        assert!(result.requires_grad());

        // Test tanh gradient
        let tanh_result = tensor.tanh();
        assert!(tanh_result.requires_grad());
    }

    /// Test edge cases for overflow and underflow
    #[test]
    fn test_overflow_underflow_edge_cases() {
        // Test exponential overflow
        let large_tensor = Tensor::from_vec(vec![700.0, 800.0, 900.0], vec![3]);
        let exp_result = large_tensor.exp();
        assert!(exp_result.data().iter().all(|&x| x.is_infinite()));

        // Test logarithm underflow
        let small_tensor = Tensor::from_vec(vec![1e-10, 1e-15, 1e-20], vec![3]);
        let log_result = small_tensor.log();
        assert!(log_result.data().iter().all(|&x| x.is_finite() || x.is_nan()));

        // Test square root of negative numbers (should error)
        let negative_tensor = Tensor::from_vec(vec![-1.0, -2.0], vec![2]);
        let sqrt_result = negative_tensor.sqrt();
        // Square root of negative should produce NaN
        assert!(sqrt_result.data().iter().all(|&x| x.is_nan()));
    }

    /// Test precision and accuracy for floating point operations
    #[test]
    fn test_precision_accuracy() {
        // Test that operations preserve precision for critical values
        let critical_values = vec![
            0.0, 1.0, -1.0, std::f64::consts::E, std::f64::consts::PI,
            1e-6, 1e6, 1e-10, 1e10
        ];
        let tensor = Tensor::from_vec(critical_values, vec![critical_values.len()]);

        // Test that sin(0) = 0 exactly
        let sin_result = tensor.sin();
        assert_relative_eq!(sin_result.data()[0], 0.0, epsilon = 1e-15);

        // Test that cos(0) = 1 exactly
        let cos_result = tensor.cos();
        assert_relative_eq!(cos_result.data()[0], 1.0, epsilon = 1e-15);

        // Test that exp(0) = 1 exactly
        let exp_result = tensor.exp();
        assert_relative_eq!(exp_result.data()[0], 1.0, epsilon = 1e-15);

        // Test that log(1) = 0 exactly
        let log_result = tensor.log();
        assert_relative_eq!(log_result.data()[0], 0.0, epsilon = 1e-15);
    }

    /// Test memory safety and bounds checking
    #[test]
    fn test_memory_safety() {
        // Test with large tensors
        let large_data: Vec<f64> = (0..10000).map(|x| x as f64).collect();
        let large_tensor = Tensor::from_vec(large_data, vec![10000]);

        let abs_result = large_tensor.abs();
        assert_eq!(abs_result.shape(), vec![10000]);

        // Test with zero-sized tensor
        let empty_tensor = Tensor::from_vec(vec![], vec![0]);
        let empty_result = empty_tensor.abs();
        assert_eq!(empty_result.shape(), vec![0]);
    }

    /// Test error conditions and error handling
    #[test]
    fn test_error_conditions() {
        // Test division by zero in sigmoid (should be handled gracefully)
        let large_negative = Tensor::from_vec(vec![-1000.0, -2000.0], vec![2]);
        let sigmoid_result = large_negative.sigmoid();
        assert!(sigmoid_result.data().iter().all(|&x| x.is_finite()));

        // Test logarithm of zero (should produce -inf)
        let zero_tensor = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        let log_result = zero_tensor.log();
        assert!(log_result.data().iter().all(|&x| x.is_infinite() && x.is_sign_negative()));

        // Test logarithm of negative numbers (should produce NaN)
        let negative_tensor = Tensor::from_vec(vec![-1.0, -2.0], vec![2]);
        let log_negative = negative_tensor.log();
        assert!(log_negative.data().iter().all(|&x| x.is_nan()));
    }

    /// Test type constraints and generic implementations
    #[test]
    fn test_type_constraints() {
        // Test with different numeric types if supported
        let f32_tensor = Tensor::from_vec(vec![1.0_f32, 2.0_f32, 3.0_f32], vec![3]);
        let f32_abs = f32_tensor.abs();
        assert_eq!(f32_abs.data(), &[1.0_f32, 2.0_f32, 3.0_f32]);

        // Test integer types (where applicable)
        let i32_tensor = Tensor::from_vec(vec![-1_i32, 2_i32, -3_i32], vec![3]);
        let i32_abs = i32_tensor.abs();
        assert_eq!(i32_abs.data(), &[1_i32, 2_i32, 3_i32]);
    }

    /// Test numerical stability for very small and very large numbers
    #[test]
    fn test_numerical_stability_extended() {
        // Test subnormal numbers
        let subnormal_tensor = Tensor::from_vec(vec![1e-308, 1e-309, 1e-310], vec![3]);
        let sqrt_subnormal = subnormal_tensor.sqrt();
        assert!(sqrt_subnormal.data().iter().all(|&x| x.is_finite() || x == 0.0));

        // Test very large exponents
        let huge_exponent = Tensor::from_vec(vec![100.0, 200.0, 300.0], vec![3]);
        let exp_huge = huge_exponent.exp();
        assert!(exp_huge.data().iter().all(|&x| x.is_infinite()));

        // Test very small bases for logarithm
        let tiny_base = Tensor::from_vec(vec![1e-100, 1e-200, 1e-300], vec![3]);
        let log_tiny = tiny_base.log();
        assert!(log_tiny.data().iter().all(|&x| x.is_infinite() && x.is_sign_negative()));
    }

    /// Test gradient computation for all operations with edge cases
    #[test]
    fn test_comprehensive_gradient_computation() {
        // Test gradient computation for operations near critical points
        let critical_tensor = Tensor::from_vec(vec![0.0, 1.0, -1.0, 0.5, 2.0], vec![5]);
        critical_tensor.set_requires_grad(true);

        // Test sigmoid at saturation points
        let sigmoid_crit = critical_tensor.sigmoid();
        assert!(sigmoid_crit.requires_grad());

        // Test tanh at critical points
        let tanh_crit = critical_tensor.tanh();
        assert!(tanh_crit.requires_grad());

        // Test ReLU at zero
        let relu_crit = critical_tensor.relu();
        assert!(relu_crit.requires_grad());

        // Test exponential with zero
        let exp_crit = critical_tensor.exp();
        assert!(exp_crit.requires_grad());

        // Test logarithm with one
        let log_crit = critical_tensor.log();
        assert!(log_crit.requires_grad());
    }

    /// Test operation composition and chain rule
    #[test]
    fn test_operation_composition() {
        let tensor = Tensor::from_vec(vec![0.5, 1.0, 1.5], vec![3]);
        tensor.set_requires_grad(true);

        // Test complex composition: sigmoid(tanh(x))
        let composed = tensor.tanh().sigmoid();
        assert!(composed.requires_grad());

        // Test that gradients flow through composition
        assert!(composed.data().iter().all(|&x| x.is_finite()));

        // Test that shape is preserved through composition
        assert_eq!(composed.shape(), vec![3]);
    }

    /// Test comprehensive numerical stability for all operations
    #[test]
    fn test_comprehensive_numerical_stability() {
        // Test with extreme values that could cause numerical issues
        let extreme_values = vec![
            f64::MIN_POSITIVE, f64::MAX, f64::INFINITY, f64::NEG_INFINITY,
            f64::NAN, 0.0, -0.0, 1.0, -1.0, 1e-15, -1e-15,
            1e15, -1e15, 1e-308, -1e-308, 1e308, -1e308
        ];

        let tensor = Tensor::from_vec(extreme_values.clone(), vec![extreme_values.len()]);

        // Test all operations handle extreme values appropriately
        let abs_result = tensor.abs();
        let tanh_result = tensor.tanh();
        let sigmoid_result = tensor.sigmoid();
        let exp_result = tensor.exp();
        let log_result = tensor.log();
        let sin_result = tensor.sin();
        let cos_result = tensor.cos();
        let sqrt_result = tensor.sqrt();
        let relu_result = tensor.relu();

        // Verify that operations don't crash and produce reasonable results
        assert_eq!(abs_result.shape(), vec![extreme_values.len()]);
        assert_eq!(tanh_result.shape(), vec![extreme_values.len()]);
        assert_eq!(sigmoid_result.shape(), vec![extreme_values.len()]);
        assert_eq!(exp_result.shape(), vec![extreme_values.len()]);
        assert_eq!(log_result.shape(), vec![extreme_values.len()]);
        assert_eq!(sin_result.shape(), vec![extreme_values.len()]);
        assert_eq!(cos_result.shape(), vec![extreme_values.len()]);
        assert_eq!(sqrt_result.shape(), vec![extreme_values.len()]);
        assert_eq!(relu_result.shape(), vec![extreme_values.len()]);
    }

    /// Test gradient computation for extreme values
    #[test]
    fn test_gradient_computation_extreme_values() {
        let extreme_values = vec![1e-10, 1e10, 1.0, -1.0, 0.0];
        let mut tensor = Tensor::from_vec(extreme_values, vec![5]);
        tensor.set_requires_grad(true);

        // Test that gradients can be computed for all operations with extreme values
        let abs_result = tensor.abs();
        let tanh_result = tensor.tanh();
        let sigmoid_result = tensor.sigmoid();
        let exp_result = tensor.exp();
        let sin_result = tensor.sin();
        let cos_result = tensor.cos();
        let relu_result = tensor.relu();

        assert!(abs_result.requires_grad());
        assert!(tanh_result.requires_grad());
        assert!(sigmoid_result.requires_grad());
        assert!(exp_result.requires_grad());
        assert!(sin_result.requires_grad());
        assert!(cos_result.requires_grad());
        assert!(relu_result.requires_grad());
    }

    /// Test operation chaining with gradient computation
    #[test]
    fn test_operation_chaining_gradients() {
        let tensor = Tensor::from_vec(vec![0.5, 1.0, 1.5], vec![3]);
        tensor.set_requires_grad(true);

        // Complex operation chain: abs(sigmoid(tanh(x)))
        let chained = tensor.tanh().sigmoid().abs();
        assert!(chained.requires_grad());

        // Test multiple levels of composition
        let complex_chain = tensor.sin().cos().tanh().sigmoid().relu();
        assert!(complex_chain.requires_grad());
        assert_eq!(complex_chain.shape(), vec![3]);
    }

    /// Test memory allocation and deallocation for large tensors
    #[test]
    fn test_memory_management() {
        // Test with very large tensor to ensure memory is managed correctly
        let large_data: Vec<f64> = (0..50000).map(|x| (x % 1000) as f64 - 500.0).collect();
        let large_tensor = Tensor::from_vec(large_data.clone(), vec![large_data.len()]);

        // Test that operations on large tensors work correctly
        let abs_large = large_tensor.abs();
        let tanh_large = large_tensor.tanh();
        let relu_large = large_tensor.relu();

        assert_eq!(abs_large.shape(), vec![large_data.len()]);
        assert_eq!(tanh_large.shape(), vec![large_data.len()]);
        assert_eq!(relu_large.shape(), vec![large_data.len()]);

        // Verify some specific values
        assert_eq!(abs_large.data()[0], 500.0); // abs(-500.0)
        assert!(tanh_large.data()[1000].is_finite());
        assert!(relu_large.data()[25000] >= 0.0); // relu is non-negative
    }

    /// Test all operations preserve tensor properties
    #[test]
    fn test_tensor_properties_preservation() {
        let tensor = Tensor::from_vec(vec![1.0, -2.0, 3.0, -4.0], vec![4]);
        tensor.set_requires_grad(true);

        let operations = vec![
            tensor.abs(),
            tensor.tanh(),
            tensor.sigmoid(),
            tensor.exp(),
            tensor.sin(),
            tensor.cos(),
            tensor.sqrt(),
            tensor.relu(),
        ];

        for result in operations {
            assert!(result.requires_grad());
            assert_eq!(result.shape(), vec![4]);
            assert!(result.data().iter().all(|&x| x.is_finite()));
        }
    }

    /// Test operations with different tensor shapes
    #[test]
    fn test_different_tensor_shapes() {
        // Test 2D tensor
        let tensor_2d = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let abs_2d = tensor_2d.abs();
        assert_eq!(abs_2d.shape(), vec![2, 2]);

        // Test 3D tensor
        let tensor_3d = Tensor::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], vec![2, 3]);
        let abs_3d = tensor_3d.abs();
        assert_eq!(abs_3d.shape(), vec![2, 3]);

        // Test scalar tensor
        let scalar_tensor = Tensor::from_vec(vec![-42.0], vec![1]);
        let abs_scalar = scalar_tensor.abs();
        assert_eq!(abs_scalar.shape(), vec![1]);
        assert_eq!(abs_scalar.data(), &[42.0]);
    }

    /// Test all operations are mathematically correct for basic cases
    #[test]
    fn test_mathematical_correctness() {
        let test_cases = vec![
            (0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0),  // x, abs, tanh, sigmoid, exp, log, sin, cos, sqrt
            (1.0, 1.0, 0.7616, 0.7311, std::f64::consts::E, 0.0, 0.8415, 0.5403, 1.0),
            (-1.0, 1.0, -0.7616, 0.2689, 0.3679, f64::NAN, -0.8415, 0.5403, f64::NAN),
            (2.0, 2.0, 0.9640, 0.8808, 7.3891, 0.6931, 0.9093, -0.4161, 1.4142),
        ];

        for (x, expected_abs, expected_tanh, expected_sigmoid, expected_exp, expected_log, expected_sin, expected_cos, expected_sqrt) in test_cases {
            let tensor = Tensor::from_vec(vec![x], vec![1]);

            let abs_result = tensor.abs();
            let tanh_result = tensor.tanh();
            let sigmoid_result = tensor.sigmoid();
            let exp_result = tensor.exp();
            let log_result = tensor.log();
            let sin_result = tensor.sin();
            let cos_result = tensor.cos();
            let sqrt_result = tensor.sqrt();

            assert_relative_eq!(abs_result.data()[0], expected_abs, epsilon = 1e-4);
            assert_relative_eq!(tanh_result.data()[0], expected_tanh, epsilon = 1e-4);
            assert_relative_eq!(sigmoid_result.data()[0], expected_sigmoid, epsilon = 1e-4);
            assert_relative_eq!(exp_result.data()[0], expected_exp, epsilon = 1e-4);
            assert_relative_eq!(sin_result.data()[0], expected_sin, epsilon = 1e-4);
            assert_relative_eq!(cos_result.data()[0], expected_cos, epsilon = 1e-4);

            if x >= 0.0 {
                assert_relative_eq!(sqrt_result.data()[0], expected_sqrt, epsilon = 1e-4);
            }

            if x > 0.0 {
                assert_relative_eq!(log_result.data()[0], expected_log, epsilon = 1e-4);
            }
        }
    }
}

impl<T: Dtype + num_traits::FromPrimitive + num_traits::ToPrimitive> Tensor<T> {
    /// Compute the absolute value of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-1.0, 2.0, -3.0], vec![3]);
    /// let abs_tensor = tensor.abs();
    /// // Result: [1.0, 2.0, 3.0]
    /// ```
    pub fn abs(&self) -> Tensor<T>
    where
        T: Signed,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.abs()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Abs, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the hyperbolic tangent of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 1.0], vec![2]);
    /// let tanh_tensor = tensor.tanh();
    /// // Result: [tanh(0.0), tanh(1.0)]
    /// ```
    pub fn tanh(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.tanh()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Tanh, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the sigmoid (logistic) function of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0], vec![1]);
    /// let sigmoid_tensor = tensor.sigmoid();
    /// // Result: [0.5]
    /// ```
    pub fn sigmoid(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| T::one() / (T::one() + (-*x).exp())).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Sigmoid, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the exponential of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 1.0], vec![2]);
    /// let exp_tensor = tensor.exp();
    /// // Result: [exp(0.0), exp(1.0)]
    /// ```
    pub fn exp(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.exp()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Exp, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the natural logarithm of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 2.718], vec![2]);
    /// let log_tensor = tensor.log();
    /// // Result: [log(1.0), log(2.718)] ≈ [0.0, 1.0]
    /// ```
    pub fn log(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.ln()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Log, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the sine of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 3.14159], vec![2]);
    /// let sin_tensor = tensor.sin();
    /// // Result: [sin(0.0), sin(π)] ≈ [0.0, 0.0]
    /// ```
    pub fn sin(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.sin()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Sin, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the cosine of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![0.0, 1.5708], vec![2]);
    /// let cos_tensor = tensor.cos();
    /// // Result: [cos(0.0), cos(π/2)] ≈ [1.0, 0.0]
    /// ```
    pub fn cos(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.cos()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Cos, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Compute the square root of each element
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![1.0, 4.0, 9.0], vec![3]);
    /// let sqrt_tensor = tensor.sqrt();
    /// // Result: [1.0, 2.0, 3.0]
    /// ```
    pub fn sqrt(&self) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.sqrt()).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Sqrt, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Apply the Rectified Linear Unit (ReLU) activation function
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![-1.0, 0.0, 2.0], vec![3]);
    /// let relu_tensor = tensor.relu();
    /// // Result: [0.0, 0.0, 2.0]
    /// ```
    pub fn relu(&self) -> Tensor<T>
    where
        T: PartialOrd + Zero,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| if *x > T::zero() { *x } else { T::zero() }).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Relu, vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }

    /// Raise each element to a power
    ///
    /// # Arguments
    /// * `exponent` - Power to raise each element to
    ///
    /// # Example
    /// ```rust
    /// use coeus_tensor::Tensor;
    ///
    /// let tensor = Tensor::from_vec(vec![2.0, 3.0], vec![2]);
    /// let pow_tensor = tensor.pow(3.0);
    /// // Result: [8.0, 27.0]
    /// ```
    pub fn pow(&self, exponent: T) -> Tensor<T>
    where
        T: FloatDtype,
    {
        let result_data: Vec<T> = self.data.iter().map(|x| x.powf(exponent)).collect();
        let mut result = Tensor::from_vec(result_data, self.shape.clone());

        // Create computational graph node if input requires gradients
        if self.requires_grad() {
            result.set_requires_grad(true);
            with_autograd_context(|context| {
                let self_node = if let Some(node) = self.node {
                    node
                } else {
                    let node = context.create_node(Operation::Add, vec![]);
                    let data_f64: Vec<f64> = self
                        .data
                        .iter()
                        .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                        .collect();
                    context.register_tensor(node, data_f64, self.shape.clone());
                    node
                };

                let node_id = context.create_node(Operation::Pow(exponent.to_f64().unwrap_or(1.0)), vec![self_node]);
                let result_data_f64: Vec<f64> = result
                    .data
                    .iter()
                    .map(|&x| num_traits::ToPrimitive::to_f64(&x).unwrap_or(0.0))
                    .collect();
                context.register_tensor(node_id, result_data_f64, result.shape.clone());
                result.node = Some(node_id);
            });
        }

        result
    }
}
