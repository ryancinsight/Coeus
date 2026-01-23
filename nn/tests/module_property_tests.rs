//! Property-based tests for neural network modules.
//!
//! This test suite uses property-based testing to verify universal properties
//! that should hold for all modules across all valid inputs.
//!
//! **Feature: coeus-architecture-enhancement**

use backend::CpuBackend;
use dtype::float::Float32;
use nn::core::module::Module;
use nn::modules::activation::{GeLU, LeakyReLU, ReLU, SiLU, ELU};
use nn::modules::linear::Linear;
use proptest::prelude::*;
use storage::DenseStorage;
use tensor::Tensor;

type TestBackend = CpuBackend<Float32>;
type TestStorage = DenseStorage<Float32>;
type TestDataType = Float32;

// ============================================================================
// Property 2: Layer Delegation to Operations
// ============================================================================

/// **Feature: coeus-architecture-enhancement, Property 2: Layer Delegation to Operations**
///
/// For any neural network layer in `nn/src/modules/`, the `forward()` method SHALL call
/// the corresponding function in `nn/src/functional/ops/` and SHALL NOT reimplement
/// the operation logic.
///
/// **Validates: Requirements 1.3, 3.2**
///
/// **Note:** This property is currently violated by most activation modules.
/// This test documents the expected behavior once refactoring is complete.
#[test]
#[ignore] // Ignored until delegation is implemented
fn property_layer_delegation_to_operations() {
    // This test would verify that modules delegate to functional/ops
    // by checking that the module's forward() implementation calls
    // the corresponding functional operation.
    //
    // Implementation approach:
    // 1. Mock or instrument functional/ops functions
    // 2. Call module forward()
    // 3. Verify functional operation was called
    // 4. Verify no duplicate logic in module
    //
    // Currently not implementable without code instrumentation or mocking.
    // This serves as documentation of the expected property.
}

// ============================================================================
// Property 5: Module Trait Implementation
// ============================================================================

/// **Feature: coeus-architecture-enhancement, Property 5: Module Trait Implementation**
///
/// For any layer in `nn/src/modules/`, the layer SHALL implement the `Module<B, S, T>` trait
/// with all required methods (forward, parameters, modules, zero_grad, train, name, clone_box).
///
/// **Validates: Requirements 3.3**
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn property_relu_implements_module_trait(
        input_size in 1usize..100,
    ) {
        let relu = ReLU::<TestBackend, TestStorage, TestDataType>::new();

        // Verify Module trait methods are implemented
        assert_eq!(relu.name(), "ReLU");
        assert_eq!(relu.parameters().len(), 0);

        // Verify forward() is callable
        let input = Tensor::ones(&[input_size]).unwrap();
        let result = relu.forward(&input);
        prop_assert!(result.is_ok());
    }

    #[test]
    fn property_gelu_implements_module_trait(
        input_size in 1usize..100,
    ) {
        let gelu = GeLU::<TestBackend, TestStorage, TestDataType>::new();

        assert_eq!(gelu.name(), "GeLU");
        assert_eq!(gelu.parameters().len(), 0);

        let input = Tensor::ones(&[input_size]).unwrap();
        let result = gelu.forward(&input);
        prop_assert!(result.is_ok());
    }

    #[test]
    fn property_silu_implements_module_trait(
        input_size in 1usize..100,
    ) {
        let silu = SiLU::<TestBackend, TestStorage, TestDataType>::new();

        assert_eq!(silu.name(), "SiLU");
        assert_eq!(silu.parameters().len(), 0);

        let input = Tensor::ones(&[input_size]).unwrap();
        let result = silu.forward(&input);
        prop_assert!(result.is_ok());
    }

    #[test]
    fn property_leaky_relu_implements_module_trait(
        input_size in 1usize..100,
    ) {
        let leaky_relu = LeakyReLU::<TestBackend, TestStorage, TestDataType>::new(
            TestDataType::new(0.01),
        );

        assert_eq!(leaky_relu.name(), "LeakyReLU");
        assert_eq!(leaky_relu.parameters().len(), 0);

        let input = Tensor::ones(&[input_size]).unwrap();
        let result = leaky_relu.forward(&input);
        prop_assert!(result.is_ok());
    }

    #[test]
    fn property_elu_implements_module_trait(
        input_size in 1usize..100,
    ) {
        let elu = ELU::<TestBackend, TestStorage, TestDataType>::new(TestDataType::new(1.0));

        assert_eq!(elu.name(), "ELU");
        assert_eq!(elu.parameters().len(), 0);

        let input = Tensor::ones(&[input_size]).unwrap();
        let result = elu.forward(&input);
        prop_assert!(result.is_ok());
    }

    #[test]
    fn property_linear_implements_module_trait(
        in_features in 1usize..50,
        out_features in 1usize..50,
        batch_size in 1usize..10,
    ) {
        let linear = Linear::<TestBackend, TestStorage, TestDataType>::new(
            in_features,
            out_features,
        ).unwrap();

        assert_eq!(linear.name(), "Linear");
        assert_eq!(linear.parameters().len(), 2); // weight and bias

        let input = Tensor::ones(&[batch_size, in_features]).unwrap();
        let result = linear.forward(&input);
        prop_assert!(result.is_ok());
    }
}

// ============================================================================
// Property 6: Parameter Management Abstraction
// ============================================================================

/// **Feature: coeus-architecture-enhancement, Property 6: Parameter Management Abstraction**
///
/// For any layer with learnable parameters, the parameters SHALL be stored as
/// `Parameter<B, S, T>` instances and SHALL be accessible through the `parameters()` method.
///
/// **Validates: Requirements 3.4**
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn property_linear_parameter_management(
        in_features in 1usize..50,
        out_features in 1usize..50,
    ) {
        let linear = Linear::<TestBackend, TestStorage, TestDataType>::new(
            in_features,
            out_features,
        ).unwrap();

        // Verify parameters are accessible
        let params = linear.parameters();
        prop_assert_eq!(params.len(), 2);

        // Verify parameter names
        prop_assert_eq!(params[0].name(), "weight");
        prop_assert_eq!(params[1].name(), "bias");

        // Verify parameter shapes
        let weight_shape = params[0].data().shape().dims();
        prop_assert_eq!(weight_shape, &[out_features, in_features]);

        let bias_shape = params[1].data().shape().dims();
        prop_assert_eq!(bias_shape, &[out_features]);
    }

    #[test]
    fn property_activation_no_parameters(
        input_size in 1usize..100,
    ) {
        // Activation functions should have no learnable parameters
        let relu = ReLU::<TestBackend, TestStorage, TestDataType>::new();
        prop_assert_eq!(relu.parameters().len(), 0);

        let gelu = GeLU::<TestBackend, TestStorage, TestDataType>::new();
        prop_assert_eq!(gelu.parameters().len(), 0);

        let silu = SiLU::<TestBackend, TestStorage, TestDataType>::new();
        prop_assert_eq!(silu.parameters().len(), 0);
    }
}

// ============================================================================
// Mathematical Properties of Activations
// ============================================================================

/// **Feature: coeus-architecture-enhancement, Property 3: Mathematical Properties**
///
/// Activation functions should satisfy their mathematical properties.
///
/// **Validates: Requirements 1.2, 1.4**
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn property_relu_output_non_negative(
        values in prop::collection::vec(-10.0f32..10.0f32, 1..100),
    ) {
        let relu = ReLU::<TestBackend, TestStorage, TestDataType>::new();

        let input_data: Vec<TestDataType> = values.iter().map(|&v| TestDataType::new(v)).collect();
        let input = Tensor::from_vec(input_data, &[values.len()]).unwrap();

        let output = relu.forward(&input).unwrap();
        let output_data = output.as_slice();

        // ReLU output should always be >= 0
        for &val in output_data {
            prop_assert!(val.get() >= 0.0);
        }
    }

    #[test]
    fn property_relu_preserves_positive_values(
        values in prop::collection::vec(0.0f32..10.0f32, 1..100),
    ) {
        let relu = ReLU::<TestBackend, TestStorage, TestDataType>::new();

        let input_data: Vec<TestDataType> = values.iter().map(|&v| TestDataType::new(v)).collect();
        let input = Tensor::from_vec(input_data.clone(), &[values.len()]).unwrap();

        let output = relu.forward(&input).unwrap();
        let output_data = output.as_slice();

        // ReLU should preserve positive values
        for (i, &val) in output_data.iter().enumerate() {
            prop_assert!((val.get() - input_data[i].get()).abs() < 1e-6);
        }
    }

    #[test]
    fn property_relu_zeros_negative_values(
        values in prop::collection::vec(-10.0f32..0.0f32, 1..100),
    ) {
        let relu = ReLU::<TestBackend, TestStorage, TestDataType>::new();

        let input_data: Vec<TestDataType> = values.iter().map(|&v| TestDataType::new(v)).collect();
        let input = Tensor::from_vec(input_data, &[values.len()]).unwrap();

        let output = relu.forward(&input).unwrap();
        let output_data = output.as_slice();

        // ReLU should zero out negative values
        for &val in output_data {
            prop_assert_eq!(val.get(), 0.0);
        }
    }

    #[test]
    fn property_leaky_relu_preserves_sign(
        values in prop::collection::vec(-10.0f32..10.0f32, 1..100),
        negative_slope in 0.01f32..0.5f32,
    ) {
        let leaky_relu = LeakyReLU::<TestBackend, TestStorage, TestDataType>::new(
            TestDataType::new(negative_slope),
        );

        let input_data: Vec<TestDataType> = values.iter().map(|&v| TestDataType::new(v)).collect();
        let input = Tensor::from_vec(input_data.clone(), &[values.len()]).unwrap();

        let output = leaky_relu.forward(&input).unwrap();
        let output_data = output.as_slice();

        // LeakyReLU should preserve the sign of the input
        for (i, &val) in output_data.iter().enumerate() {
            let input_val = input_data[i].get();
            let output_val = val.get();

            if input_val >= 0.0 {
                prop_assert!(output_val >= 0.0);
            } else {
                prop_assert!(output_val <= 0.0);
            }
        }
    }
}

// ============================================================================
// Linear Layer Properties
// ============================================================================

/// **Feature: coeus-architecture-enhancement, Property: Linear Layer Shape Preservation**
///
/// Linear layers should preserve batch dimension and transform feature dimension.
///
/// **Validates: Requirements 3.2**
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn property_linear_output_shape(
        in_features in 1usize..50,
        out_features in 1usize..50,
        batch_size in 1usize..10,
    ) {
        let linear = Linear::<TestBackend, TestStorage, TestDataType>::new(
            in_features,
            out_features,
        ).unwrap();

        let input = Tensor::ones(&[batch_size, in_features]).unwrap();
        let output = linear.forward(&input).unwrap();

        // Output should have shape [batch_size, out_features]
        prop_assert_eq!(output.shape().dims(), &[batch_size, out_features]);
    }

    #[test]
    fn property_linear_rejects_wrong_input_shape(
        in_features in 1usize..50,
        out_features in 1usize..50,
        wrong_features in 1usize..50,
    ) {
        prop_assume!(wrong_features != in_features);

        let linear = Linear::<TestBackend, TestStorage, TestDataType>::new(
            in_features,
            out_features,
        ).unwrap();

        let wrong_input = Tensor::ones(&[2, wrong_features]).unwrap();
        let result = linear.forward(&wrong_input);

        // Should fail with wrong input shape
        prop_assert!(result.is_err());
    }
}

// ============================================================================
// Module Cloning Properties
// ============================================================================

/// **Feature: coeus-architecture-enhancement, Property: Module Cloning**
///
/// Modules should be clonable and maintain their configuration.
///
/// **Validates: Requirements 3.3**
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn property_linear_clone_preserves_configuration(
        in_features in 1usize..50,
        out_features in 1usize..50,
    ) {
        let linear = Linear::<TestBackend, TestStorage, TestDataType>::new(
            in_features,
            out_features,
        ).unwrap();

        let cloned = linear.clone();

        // Cloned module should have same configuration
        prop_assert_eq!(cloned.in_features, linear.in_features);
        prop_assert_eq!(cloned.out_features, linear.out_features);
        prop_assert_eq!(cloned.parameters().len(), linear.parameters().len());
    }

    #[test]
    fn property_activation_clone_box(
        input_size in 1usize..100,
    ) {
        let relu = ReLU::<TestBackend, TestStorage, TestDataType>::new();
        let cloned = relu.clone_box();

        prop_assert_eq!(cloned.name(), "ReLU");
        prop_assert_eq!(cloned.parameters().len(), 0);
    }
}

// ============================================================================
// Zero Grad Properties
// ============================================================================

/// **Feature: coeus-architecture-enhancement, Property: Zero Grad**
///
/// Calling zero_grad() should detach gradients from parameters.
///
/// **Validates: Requirements 3.4**
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn property_linear_zero_grad(
        in_features in 1usize..50,
        out_features in 1usize..50,
    ) {
        let mut linear = Linear::<TestBackend, TestStorage, TestDataType>::new(
            in_features,
            out_features,
        ).unwrap();

        // Initially parameters should require gradients
        let params_before = linear.parameters();
        prop_assert!(params_before[0].requires_grad());
        prop_assert!(params_before[1].requires_grad());

        // After zero_grad, parameters should not require gradients
        linear.zero_grad();
        let params_after = linear.parameters();
        prop_assert!(!params_after[0].requires_grad());
        prop_assert!(!params_after[1].requires_grad());
    }
}
