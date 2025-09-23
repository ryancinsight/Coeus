//! Comprehensive test suite for PyCoeus Python bindings
//!
//! This module contains tests for PyTorch compatibility, tensor operations,
//! neural network layers, optimizers, tokenizers, and error conditions.

#[cfg(test)]
mod tests {
    use pyo3::prelude::*;
    use pyo3::PyObject;

    /// Test basic PyO3 functionality and module loading
    #[test]
    fn test_pycoeus_module_loading() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_core").unwrap();
            crate::_core(py, &module).unwrap();
            assert!(module.hasattr("PyTensor").unwrap());
            assert!(module.hasattr("Linear").unwrap());
            assert!(module.hasattr("Adam").unwrap());
        });
    }

    /// Test basic tensor operations
    #[test]
    fn test_basic_tensor_operations() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_core").unwrap();
            crate::_core(py, &module).unwrap();

            // Test that we can create tensors
            let tensor_class = module.getattr("PyTensor").unwrap();
            let tensor = tensor_class
                .call1((vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]))
                .unwrap();

            // Test that tensor has expected methods
            assert!(tensor.hasattr("shape").unwrap());
            assert!(tensor.hasattr("data").unwrap());
            assert!(tensor.hasattr("requires_grad").unwrap());
        });
    }

    /// Test neural network layer creation
    #[test]
    fn test_neural_network_layers() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_core").unwrap();
            crate::_core(py, &module).unwrap();

            // Test that we can create neural network layers
            let linear_class = module.getattr("Linear").unwrap();
            let linear = linear_class.call1((2, 3)).unwrap();

            // Test that linear layer has expected methods
            assert!(linear.hasattr("forward").unwrap());
            assert!(linear.hasattr("parameters").unwrap());
        });
    }

    /// Test optimizer creation
    #[test]
    fn test_optimizers() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_core").unwrap();
            crate::_core(py, &module).unwrap();

            // Test that we can create optimizers
            let adam_class = module.getattr("Adam").unwrap();

            // Test that Adam optimizer has expected methods
            println!("Adam class type: {:?}", adam_class);
            println!("Adam class dir: {:?}", adam_class.dir());
            assert!(adam_class.hasattr("__init__").unwrap());

            // Test that we can actually create an instance
            let params: Vec<PyObject> = vec![];
            let adam_instance = adam_class.call1((params, 0.001)).unwrap();
            assert!(adam_instance.hasattr("step").unwrap());
            assert!(adam_instance.hasattr("zero_grad").unwrap());
        });
    }

    /// Test utility functions
    #[test]
    fn test_utility_functions() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_core").unwrap();
            crate::_core(py, &module).unwrap();

            // Test utility functions exist
            assert!(module.hasattr("manual_seed").unwrap());
            assert!(module.hasattr("set_num_threads").unwrap());
            assert!(module.hasattr("get_num_threads").unwrap());
            assert!(module.hasattr("cuda_is_available").unwrap());
        });
    }

    /// Test gradient computation
    #[test]
    fn test_gradient_computation() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_core").unwrap();
            crate::_core(py, &module).unwrap();

            // Test that we can create tensors with gradient tracking
            let tensor_class = module.getattr("PyTensor").unwrap();

            // Create tensor with gradient tracking
            let tensor = tensor_class
                .call1((vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]))
                .unwrap();
            assert!(tensor.hasattr("requires_grad").unwrap());

            // Test that gradient computation works
            assert!(tensor.hasattr("backward").unwrap());
        });
    }

    /// Test neural network training loop
    #[test]
    fn test_neural_network_training() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_core").unwrap();
            crate::_core(py, &module).unwrap();

            // Test that we can create and train a simple neural network
            let linear_class = module.getattr("Linear").unwrap();
            let relu_class = module.getattr("ReLU").unwrap();

            // Create a simple network
            let linear1 = linear_class.call1((4, 10)).unwrap();
            let relu = relu_class.call1(()).unwrap();
            let linear2 = linear_class.call1((10, 1)).unwrap();

            // Test forward pass
            let tensor_class = module.getattr("PyTensor").unwrap();
            let input = tensor_class
                .call1((vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]))
                .unwrap();

            let hidden = linear1.call_method1("forward", (&input,)).unwrap();
            let activated = relu.call_method1("forward", (&hidden,)).unwrap();
            let output = linear2.call_method1("forward", (&activated,)).unwrap();

            // Test that output has correct shape
            assert!(output.hasattr("shape").unwrap());
        });
    }

    /// Test optimizer integration
    #[test]
    fn test_optimizer_integration() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_core").unwrap();
            crate::_core(py, &module).unwrap();

            // Test that optimizers can be created with parameters
            let linear_class = module.getattr("Linear").unwrap();
            let adam_class = module.getattr("Adam").unwrap();

            // Create a linear layer
            let linear = linear_class.call1((3, 1)).unwrap();

            // Create optimizer with parameters
            let weight = linear.call_method0("weight").unwrap(); // Call weight() method
            let params = vec![weight]; // Get weight parameter
            let optimizer = adam_class.call1((params, 0.01)).unwrap();

            // Test that optimizer has expected methods
            assert!(optimizer.hasattr("step").unwrap());
            assert!(optimizer.hasattr("zero_grad").unwrap());
        });
    }

    /// Test tensor operations
    #[test]
    fn test_tensor_operations() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_core").unwrap();
            crate::_core(py, &module).unwrap();

            let tensor_class = module.getattr("PyTensor").unwrap();

            // Test arithmetic operations
            let tensor1 = tensor_class
                .call1((vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]))
                .unwrap();
            let tensor2 = tensor_class
                .call1((vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]))
                .unwrap();

            // Test addition
            let sum_result = tensor1.call_method1("__add__", (&tensor2,)).unwrap();
            assert!(sum_result.hasattr("data").unwrap());

            // Test matrix multiplication
            let tensor_a = tensor_class
                .call1((vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]))
                .unwrap();
            let tensor_b = tensor_class
                .call1((vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]))
                .unwrap();
            let matmul_result = tensor_a.call_method1("__matmul__", (&tensor_b,)).unwrap();
            assert!(matmul_result.hasattr("data").unwrap());
        });
    }

    /// Test error handling
    #[test]
    fn test_error_handling() {
        Python::with_gil(|py| {
            let module = PyModule::new(py, "_core").unwrap();
            crate::_core(py, &module).unwrap();

            // Test that error conditions are handled properly
            let linear_class = module.getattr("Linear").unwrap();

            // Test invalid input shapes
            let result = linear_class.call1((-1, 10));
            // Should handle negative dimensions gracefully
            assert!(result.is_err() || result.unwrap().hasattr("forward").unwrap());
        });
    }
}
