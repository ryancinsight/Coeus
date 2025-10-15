// Test that can be run to validate Python bindings work
// This would normally be tested with Python, but we can at least
// verify the Rust side compiles and basic functionality works

#[cfg(test)]
mod tests {
    use coeus_backend::CpuBackend;
    use coeus_dtype::float::Float32;
    use coeus_nn::{functional, Linear, Module};
    use coeus_optim::{Adagrad, Adam, Optimizer, RMSprop, SGD};
    use coeus_storage::DenseStorage;
    use coeus_tensor::Tensor;

    #[test]
    fn test_tensor_creation_for_python() {
        // Test that tensor creation works as expected for Python bindings
        let tensor = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::zeros(&[3, 3])
            .expect("Tensor creation should succeed");

        assert_eq!(tensor.shape().dims(), &[3, 3]);
        assert_eq!(tensor.len(), 9);
    }

    #[test]
    fn test_tensor_arithmetic_for_python() {
        // Test basic arithmetic that Python bindings will expose
        let a = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 2])
            .expect("Tensor creation should succeed");
        let b = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 2])
            .expect("Tensor creation should succeed");

        let c = &a + &b;
        // 1 + 1 = 2 for each element
        assert_eq!(c.as_slice().iter().map(|&x| x.0 as i32).sum::<i32>(), 8); // 2 * 4 elements
    }

    #[test]
    fn test_linear_layer_for_python() {
        let layer = Linear::<CpuBackend, DenseStorage<Float32>, Float32>::new(10, 5).unwrap();
        assert_eq!(layer.in_features, 10);
        assert_eq!(layer.out_features, 5);

        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[2, 10]).unwrap();
        let output = Module::forward(&layer, &input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 5]);
    }

    #[test]
    fn test_functional_operations_for_python() {
        let input = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(-1.0), Float32::new(0.0), Float32::new(1.0)],
            &[3],
        )
        .unwrap();

        let relu_out = functional::relu(&input).unwrap();
        assert_eq!(relu_out.shape().dims(), &[3]);

        let sigmoid_out = functional::sigmoid(&input).unwrap();
        assert_eq!(sigmoid_out.shape().dims(), &[3]);

        let tanh_out = functional::tanh(&input).unwrap();
        assert_eq!(tanh_out.shape().dims(), &[3]);
    }

    #[test]
    fn test_loss_functions_for_python() {
        let pred = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.0), Float32::new(2.0), Float32::new(3.0)],
            &[3],
        )
        .unwrap();
        let target = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(1.1), Float32::new(2.1), Float32::new(2.9)],
            &[3],
        )
        .unwrap();

        let mse_loss = functional::mse_loss(&pred, &target).unwrap();
        assert_eq!(mse_loss.shape().dims(), &[] as &[usize]); // Scalar output

        // For cross-entropy, we need logits and targets
        let logits = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(2.0), Float32::new(1.0), Float32::new(0.1)],
            &[1, 3],
        )
        .unwrap();
        let targets = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::from_vec(
            vec![Float32::new(0.0)],
            &[1],
        )
        .unwrap();

        let ce_loss = functional::cross_entropy(&logits, &targets).unwrap();
        assert_eq!(ce_loss.shape().dims(), &[] as &[usize]); // Scalar output
    }

    #[test]
    fn test_sgd_optimizer_for_python() {
        let mut optimizer = SGD::new(0.01, 0.0, 0.0, 0.0, false).unwrap();

        // Create a test parameter
        let param_data = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[5]).unwrap();
        let mut param = param_data;
        // Set a gradient for the parameter (normally this would come from backward pass)
        let grad_data = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[5]).unwrap();
        param.set_grad(grad_data).unwrap();

        optimizer.add_param(param).unwrap();

        // Test basic optimizer operations (gradients already set)
        let step_count = optimizer.step().unwrap();
        assert_eq!(step_count, 1); // 1 parameter
    }

    #[test]
    fn test_adam_optimizer_for_python() {
        let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8).unwrap();

        // Create a test parameter
        let param_data = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[5]).unwrap();
        let mut param = param_data;
        // Set a gradient for the parameter (normally this would come from backward pass)
        let grad_data = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[5]).unwrap();
        param.set_grad(grad_data).unwrap();

        optimizer.add_param(param).unwrap();

        // Test basic optimizer operations (gradients already set)
        let step_count = optimizer.step().unwrap();
        assert_eq!(step_count, 1); // 1 parameter
    }

    #[test]
    fn test_rmsprop_optimizer_for_python() {
        let mut optimizer = RMSprop::new(0.01, 0.99, 1e-8, 0.0, 0.0, false).unwrap();

        // Create a test parameter
        let param_data = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[5]).unwrap();
        let mut param = param_data;
        // Set a gradient for the parameter (normally this would come from backward pass)
        let grad_data = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[5]).unwrap();
        param.set_grad(grad_data).unwrap();

        optimizer.add_param(param).unwrap();

        // Test basic optimizer operations (gradients already set)
        let step_count = optimizer.step().unwrap();
        assert_eq!(step_count, 1); // 1 parameter
    }

    #[test]
    fn test_adagrad_optimizer_for_python() {
        let mut optimizer = Adagrad::new(0.01, 0.0, 0.0, 1e-10).unwrap();

        // Create a test parameter
        let param_data = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[5]).unwrap();
        let mut param = param_data;
        // Set a gradient for the parameter (normally this would come from backward pass)
        let grad_data = Tensor::<CpuBackend, DenseStorage<Float32>, Float32>::ones(&[5]).unwrap();
        param.set_grad(grad_data).unwrap();

        optimizer.add_param(param).unwrap();

        // Test basic optimizer operations (gradients already set)
        let step_count = optimizer.step().unwrap();
        assert_eq!(step_count, 1); // 1 parameter
    }
}
