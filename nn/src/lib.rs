//! # Coeus Neural Networks
//!
//! PyTorch-like neural network modules built on top of the Coeus tensor library.
//!
//! This crate provides:
//! - **Linear Layers**: Fully connected neural network layers
//! - **Convolutional Layers**: 1D, 2D, and 3D convolution operations
//! - **Recurrent Layers**: LSTM, GRU, and RNN cells
//! - **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, etc.
//! - **Normalization Layers**: BatchNorm, LayerNorm, GroupNorm
//! - **Dropout**: Regularization through random neuron deactivation
//! - **Loss Functions**: MSE, CrossEntropy, NLLLoss, etc.
//! - **Optimizers**: SGD, Adam, AdamW, RMSprop
//!
//! ## Basic Usage
//!
//! ```rust,no_run
//! use coeus_nn::{Linear, ReLU, Sequential, Module};
//! use coeus_tensor::Tensor;
//!
//! // Create a simple neural network
//! let model = Sequential::new(vec![
//!     Box::new(Linear::<f32>::new(784, 128)),
//!     Box::new(ReLU::new()),
//!     Box::new(Linear::<f32>::new(128, 10)),
//! ]);
//!
//! // Forward pass
//! let input = Tensor::from_vec(vec![0.0; 784], vec![1, 784]);
//! let output = model.forward(&input);
//! ```
//!
//! ## Automatic Differentiation
//!
//! All modules automatically integrate with Coeus' autograd system:
//!
//! ```rust,no_run
//! use coeus_nn::{Linear, MseLoss, Module};
//! use coeus_tensor::Tensor;
//!
//! let mut layer = Linear::<f32>::new(10, 1);
//! let loss_fn = MseLoss::new();
//!
//! // Forward pass with gradient tracking
//! let input = Tensor::from_vec_with_grad(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![10]);
//! let target = Tensor::scalar(5.0);
//!
//! let output = layer.forward(&input).unwrap();
//! let loss = loss_fn.forward(&output, &target).unwrap();
//!
//! // Backward pass
//! loss.backward().unwrap();
//!
//! // Access gradients
//! if let Some(grad) = layer.weight.grad() {
//!     println!("Weight gradient shape: {:?}", grad.shape());
//! }
//! ```
//!
//! ## Custom Modules
//!
//! Implement the `Module` trait to create custom neural network components:
//!
//! ```rust,no_run
//! use coeus_nn::{Module, NNError};
//! use coeus_tensor::{Tensor, Mul};
//!
//! struct CustomLayer {
//!     weight: Tensor<f32>,
//! }
//!
//! impl Module<f32> for CustomLayer {
//!     fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>, NNError> {
//!         // Custom forward implementation
//!         input.mul(&self.weight).map_err(|e| NNError::InvalidInput {
//!             message: format!("Multiplication failed: {}", e)
//!         })
//!     }
//!
//!     fn parameters(&self) -> Vec<&Tensor<f32>> {
//!         vec![&self.weight]
//!     }
//!
//!     fn parameters_mut(&mut self) -> Vec<&mut Tensor<f32>> {
//!         vec![&mut self.weight]
//!     }
//! }
//! ```
//!
//! ## References
//!
//! - [PyTorch Documentation](https://pytorch.org/docs/)
//! - [Deep Learning Book](https://www.deeplearningbook.org/)
//! - [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

pub mod activations;
pub mod containers;
pub mod init;
pub mod losses;
pub mod modules;
pub mod optimizers;
pub mod validation;

pub use activations::{
    ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU, GELU, Hardtanh, LogSigmoid, SELU, CELU,
    Hardshrink, Tanhshrink, Threshold, PReLU, RReLU, Softmin, Softmax2d,
};
pub use containers::*;
pub use init::*;
pub use losses::*;
pub use modules::*;
pub use optimizers::*;
pub use validation::*;

/// Result type for neural network operations
pub type Result<T> = std::result::Result<T, NNError>;

/// Errors that can occur during neural network operations
#[derive(Debug, thiserror::Error)]
pub enum NNError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("Initialization error: {message}")]
    InitializationError { message: String },

    #[error("Forward pass error: {message}")]
    ForwardError { message: String },

    #[error("Backward pass error: {message}")]
    BackwardError { message: String },

    #[error("Tensor error: {0}")]
    TensorError(#[from] coeus_tensor::TensorError),
}

/// Core trait for neural network modules
/// Uses FloatDtype to support gradient computation during training
pub trait Module<T: coeus_dtype::FloatDtype> {
    /// Forward pass through the module
    fn forward(&self, input: &coeus_tensor::Tensor<T>) -> crate::Result<coeus_tensor::Tensor<T>>;

    /// Get all parameters of the module
    fn parameters(&self) -> Vec<&coeus_tensor::Tensor<T>>;

    /// Get mutable references to all parameters
    fn parameters_mut(&mut self) -> Vec<&mut coeus_tensor::Tensor<T>>;

    /// Set the module to training mode
    fn train(&mut self) {
        // Default implementation: do nothing
    }

    /// Set the module to evaluation mode
    fn eval(&mut self) {
        // Default implementation: do nothing
    }

    /// Zero all gradients in the module
    fn zero_grad(&mut self) {
        // Zero gradients for all parameters using autograd integration
        for param in self.parameters_mut() {
            param.zero_grad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_tensor::Tensor;

    #[test]
    fn test_module_trait() {
        // Test that we can create and use modules
        let linear = modules::Linear::new(10, 5);
        let input = Tensor::from_vec(vec![1.0; 10], vec![10]);

        let output = linear
            .forward(&input)
            .expect("Module trait test forward should succeed");
        assert_eq!(output.shape(), &[5]);
    }

    #[test]
    fn test_rnn_basic_forward() {
        // Test basic RNN forward pass with known mathematical properties
        let input_size = 3;
        let hidden_size = 2;
        let seq_len = 2;
        let batch_size = 1;

        let rnn = modules::Rnn::<f32>::new(input_size, hidden_size);

        // Create input sequence: shape (seq_len, batch_size, input_size)
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 timesteps * 1 batch * 3 features
        let input = Tensor::from_vec(input_data, vec![seq_len, batch_size, input_size]);

        // Forward pass
        let (output, h_n) = rnn
            .forward(&input, None)
            .expect("RNN forward should succeed");

        // Validate output shapes
        assert_eq!(output.shape(), &[seq_len, batch_size, hidden_size]);
        assert_eq!(h_n.shape(), &[batch_size, hidden_size]);

        // Test that output is not all zeros (indicating computation occurred)
        let output_sum: f32 = output.data().iter().sum();
        assert!(output_sum.abs() > 0.0, "RNN output should not be all zeros");

        // Test that hidden state is not all zeros
        let hidden_sum: f32 = h_n.data().iter().sum();
        assert!(
            hidden_sum.abs() > 0.0,
            "RNN hidden state should not be all zeros"
        );
    }

    #[test]
    fn test_rnn_sequence_processing() {
        // Test that RNN properly processes sequences timestep by timestep
        let input_size = 2;
        let hidden_size = 3;
        let seq_len = 3;
        let batch_size = 1;

        let rnn = modules::Rnn::<f32>::new(input_size, hidden_size);

        // Create input sequence with distinct values per timestep
        let input_data = vec![
            1.0, 2.0, // timestep 0
            3.0, 4.0, // timestep 1
            5.0, 6.0, // timestep 2
        ];
        let input = Tensor::from_vec(input_data, vec![seq_len, batch_size, input_size]);

        let (output, h_n) = rnn
            .forward(&input, None)
            .expect("RNN forward should succeed");

        // Validate shapes
        assert_eq!(output.shape(), &[seq_len, batch_size, hidden_size]);
        assert_eq!(h_n.shape(), &[batch_size, hidden_size]);

        // Test that different timesteps produce different outputs
        // (This is a basic sanity check - in a real RNN, outputs should differ)
        let timestep_0_output = output.data()[0..hidden_size].iter().sum::<f32>();
        let timestep_1_output = output.data()[hidden_size..2 * hidden_size]
            .iter()
            .sum::<f32>();
        let timestep_2_output = output.data()[2 * hidden_size..3 * hidden_size]
            .iter()
            .sum::<f32>();

        // At minimum, not all timesteps should produce identical outputs
        // (This is a weak test but better than no validation)
        let outputs_identical = (timestep_0_output - timestep_1_output).abs() < 1e-10
            && (timestep_1_output - timestep_2_output).abs() < 1e-10;
        assert!(
            !outputs_identical,
            "RNN should produce different outputs for different timesteps"
        );
    }

    #[test]
    fn test_rnn_parameters() {
        // Test that RNN parameters are properly accessible
        let input_size = 4;
        let hidden_size = 3;

        let rnn = modules::Rnn::<f32>::new(input_size, hidden_size);

        // Check parameter count (weight_ih, weight_hh, bias_ih, bias_hh)
        let params = rnn.parameters();
        assert_eq!(params.len(), 4);

        // Check parameter shapes
        assert_eq!(params[0].shape(), &[hidden_size, input_size]); // weight_ih
        assert_eq!(params[1].shape(), &[hidden_size, hidden_size]); // weight_hh
        assert_eq!(params[2].shape(), &[hidden_size]); // bias_ih
        assert_eq!(params[3].shape(), &[hidden_size]); // bias_hh
    }

    #[test]
    fn test_rnn_with_initial_hidden_state() {
        // Test RNN with provided initial hidden state
        let input_size = 2;
        let hidden_size = 3;
        let seq_len = 2;
        let batch_size = 1;

        let rnn = modules::Rnn::<f32>::new(input_size, hidden_size);

        // Create input
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_vec(input_data, vec![seq_len, batch_size, input_size]);

        // Create initial hidden state
        let h_0_data = vec![0.1, 0.2, 0.3];
        let h_0 = Tensor::from_vec(h_0_data, vec![batch_size, hidden_size]);

        // Forward pass with initial state
        let (output1, _) = rnn
            .forward(&input, Some(&h_0))
            .expect("RNN forward with h_0 should succeed");

        // Forward pass without initial state (should use zeros)
        let (output2, _) = rnn
            .forward(&input, None)
            .expect("RNN forward without h_0 should succeed");

        // Outputs should be different when using non-zero initial state
        let diff: f32 = output1
            .data()
            .iter()
            .zip(output2.data().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "RNN should produce different outputs with non-zero vs zero initial state"
        );
    }

    #[test]
    fn test_rnn_gradient_flow() {
        // Test that RNN can be used in contexts requiring gradient computation
        // (Full gradient validation is complex and may require more autograd fixes)
        let input_size = 2;
        let hidden_size = 2;
        let seq_len = 1;
        let batch_size = 1;

        let rnn = modules::Rnn::<f32>::new(input_size, hidden_size);

        // Create input
        let input_data = vec![1.0, 2.0];
        let input = Tensor::from_vec(input_data, vec![seq_len, batch_size, input_size]);

        // Forward pass should work
        let (output, _) = rnn
            .forward(&input, None)
            .expect("RNN forward should succeed");

        // Basic validation that we can compute operations on the output
        let output_sum = output.sum();
        assert!(
            !output_sum.data().is_empty(),
            "Output sum should be computable"
        );

        // Test that RNN parameters exist and have expected shapes
        let params = rnn.parameters();
        assert!(!params.is_empty(), "RNN should have parameters");
        assert_eq!(
            params[0].shape(),
            &[hidden_size, input_size],
            "Weight_ih shape should be correct"
        );
    }

    #[test]
    fn test_rnn_shape_validation() {
        // Test that RNN properly validates input shapes
        let input_size = 3;
        let hidden_size = 2;
        let seq_len = 2;
        let batch_size = 1;

        let rnn = modules::Rnn::<f32>::new(input_size, hidden_size);

        // Test with wrong input size
        let wrong_input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![seq_len, batch_size, 2]); // input_size = 2, should be 3
        let result = rnn.forward(&wrong_input, None);
        assert!(
            result.is_err(),
            "RNN should reject inputs with wrong input_size"
        );

        // Test with correct input size
        let correct_input = Tensor::from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![seq_len, batch_size, input_size],
        );
        let result = rnn.forward(&correct_input, None);
        assert!(
            result.is_ok(),
            "RNN should accept inputs with correct shape"
        );
    }
}
