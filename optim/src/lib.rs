//! # Coeus Optim
//!
//! PyTorch-style optimization algorithms for the Coeus tensor library.
//!
//! This crate provides a comprehensive set of optimization algorithms
//! compatible with PyTorch's `torch.optim` module, designed for training
//! neural networks and other machine learning models.
//!
//! ## Features
//!
//! - **Gradient-based Optimization**: SGD, Adam, AdamW, RMSprop, and more
//! - **Learning Rate Scheduling**: StepLR, ExponentialLR, CosineAnnealingLR
//! - **Weight Decay**: L2 regularization support
//! - **Parameter Groups**: Different settings for different parameter groups
//! - **State Management**: Automatic optimizer state tracking
//!
//! ## Quick Start
//!
//! ```rust
//! use coeus_optim::{Adam, Optimizer};
//! use coeus_tensor::Tensor;
//!
//! // Create some model parameters
//! let mut param1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
//! let mut param2 = Tensor::from_vec(vec![0.5, -0.5], vec![2]);
//! param1.set_requires_grad(true);
//! param2.set_requires_grad(true);
//!
//! // Create optimizer
//! let mut optimizer = Adam::new(vec![param1.clone(), param2.clone()], 0.001);
//!
//! // Training loop
//! for _ in 0..100 {
//!     // Compute loss and gradients...
//!
//!     // Update parameters
//!     optimizer.step();
//!     optimizer.zero_grad();
//! }
//! ```
//!
//! ## Supported Optimizers
//!
//! - **SGD**: Stochastic Gradient Descent with momentum and weight decay
//! - **Adam**: Adaptive Moment Estimation optimizer
//! - **AdamW**: Adam with decoupled weight decay
//! - **RMSprop**: Root Mean Square Propagation
//! - **Adagrad**: Adaptive Gradient Algorithm
//! - **Adadelta**: Adaptive learning rate method
//! - **LBFGS**: Limited-memory BFGS (planned)
//!
//! ## Learning Rate Schedulers
//!
//! - **StepLR**: Decays learning rate by gamma every step_size epochs
//! - **ExponentialLR**: Decays learning rate exponentially
//! - **CosineAnnealingLR**: Cosine annealing learning rate schedule
//! - **ReduceLROnPlateau**: Reduces LR when metric stops improving

pub mod error;
pub mod optimizer;
pub mod optimizers;
pub mod schedulers;

/// Result type alias for optimization operations
pub type Result<T> = std::result::Result<T, error::OptimError>;

pub use error::OptimError;
pub use optimizer::{BaseOptimizer, Optimizer, ParamGroup};
pub use optimizers::*;
pub use schedulers::*;

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_tensor::Tensor;

    #[test]
    fn test_param_group_operations() {
        let param1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let param2 = Tensor::from_vec(vec![4.0, 5.0], vec![2]);

        // Create parameter group
        let mut group = ParamGroup::new(vec![param1.clone(), param2.clone()], 0.01, 0.0001);

        assert_eq!(group.lr, 0.01);
        assert_eq!(group.weight_decay, 0.0001);
        assert_eq!(group.params.len(), 2);

        // Test parameter access
        assert_eq!(group.parameters().len(), 2);
        assert_eq!(group.parameters()[0].shape(), &[3]);
        assert_eq!(group.parameters()[1].shape(), &[2]);

        // Test mutable access
        assert_eq!(group.parameters_mut().len(), 2);

        // Test option setting
        group = group.with_option("momentum", 0.9);
        assert_eq!(group.get_option("momentum"), Some(&0.9));
        assert_eq!(group.get_option("nonexistent"), None);
    }

    #[test]
    fn test_param_group_from_slice() {
        let params = vec![
            Tensor::from_vec(vec![1.0, 2.0], vec![2]),
            Tensor::from_vec(vec![3.0], vec![1]),
        ];

        let group = ParamGroup::from_params(&params, 0.001, 0.0);

        assert_eq!(group.lr, 0.001);
        assert_eq!(group.weight_decay, 0.0);
        assert_eq!(group.params.len(), 2);
    }

    #[test]
    fn test_adam_optimizer_creation() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3])];
        let optimizer = Adam::new(params.clone(), 0.001);

        assert_eq!(optimizer.name(), "Adam");
        assert_eq!(optimizer.param_groups().len(), 1);
        assert_eq!(optimizer.param_groups()[0].lr, 0.001);

        // Test default parameters
        assert_eq!(optimizer.beta1(), 0.9);
        assert_eq!(optimizer.beta2(), 0.999);
        assert_eq!(optimizer.eps(), 1e-8);
        assert!(!optimizer.amsgrad());
    }

    #[test]
    fn test_adam_with_custom_options() {
        let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let optimizer = Adam::with_options(params, 0.01, 0.8, 0.95, 1e-10, true);

        assert_eq!(optimizer.beta1(), 0.8);
        assert_eq!(optimizer.beta2(), 0.95);
        assert_eq!(optimizer.eps(), 1e-10);
        assert!(optimizer.amsgrad());
    }

    #[test]
    fn test_adam_zero_grad() {
        let mut param = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        param.set_requires_grad(true);

        // Manually set a gradient (simplified test)
        // In real usage, gradients would come from backpropagation

        let mut optimizer = Adam::new(vec![param], 0.001);
        optimizer.zero_grad();

        // Verify zero_grad doesn't crash and maintains structure
        assert_eq!(optimizer.param_groups().len(), 1);
    }

    #[test]
    fn test_adam_learning_rate_management() {
        let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let mut optimizer = Adam::new(params, 0.01);

        // Test getting learning rate
        assert_eq!(optimizer.get_lr(0), Some(0.01));
        assert_eq!(optimizer.get_lr(1), None); // Invalid group index

        // Test setting learning rate
        assert!(optimizer.set_lr(0, 0.001).is_ok());
        assert_eq!(optimizer.get_lr(0), Some(0.001));

        // Test setting invalid group index
        assert!(optimizer.set_lr(1, 0.001).is_err());
    }

    #[test]
    fn test_adam_parameter_groups() {
        let params1 = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
        let params2 = vec![Tensor::from_vec(vec![3.0], vec![1])];

        let mut optimizer = Adam::new(params1, 0.01);
        optimizer.add_param_group(ParamGroup::new(params2, 0.001, 0.0));

        assert_eq!(optimizer.param_groups().len(), 2);
        assert_eq!(optimizer.param_groups()[0].lr, 0.01);
        assert_eq!(optimizer.param_groups()[1].lr, 0.001);
    }

    #[test]
    fn test_sgd_optimizer_creation() {
        let params = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
        let optimizer = Sgd::new(params.clone(), 0.01);

        assert_eq!(optimizer.name(), "SGD");
        assert_eq!(optimizer.param_groups().len(), 1);
        assert_eq!(optimizer.param_groups()[0].lr, 0.01);
        assert_eq!(optimizer.momentum(), 0.0); // Default no momentum
    }

    #[test]
    fn test_sgd_with_momentum() {
        let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let optimizer = Sgd::with_momentum(params, 0.01, 0.9);

        assert_eq!(optimizer.momentum(), 0.9_f32);
    }

    #[test]
    fn test_step_lr_scheduler() {
        let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let mut optimizer = Adam::new(params, 0.01);

        // Initial learning rate
        assert_eq!(optimizer.get_lr(0), Some(0.01));

        // Create and use scheduler, checking LR after dropping scheduler
        {
            let mut scheduler = StepLR::new(&mut optimizer, 5, 0.1);
            // Step scheduler (should not change LR yet)
            for _ in 0..4 {
                let _ = scheduler.step();
            }
        }
        assert_eq!(optimizer.get_lr(0), Some(0.01));

        // 5th step should change LR
        {
            let mut scheduler = StepLR::new(&mut optimizer, 5, 0.1);
            for _ in 0..5 {
                let _ = scheduler.step();
            }
        }
        assert_eq!(optimizer.get_lr(0), Some(0.001)); // 0.01 * 0.1
    }

    #[test]
    fn test_exponential_lr_scheduler() {
        let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let mut optimizer = Adam::new(params, 0.01);

        assert_eq!(optimizer.get_lr(0), Some(0.01));

        {
            let mut scheduler = ExponentialLR::new(&mut optimizer, 0.9);
            let _ = scheduler.step();
        }
        let lr = optimizer.get_lr(0).unwrap();
        assert!((lr - 0.009_f64).abs() < 1e-10_f64); // 0.01 * 0.9, with floating point tolerance

        {
            let mut scheduler = ExponentialLR::new(&mut optimizer, 0.9);
            let _ = scheduler.step();
        }
        let lr2 = optimizer.get_lr(0).unwrap();
        assert!((lr2 - 0.0081_f64).abs() < 1e-10_f64); // 0.009 * 0.9, with floating point tolerance
    }

    #[test]
    fn test_cosine_annealing_lr_scheduler() {
        let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let mut optimizer = Adam::new(params, 0.01);

        // Cosine annealing over 10 steps
        let _scheduler = CosineAnnealingLR::new(&mut optimizer, 10, 0.0);

        // Test that it starts at initial LR
        assert_eq!(optimizer.get_lr(0), Some(0.01));

        // Test that it ends near minimum after max_steps
        {
            let mut temp_scheduler = CosineAnnealingLR::new(&mut optimizer, 10, 0.0);
            for _ in 0..10 {
                let _ = temp_scheduler.step();
            }
        }
        // After 10 steps, should be close to eta_min (0.0)
        let final_lr = optimizer.get_lr(0).unwrap();
        assert!((0.0..0.001).contains(&final_lr)); // Very small but not exactly 0
    }

    #[test]
    fn test_optimizer_state_management() {
        let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let optimizer = Adam::new(params, 0.001);

        // Test that optimizer has state (even if empty initially)
        let state = optimizer.state();
        assert!(state.is_empty()); // Initially empty

        // Test that different optimizer types work
        let sgd = Sgd::new(vec![Tensor::from_vec(vec![1.0], vec![1])], 0.01);
        assert_eq!(sgd.name(), "SGD");

        let rmsprop = Rmsprop::new(vec![Tensor::from_vec(vec![1.0], vec![1])], 0.001);
        assert_eq!(rmsprop.name(), "RMSprop");

        let adagrad = Adagrad::new(vec![Tensor::from_vec(vec![1.0], vec![1])], 0.01);
        assert_eq!(adagrad.name(), "Adagrad");

        let adamw = AdamW::new(vec![Tensor::from_vec(vec![1.0], vec![1])], 0.001);
        assert_eq!(adamw.name(), "AdamW");
    }

    #[test]
    fn test_multiple_parameter_groups() {
        let params1 = vec![Tensor::from_vec(vec![1.0, 2.0], vec![2])];
        let params2 = vec![Tensor::from_vec(vec![3.0], vec![1])];

        let mut optimizer = Adam::new(params1, 0.01);
        optimizer.add_param_group(ParamGroup::new(params2, 0.001, 0.0));

        assert_eq!(optimizer.param_groups().len(), 2);
        assert_eq!(optimizer.get_lr(0), Some(0.01));
        assert_eq!(optimizer.get_lr(1), Some(0.001));
    }

    #[test]
    fn test_error_conditions() {
        let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let mut optimizer = Adam::new(params, 0.01);

        // Test invalid group index for set_lr
        let result = optimizer.set_lr(5, 0.001); // Group 5 doesn't exist
        assert!(result.is_err());

        // Test invalid group index for get_lr
        assert_eq!(optimizer.get_lr(5), None);
    }

    #[test]
    fn test_scheduler_edge_cases() {
        let params = vec![Tensor::from_vec(vec![1.0], vec![1])];
        let mut optimizer = Adam::new(params, 0.01);

        // Test StepLR with step_size = 1 (changes every step)
        assert_eq!(optimizer.get_lr(0), Some(0.01));

        {
            let mut step_scheduler = StepLR::new(&mut optimizer, 1, 0.5);
            let _ = step_scheduler.step();
        }
        assert_eq!(optimizer.get_lr(0), Some(0.005)); // 0.01 * 0.5

        {
            let mut step_scheduler = StepLR::new(&mut optimizer, 1, 0.5);
            let _ = step_scheduler.step();
        }
        assert_eq!(optimizer.get_lr(0), Some(0.0025)); // 0.005 * 0.5

        // Test ExponentialLR with gamma = 1.0 (no change)
        let mut optimizer2 = Adam::new(vec![Tensor::from_vec(vec![1.0], vec![1])], 0.01);
        assert_eq!(optimizer2.get_lr(0), Some(0.01));

        {
            let mut exp_scheduler = ExponentialLR::new(&mut optimizer2, 1.0); // gamma = 1.0 (no change)
            let _ = exp_scheduler.step();
        }
        assert_eq!(optimizer2.get_lr(0), Some(0.01)); // Should remain the same
    }
}
