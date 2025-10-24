//! Meta-Learning Optimizers.
//!
//! This module implements learnable optimization algorithms that can adapt
//! their update rules during meta-learning.

use rand::Rng;

/// Learnable optimizer that adapts its learning rate per parameter
#[derive(Debug)]
pub struct MetaSGD {
    /// Base learning rate
    pub base_lr: f64,
    /// Learned per-parameter learning rates (log scale)
    pub learned_lrs: Vec<f64>,
}

impl MetaSGD {
    /// Create a new MetaSGD optimizer
    pub fn new(base_lr: f64, num_params: usize) -> Self {
        Self {
            base_lr,
            learned_lrs: vec![0.0; num_params], // Initialize to zero (lr = base_lr * exp(0))
        }
    }

    /// Get learning rate for a parameter
    pub fn get_lr(&self, param_idx: usize) -> f64 {
        self.base_lr * self.learned_lrs[param_idx].exp()
    }

    /// Update learned learning rates
    pub fn update_lr(&mut self, param_idx: usize, lr_gradient: f64, meta_lr: f64) {
        self.learned_lrs[param_idx] -= meta_lr * lr_gradient;
    }
}

/// Learnable LSTM-based optimizer
#[derive(Debug)]
pub struct MetaLSTM {
    /// Hidden state size
    pub hidden_size: usize,
    /// LSTM cell state
    pub cell_state: Vec<f64>,
    /// LSTM hidden state
    pub hidden_state: Vec<f64>,
    /// Input-to-hidden weights
    pub w_ih: Vec<Vec<f64>>,
    /// Hidden-to-hidden weights
    pub w_hh: Vec<Vec<f64>>,
    /// Biases
    pub b_ih: Vec<f64>,
    pub b_hh: Vec<f64>,
}

impl MetaLSTM {
    /// Create a new MetaLSTM optimizer
    pub fn new(hidden_size: usize, input_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        Self {
            hidden_size,
            cell_state: vec![0.0; hidden_size],
            hidden_state: vec![0.0; hidden_size],
            w_ih: (0..hidden_size * 4)
                .map(|_| (0..input_size).map(|_| rng.gen_range(-0.1..=0.1)).collect())
                .collect(),
            w_hh: (0..hidden_size * 4)
                .map(|_| {
                    (0..hidden_size)
                        .map(|_| rng.gen_range(-0.1..=0.1))
                        .collect()
                })
                .collect(),
            b_ih: vec![0.0; hidden_size * 4],
            b_hh: vec![0.0; hidden_size * 4],
        }
    }

    /// Compute update for a gradient
    pub fn compute_update(&mut self, gradient: f64, param_idx: usize) -> f64 {
        // Simplified LSTM computation for single gradient input
        // In practice, this would be much more sophisticated

        let _input = vec![gradient, self.hidden_state[param_idx % self.hidden_size]];

        // Compute gates (simplified)
        let forget_gate = 0.5; // sigmoid(w_f * [grad, h] + b_f)
        let input_gate = 0.5; // sigmoid(w_i * [grad, h] + b_i)
        let output_gate = 0.5; // sigmoid(w_o * [grad, h] + b_o)
        let candidate = gradient.tanh(); // tanh(w_c * [grad, h] + b_c)

        // Update cell state
        self.cell_state[param_idx % self.hidden_size] =
            forget_gate * self.cell_state[param_idx % self.hidden_size] + input_gate * candidate;

        // Update hidden state
        self.hidden_state[param_idx % self.hidden_size] =
            output_gate * self.cell_state[param_idx % self.hidden_size].tanh();

        // Return update (simplified)
        -0.01 * gradient
    }
}

/// Learnable Adam-style optimizer
#[derive(Debug)]
pub struct MetaAdam {
    /// Base learning rate
    pub base_lr: f64,
    /// Learned beta1 parameters
    pub learned_beta1: Vec<f64>,
    /// Learned beta2 parameters
    pub learned_beta2: Vec<f64>,
    /// First moment estimates
    pub m: Vec<f64>,
    /// Second moment estimates
    pub v: Vec<f64>,
    /// Timestep
    pub t: usize,
}

impl MetaAdam {
    /// Create a new MetaAdam optimizer
    pub fn new(base_lr: f64, num_params: usize) -> Self {
        Self {
            base_lr,
            learned_beta1: vec![0.9; num_params], // Initialize to standard Adam beta1
            learned_beta2: vec![0.999; num_params], // Initialize to standard Adam beta2
            m: vec![0.0; num_params],
            v: vec![0.0; num_params],
            t: 0,
        }
    }

    /// Compute Adam update for a parameter
    pub fn compute_update(&mut self, gradient: f64, param_idx: usize) -> f64 {
        self.t += 1;

        let beta1 = self.learned_beta1[param_idx].clamp(0.0, 0.999);
        let beta2 = self.learned_beta2[param_idx].clamp(0.0, 0.999);

        // Update biased first moment estimate
        self.m[param_idx] = beta1 * self.m[param_idx] + (1.0 - beta1) * gradient;

        // Update biased second raw moment estimate
        self.v[param_idx] = beta2 * self.v[param_idx] + (1.0 - beta2) * gradient * gradient;

        // Compute bias-corrected first moment estimate
        let m_hat = self.m[param_idx] / (1.0 - beta1.powi(self.t as i32));

        // Compute bias-corrected second raw moment estimate
        let v_hat = self.v[param_idx] / (1.0 - beta2.powi(self.t as i32));

        // Compute update
        let lr = self.base_lr * (1.0 - beta2.powi(self.t as i32)).sqrt()
            / (1.0 - beta1.powi(self.t as i32));
        -lr * m_hat / (v_hat.sqrt() + 1e-8)
    }

    /// Update learned beta parameters
    pub fn update_betas(
        &mut self,
        param_idx: usize,
        beta1_grad: f64,
        beta2_grad: f64,
        meta_lr: f64,
    ) {
        self.learned_beta1[param_idx] -= meta_lr * beta1_grad;
        self.learned_beta2[param_idx] -= meta_lr * beta2_grad;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_sgd() {
        let mut optimizer = MetaSGD::new(0.01, 10);

        assert_eq!(optimizer.get_lr(0), 0.01); // exp(0) * 0.01 = 0.01

        // Update learning rate
        optimizer.update_lr(0, 1.0, 0.001);
        assert!(optimizer.get_lr(0) < 0.01); // Should decrease
    }

    #[test]
    fn test_meta_lstm() {
        let mut optimizer = MetaLSTM::new(10, 1);

        let update = optimizer.compute_update(1.0, 0);
        assert!(update < 0.0); // Should be negative for positive gradient
    }

    #[test]
    fn test_meta_adam() {
        let mut optimizer = MetaAdam::new(0.001, 10);

        let update = optimizer.compute_update(1.0, 0);
        assert!(update < 0.0); // Should be negative for positive gradient
    }
}
