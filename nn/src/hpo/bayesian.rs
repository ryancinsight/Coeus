//! Bayesian Optimization.
//!
//! This module implements Bayesian optimization using Gaussian processes
//! as surrogate models and acquisition functions for efficient exploration.

use super::space::{HyperparameterConfig, HyperparameterSpace};
use crate::error::{NNError, Result};

/// Gaussian Process surrogate model
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    /// Training inputs (hyperparameter configurations)
    pub inputs: Vec<Vec<f64>>,
    /// Training targets (objective function values)
    pub targets: Vec<f64>,
    /// Length scale parameters
    pub length_scales: Vec<f64>,
    /// Signal variance
    pub signal_var: f64,
    /// Noise variance
    pub noise_var: f64,
}

impl GaussianProcess {
    /// Create a new Gaussian process
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a training point
    pub fn add_point(&mut self, input: Vec<f64>, target: f64) {
        self.inputs.push(input);
        self.targets.push(target);
        self.update_parameters();
    }

    /// Predict mean and variance at a test point
    pub fn predict(&self, test_input: &[f64]) -> (f64, f64) {
        if self.inputs.is_empty() {
            return (0.0, 1.0); // Prior mean and variance
        }

        // Compute covariance between test point and training points
        let mut k_star = Vec::with_capacity(self.inputs.len());
        for train_input in &self.inputs {
            let cov = self.se_kernel(train_input, test_input);
            k_star.push(cov);
        }

        // Compute covariance matrix of training points
        let k_matrix = self.compute_covariance_matrix();

        // Add noise to diagonal
        let mut k_matrix_noisy = k_matrix.clone();
        for (i, row) in k_matrix_noisy.iter_mut().enumerate() {
            row[i] += self.noise_var;
        }

        // Compute predictive mean and variance
        let _k_star_t = k_star.clone();
        let alpha = self.solve_linear_system(&k_matrix_noisy, &self.targets);

        let mean = k_star
            .iter()
            .zip(alpha.iter())
            .map(|(k, a)| k * a)
            .sum::<f64>();

        // Compute variance
        let v = self.solve_linear_system(&k_matrix_noisy, &k_star);
        let variance = self.signal_var
            - k_star
                .iter()
                .zip(v.iter())
                .map(|(ks, vs)| ks * vs)
                .sum::<f64>();

        (mean, variance.max(0.0))
    }

    /// Squared exponential kernel
    fn se_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let mut sum = 0.0;
        for (i, (a, b)) in x1.iter().zip(x2.iter()).enumerate() {
            let length_scale = self.length_scales.get(i).copied().unwrap_or(1.0);
            sum += (a - b).powi(2) / (2.0 * length_scale.powi(2));
        }
        self.signal_var * (-sum).exp()
    }

    /// Compute covariance matrix
    fn compute_covariance_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.inputs.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for (i, row) in matrix.iter_mut().enumerate().take(n) {
            #[allow(clippy::needless_range_loop)]
            for j in 0..n {
                row[j] = self.se_kernel(&self.inputs[i], &self.inputs[j]);
            }
        }

        matrix
    }

    /// Solve linear system Ax = b using Gaussian elimination (simplified)
    fn solve_linear_system(&self, a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        // Very basic implementation - in practice, use a proper linear algebra library
        let n = b.len();
        let mut x = vec![0.0; n];

        // Simple forward substitution for diagonal dominant matrix
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += a[i][j] * x[j];
            }
            x[i] = (b[i] - sum) / a[i][i].max(1e-10);
        }

        x
    }

    /// Update GP parameters using maximum likelihood estimation
    fn update_parameters(&mut self) {
        if self.inputs.is_empty() {
            return;
        }

        let dim = self.inputs[0].len();

        // Initialize length scales
        if self.length_scales.len() != dim {
            self.length_scales = vec![1.0; dim];
        }

        // Simple parameter optimization (in practice, use gradient-based optimization)
        for _ in 0..10 {
            // Limited optimization steps
            let mut gradients = vec![0.0; dim + 2]; // length_scales + signal_var + noise_var

            // Compute gradients (simplified)
            #[allow(clippy::needless_range_loop)]
            for i in 0..dim {
                let eps = 1e-4;
                let original = self.length_scales[i];

                // Numerical gradient for length scale
                self.length_scales[i] = original + eps;
                let loss_plus = self.negative_log_likelihood();

                self.length_scales[i] = original - eps;
                let loss_minus = self.negative_log_likelihood();

                self.length_scales[i] = original;
                gradients[i] = (loss_plus - loss_minus) / (2.0 * eps);
            }

            // Update parameters (gradient descent)
            for (i, length_scale) in self.length_scales.iter_mut().enumerate().take(dim) {
                *length_scale -= 0.01 * gradients[i];
                *length_scale = length_scale.max(1e-3);
            }
        }
    }

    /// Compute negative log likelihood
    fn negative_log_likelihood(&self) -> f64 {
        if self.inputs.is_empty() {
            return 0.0;
        }

        let k_matrix = self.compute_covariance_matrix();
        let mut k_noisy = k_matrix.clone();
        for (i, row) in k_noisy.iter_mut().enumerate() {
            row[i] += self.noise_var;
        }

        // Compute log determinant and quadratic form (simplified)
        let n = self.targets.len() as f64;
        let log_det = k_noisy
            .iter()
            .enumerate()
            .map(|(i, row)| row[i].ln())
            .sum::<f64>();

        let alpha = self.solve_linear_system(&k_noisy, &self.targets);
        let quad_form = self
            .targets
            .iter()
            .zip(alpha.iter())
            .map(|(t, a)| t * a)
            .sum::<f64>();

        0.5 * (n * (2.0 * std::f64::consts::PI).ln() + log_det + quad_form)
    }
}

impl Default for GaussianProcess {
    fn default() -> Self {
        Self {
            inputs: Vec::new(),
            targets: Vec::new(),
            length_scales: Vec::new(),
            signal_var: 1.0,
            noise_var: 1e-6,
        }
    }
}

/// Acquisition functions for Bayesian optimization
#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    ExpectedImprovement { xi: f64 },
    /// Upper Confidence Bound
    UpperConfidenceBound { beta: f64 },
    /// Probability of Improvement
    ProbabilityOfImprovement { tau: f64 },
}

impl AcquisitionFunction {
    /// Evaluate acquisition function
    pub fn evaluate(&self, mean: f64, std: f64, best_value: f64) -> f64 {
        match self {
            AcquisitionFunction::ExpectedImprovement { xi } => {
                self.expected_improvement(mean, std, best_value, *xi)
            }
            AcquisitionFunction::UpperConfidenceBound { beta } => mean + beta.sqrt() * std,
            AcquisitionFunction::ProbabilityOfImprovement { tau } => {
                self.probability_of_improvement(mean, std, *tau)
            }
        }
    }

    /// Expected Improvement acquisition function
    fn expected_improvement(&self, mean: f64, std: f64, best_value: f64, xi: f64) -> f64 {
        let improvement = best_value - mean - xi;
        let z = improvement / std.max(1e-9);

        if std <= 1e-9 {
            return 0.0;
        }

        improvement * Self::normal_cdf(z) + std * Self::normal_pdf(z)
    }

    /// Probability of Improvement acquisition function
    fn probability_of_improvement(&self, mean: f64, std: f64, tau: f64) -> f64 {
        if std <= 1e-9 {
            return if mean > tau { 1.0 } else { 0.0 };
        }

        let z = (mean - tau) / std;
        1.0 - Self::normal_cdf(z)
    }

    /// Standard normal CDF (approximation)
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + Self::erf(x / 2.0_f64.sqrt()))
    }

    /// Standard normal PDF
    fn normal_pdf(x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    /// Error function approximation
    fn erf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }
}

/// Bayesian Optimization algorithm
#[derive(Debug)]
pub struct BayesianOptimizer {
    /// Gaussian process surrogate model
    pub gp: GaussianProcess,
    /// Acquisition function
    pub acquisition: AcquisitionFunction,
    /// Hyperparameter space
    pub space: HyperparameterSpace,
    /// Best observed value
    pub best_value: f64,
    /// Optimization iterations
    pub iterations: usize,
}

impl BayesianOptimizer {
    /// Create a new Bayesian optimizer
    pub fn new(space: HyperparameterSpace) -> Self {
        Self {
            gp: GaussianProcess::new(),
            acquisition: AcquisitionFunction::ExpectedImprovement { xi: 0.01 },
            space,
            best_value: f64::INFINITY,
            iterations: 0,
        }
    }

    /// Suggest next hyperparameter configuration to evaluate
    pub fn suggest(&mut self) -> Result<HyperparameterConfig> {
        if self.gp.inputs.is_empty() {
            // Initial random suggestion
            return self.space.sample();
        }

        // Optimize acquisition function
        let mut best_acq = f64::NEG_INFINITY;
        let mut best_config = None;

        // Random search over acquisition function (in practice, use proper optimization)
        for _ in 0..100 {
            let candidate = self.space.sample()?;
            let vector = candidate.to_vector(&self.space);

            let (mean, var) = self.gp.predict(&vector);
            let std = var.sqrt();
            let acq_value = self.acquisition.evaluate(mean, std, self.best_value);

            if acq_value > best_acq {
                best_acq = acq_value;
                best_config = Some(candidate);
            }
        }

        best_config.ok_or_else(|| NNError::InvalidConfiguration {
            message: "Could not find optimal acquisition".to_string(),
        })
    }

    /// Observe the result of evaluating a configuration
    pub fn observe(&mut self, config: &HyperparameterConfig, value: f64) {
        let vector = config.to_vector(&self.space);
        self.gp.add_point(vector, value);

        if value < self.best_value {
            self.best_value = value;
        }

        self.iterations += 1;
    }

    /// Run full Bayesian optimization
    pub fn optimize<F>(
        &mut self,
        objective: F,
        max_evaluations: usize,
    ) -> Result<crate::hpo::optimizer::OptimizationResult>
    where
        F: Fn(&HyperparameterConfig) -> Result<f64> + Send + Sync,
    {
        // Initial evaluations
        for _ in 0..5.min(max_evaluations) {
            let config = self.space.sample()?;
            let value = objective(&config)?;
            self.observe(&config, value);
        }

        // Main optimization loop
        for _ in 5..max_evaluations {
            let config = self.suggest()?;
            let value = objective(&config)?;
            self.observe(&config, value);
        }

        // Return best configuration found
        let mut best_config = None;
        let mut best_value = f64::INFINITY;

        for (i, target) in self.gp.targets.iter().enumerate() {
            if *target < best_value {
                best_value = *target;
                let vector = &self.gp.inputs[i];
                best_config = Some(HyperparameterConfig::from_vector(vector, &self.space)?);
            }
        }

        let best_config = best_config.ok_or_else(|| NNError::InvalidConfiguration {
            message: "No evaluations performed".to_string(),
        })?;

        Ok(crate::hpo::optimizer::OptimizationResult {
            best_config,
            best_value,
            evaluations: self.gp.targets.len(),
            total_time: std::time::Duration::from_secs(0), // TODO: track actual time
            history: vec![],                               // TODO: populate history
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::space::Hyperparameter;
    use super::*;

    #[test]
    fn test_gaussian_process() {
        let mut gp = GaussianProcess::new();

        // Add some training points
        gp.add_point(vec![0.0], 1.0);
        gp.add_point(vec![1.0], 2.0);
        gp.add_point(vec![2.0], 3.0);

        // Test prediction
        let (mean, var) = gp.predict(&[1.5]);
        assert!(mean.is_finite());
        assert!(var >= 0.0);
    }

    #[test]
    fn test_acquisition_functions() {
        let ei = AcquisitionFunction::ExpectedImprovement { xi: 0.01 };
        let ucb = AcquisitionFunction::UpperConfidenceBound { beta: 2.0 };
        let pi = AcquisitionFunction::ProbabilityOfImprovement { tau: 1.0 };

        let mean = 0.5;
        let std = 0.1;
        let best = 1.0;

        assert!(ei.evaluate(mean, std, best).is_finite());
        assert!(ucb.evaluate(mean, std, best).is_finite());
        assert!(pi.evaluate(mean, std, best).is_finite());
    }

    #[test]
    fn test_bayesian_optimizer() {
        let mut space = HyperparameterSpace::new();
        space.add_parameter(Hyperparameter::Float {
            name: "x".to_string(),
            min: -2.0,
            max: 2.0,
            log_scale: false,
        });

        let mut optimizer = BayesianOptimizer::new(space);

        // Simple quadratic objective
        let objective = |config: &HyperparameterConfig| {
            let x = config.get_float("x", 0.0);
            Ok(x * x) // Minimize x^2
        };

        let result = optimizer.optimize(objective, 10).unwrap();
        let x = result.best_config.get_float("x", 0.0);

        // Should be close to 0
        assert!(x.abs() < 1.0);
    }
}
