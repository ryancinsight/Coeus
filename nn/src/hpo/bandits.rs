//! Multi-Armed Bandit Algorithms.
//!
//! This module implements various multi-armed bandit algorithms for
//! hyperparameter optimization, including Thompson sampling, UCB, and EXP3.

use rand::Rng;

use super::optimizer::OptimizationResult;
use super::space::{HyperparameterConfig, HyperparameterSpace};
use crate::error::Result;

/// Multi-armed bandit algorithm types
#[derive(Debug, Clone)]
pub enum BanditAlgorithm {
    /// Thompson Sampling
    ThompsonSampling,
    /// Upper Confidence Bound
    UpperConfidenceBound { c: f64 },
    /// Exponential-weight algorithm for Exploration and Exploitation (EXP3)
    Exp3 { gamma: f64 },
}

/// Multi-armed bandit optimizer
#[derive(Debug)]
pub struct BanditOptimizer {
    /// Bandit algorithm to use
    pub algorithm: BanditAlgorithm,
    /// Hyperparameter space
    pub space: HyperparameterSpace,
    /// Arms (discretized hyperparameter configurations)
    pub arms: Vec<HyperparameterConfig>,
    /// Rewards for each arm
    pub rewards: Vec<Vec<f64>>,
    /// Counts for each arm
    pub counts: Vec<usize>,
    /// Current iteration
    pub iteration: usize,
}

impl BanditOptimizer {
    /// Create a new bandit optimizer
    pub fn new(space: HyperparameterSpace) -> Self {
        Self {
            algorithm: BanditAlgorithm::ThompsonSampling,
            space,
            arms: Vec::new(),
            rewards: Vec::new(),
            counts: Vec::new(),
            iteration: 0,
        }
    }

    /// Set the bandit algorithm
    pub fn with_algorithm(mut self, algorithm: BanditAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Initialize arms by discretizing the hyperparameter space
    pub fn initialize_arms(&mut self, num_arms: usize) -> Result<()> {
        self.arms.clear();
        self.rewards.clear();
        self.counts.clear();

        // Sample random configurations as arms
        for _ in 0..num_arms {
            let config = self.space.sample()?;
            self.arms.push(config);
            self.rewards.push(Vec::new());
            self.counts.push(0);
        }

        Ok(())
    }

    /// Select an arm using the bandit algorithm
    pub fn select_arm(&mut self) -> usize {
        match &self.algorithm {
            BanditAlgorithm::ThompsonSampling => self.thompson_sampling(),
            BanditAlgorithm::UpperConfidenceBound { c } => self.ucb(*c),
            BanditAlgorithm::Exp3 { gamma } => self.exp3(*gamma),
        }
    }

    /// Thompson sampling implementation
    fn thompson_sampling(&self) -> usize {
        let mut rng = rand::thread_rng();
        let mut best_arm = 0;
        let mut best_sample = f64::NEG_INFINITY;

        for (i, rewards) in self.rewards.iter().enumerate() {
            let count = self.counts[i];
            if count == 0 {
                return i; // Try untried arms first
            }

            let mean = rewards.iter().sum::<f64>() / count as f64;
            let variance = if count > 1 {
                rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (count - 1) as f64
            } else {
                1.0
            };

            // Sample from posterior (assuming Gaussian likelihood)
            let sample = rng.gen::<f64>() * variance.sqrt() + mean;
            if sample > best_sample {
                best_sample = sample;
                best_arm = i;
            }
        }

        best_arm
    }

    /// Upper Confidence Bound implementation
    fn ucb(&self, c: f64) -> usize {
        let total_pulls = self.counts.iter().sum::<usize>() as f64;

        let mut best_arm = 0;
        let mut best_ucb = f64::NEG_INFINITY;

        for (i, count) in self.counts.iter().enumerate() {
            let mean = if *count > 0 {
                self.rewards[i].iter().sum::<f64>() / *count as f64
            } else {
                0.0
            };

            let ucb_value = if *count == 0 {
                f64::INFINITY // Always try untried arms
            } else {
                mean + c * (total_pulls.ln() / *count as f64).sqrt()
            };

            if ucb_value > best_ucb {
                best_ucb = ucb_value;
                best_arm = i;
            }
        }

        best_arm
    }

    /// EXP3 implementation
    fn exp3(&self, gamma: f64) -> usize {
        let mut rng = rand::thread_rng();
        let num_arms = self.arms.len() as f64;

        // Calculate weights
        let mut weights = vec![1.0; self.arms.len()];
        for (i, rewards) in self.rewards.iter().enumerate() {
            if !rewards.is_empty() {
                let estimated_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;
                weights[i] *= (gamma * estimated_reward / num_arms).exp();
            }
        }

        let weight_sum: f64 = weights.iter().sum();

        // Sample from probability distribution
        let r: f64 = rng.gen();
        let mut cumsum = 0.0;

        for (i, &weight) in weights.iter().enumerate() {
            cumsum += (1.0 - gamma) * (weight / weight_sum) + gamma / num_arms;
            if r <= cumsum {
                return i;
            }
        }

        self.arms.len() - 1 // fallback
    }

    /// Update the selected arm with observed reward
    pub fn update_arm(&mut self, arm_idx: usize, reward: f64) {
        self.rewards[arm_idx].push(reward);
        self.counts[arm_idx] += 1;
    }

    /// Get statistics for each arm
    pub fn arm_statistics(&self) -> Vec<ArmStats> {
        self.arms
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let count = self.counts[i];
                let mean = if count > 0 {
                    self.rewards[i].iter().sum::<f64>() / count as f64
                } else {
                    0.0
                };

                let variance = if count > 1 {
                    self.rewards[i]
                        .iter()
                        .map(|r| (r - mean).powi(2))
                        .sum::<f64>()
                        / (count - 1) as f64
                } else {
                    0.0
                };

                ArmStats {
                    index: i,
                    pulls: count,
                    mean_reward: mean,
                    variance,
                    best_reward: self.rewards[i]
                        .iter()
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                    worst_reward: self.rewards[i].iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                }
            })
            .collect()
    }
}

impl BanditOptimizer {
    /// Run hyperparameter optimization
    pub fn optimize<F>(
        &mut self,
        objective: F,
        max_evaluations: usize,
    ) -> Result<OptimizationResult>
    where
        F: Fn(&HyperparameterConfig) -> Result<f64> + Send + Sync,
    {
        use std::time::Instant;

        let start_time = Instant::now();

        // Initialize arms if not done
        if self.arms.is_empty() {
            self.initialize_arms(max_evaluations.min(50))?; // Limit to reasonable number of arms
        }

        let mut history = Vec::new();
        let mut best_value = f64::INFINITY;
        let mut best_config = self.arms[0].clone();

        // Bandit optimization loop
        for _ in 0..max_evaluations {
            let arm_idx = self.select_arm();
            let config = self.arms[arm_idx].clone(); // Clone to avoid borrowing issues

            let value = objective(&config)?;
            self.update_arm(arm_idx, -value); // Negative because bandits maximize, we minimize

            if value < best_value {
                best_value = value;
                best_config = config.clone();
            }

            history.push((config, value, start_time.elapsed()));
        }

        Ok(OptimizationResult {
            best_config,
            best_value,
            evaluations: max_evaluations,
            total_time: start_time.elapsed(),
            history,
        })
    }

    /// Get optimizer name
    pub fn name(&self) -> &str {
        match &self.algorithm {
            BanditAlgorithm::ThompsonSampling => "ThompsonSampling",
            BanditAlgorithm::UpperConfidenceBound { .. } => "UCB",
            BanditAlgorithm::Exp3 { .. } => "EXP3",
        }
    }
}

/// Statistics for a bandit arm
#[derive(Debug, Clone)]
pub struct ArmStats {
    /// Arm index
    pub index: usize,
    /// Number of pulls
    pub pulls: usize,
    /// Mean reward
    pub mean_reward: f64,
    /// Reward variance
    pub variance: f64,
    /// Best reward observed
    pub best_reward: f64,
    /// Worst reward observed
    pub worst_reward: f64,
}

/// Contextual bandit for hyperparameter optimization
pub struct ContextualBandit {
    /// Feature extractor for configurations
    pub feature_extractor: Box<dyn Fn(&HyperparameterConfig) -> Vec<f64>>,
    /// Linear bandit parameters (one per feature)
    pub parameters: Vec<f64>,
    /// Learning rate
    pub learning_rate: f64,
    /// Regularization parameter
    pub regularization: f64,
}

impl ContextualBandit {
    /// Create a new contextual bandit
    pub fn new(feature_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let parameters = (0..feature_dim).map(|_| rng.gen::<f64>() * 0.1).collect();

        Self {
            feature_extractor: Box::new(|_| Vec::new()), // Default no-op
            parameters,
            learning_rate: 0.01,
            regularization: 0.01,
        }
    }

    /// Set feature extractor
    pub fn with_feature_extractor<F>(mut self, extractor: F) -> Self
    where
        F: Fn(&HyperparameterConfig) -> Vec<f64> + 'static,
    {
        self.feature_extractor = Box::new(extractor);
        self
    }

    /// Predict reward for a configuration
    pub fn predict(&self, config: &HyperparameterConfig) -> f64 {
        let features = (self.feature_extractor)(config);
        if features.len() != self.parameters.len() {
            return 0.0; // Default prediction
        }

        features
            .iter()
            .zip(&self.parameters)
            .map(|(f, p)| f * p)
            .sum::<f64>()
    }

    /// Update parameters based on observed reward
    pub fn update(&mut self, config: &HyperparameterConfig, observed_reward: f64) {
        let features = (self.feature_extractor)(config);
        if features.len() != self.parameters.len() {
            return;
        }

        let prediction = self.predict(config);
        let error = observed_reward - prediction;

        // Update parameters (linear regression with regularization)
        for (i, feature) in features.iter().enumerate() {
            self.parameters[i] += self.learning_rate * error * feature;
            self.parameters[i] -= self.learning_rate * self.regularization * self.parameters[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::space::Hyperparameter;
    use super::*;

    #[test]
    fn test_bandit_initialization() {
        let mut space = HyperparameterSpace::new();
        space.add_parameter(Hyperparameter::Float {
            name: "x".to_string(),
            min: -1.0,
            max: 1.0,
            log_scale: false,
        });

        let mut bandit = BanditOptimizer::new(space);
        bandit.initialize_arms(5).unwrap();

        assert_eq!(bandit.arms.len(), 5);
        assert_eq!(bandit.counts.len(), 5);
        assert_eq!(bandit.rewards.len(), 5);
    }

    #[test]
    fn test_thompson_sampling() {
        let space = HyperparameterSpace::neural_network_space();
        let mut bandit = BanditOptimizer::new(space);

        // Add some fake data
        bandit.arms.push(HyperparameterConfig::new());
        bandit.rewards.push(vec![1.0, 2.0, 3.0]);
        bandit.counts.push(3);

        bandit.arms.push(HyperparameterConfig::new());
        bandit.rewards.push(vec![0.5, 1.5]);
        bandit.counts.push(2);

        let selected = bandit.thompson_sampling();
        assert!(selected < bandit.arms.len());
    }

    #[test]
    fn test_contextual_bandit() {
        let mut bandit = ContextualBandit::new(3);

        bandit = bandit.with_feature_extractor(|config| {
            vec![
                config.get_float("x", 0.0),
                config.get_float("y", 0.0),
                1.0, // bias term
            ]
        });

        let mut config = HyperparameterConfig::new();
        config.set(
            "x".to_string(),
            super::super::space::HyperparameterValue::Float(1.0),
        );
        config.set(
            "y".to_string(),
            super::super::space::HyperparameterValue::Float(2.0),
        );

        let prediction = bandit.predict(&config);
        assert!(prediction.is_finite());

        bandit.update(&config, 5.0);
        let new_prediction = bandit.predict(&config);
        // Prediction should change after update
        assert_ne!(prediction, new_prediction);
    }

    #[test]
    fn test_bandit_optimizer() {
        let mut space = HyperparameterSpace::new();
        space.add_parameter(Hyperparameter::Float {
            name: "x".to_string(),
            min: -2.0,
            max: 2.0,
            log_scale: false,
        });

        let mut optimizer = BanditOptimizer::new(space);

        let objective = |config: &HyperparameterConfig| {
            let x = config.get_float("x", 0.0);
            Ok(x * x) // Minimize x^2
        };

        let result = optimizer.optimize(objective, 10).unwrap();
        assert!(result.best_value >= 0.0);
        assert_eq!(result.evaluations, 10);
    }
}
