//! Multi-Fidelity Hyperparameter Optimization.
//!
//! This module implements multi-fidelity optimization techniques including
//! Hyperband, Successive Halving, and BOHB (Bayesian Optimization with
//! Hyperband).

use super::space::{HyperparameterConfig, HyperparameterSpace};
use crate::core::error::{NNError, Result};
use std::time::Duration;

/// Bracket configuration for Hyperband
#[derive(Debug)]
pub struct BracketConfig {
    /// Number of configurations for this bracket
    pub num_configs: usize,
    /// Resource allocation for this bracket
    pub resource: f64,
}

/// Hyperband optimizer
#[derive(Debug)]
pub struct HyperbandOptimizer {
    /// Hyperparameter space
    pub space: HyperparameterSpace,
    /// Bracket configurations
    pub bracket_configs: Vec<BracketConfig>,
    /// Maximum resource allocation
    pub max_resource: f64,
}

impl HyperbandOptimizer {
    /// Create a new Hyperband optimizer
    pub fn new(space: HyperparameterSpace) -> Self {
        let max_resource = 100.0f64; // Maximum "epochs" or resource units
        let reduction_factor = 3.0f64; // Reduce resources by factor of 3 each round

        // Calculate number of brackets
        let num_brackets = ((max_resource).ln() / reduction_factor.ln()) as usize + 1;

        let mut bracket_configs = Vec::new();

        for bracket in 0..num_brackets {
            let s = bracket as f64;
            let num_configs = (reduction_factor.powf(num_brackets as f64 - 1.0 - s)
                * (num_brackets as f64 + 1.0)
                / (s + 1.0))
                .ceil() as usize;

            bracket_configs.push(BracketConfig {
                num_configs,
                resource: reduction_factor.powf(s),
            });
        }

        Self {
            space,
            bracket_configs,
            max_resource,
        }
    }

    /// Run a single bracket
    pub fn run_bracket<F>(&self, bracket_idx: usize, objective: F) -> Result<BracketResult>
    where
        F: Fn(&HyperparameterConfig, f64) -> Result<f64> + Send + Sync,
    {
        if bracket_idx >= self.bracket_configs.len() {
            return Err(NNError::InvalidConfiguration {
                message: "Invalid bracket index".to_string(),
            });
        }

        let config = &self.bracket_configs[bracket_idx];
        let mut configs = Vec::new();
        let mut history = Vec::new();

        // Sample initial configurations
        for _ in 0..config.num_configs {
            let hyper_config = self.space.sample()?;
            configs.push(hyper_config);
        }

        // Evaluate at current resource level
        for hyper_config in &configs {
            let fitness = objective(hyper_config, config.resource)?;
            history.push((hyper_config.clone(), fitness, Duration::from_secs(1)));
        }

        // Sort and keep top half
        configs.sort_by(|a, b| {
            let fitness_a = history
                .iter()
                .find(|(c, _, _)| c == a)
                .map(|(_, f, _)| f)
                .unwrap_or(&f64::INFINITY);
            let fitness_b = history
                .iter()
                .find(|(c, _, _)| c == b)
                .map(|(_, f, _)| f)
                .unwrap_or(&f64::INFINITY);
            fitness_a.partial_cmp(fitness_b).unwrap()
        });

        let keep_count = (configs.len() + 1) / 2;
        configs.truncate(keep_count);

        let best_config = configs.first().cloned();
        let best_fitness = best_config
            .as_ref()
            .and_then(|c| {
                history
                    .iter()
                    .find(|(config, _, _)| config == c)
                    .map(|(_, f, _)| f)
            })
            .copied();

        Ok(BracketResult {
            bracket: bracket_idx,
            best_config,
            best_fitness,
            evaluations: history.len(),
            history,
        })
    }
}

/// Bracket result from Hyperband
#[derive(Debug)]
pub struct BracketResult {
    /// Bracket index
    pub bracket: usize,
    /// Best configuration found
    pub best_config: Option<HyperparameterConfig>,
    /// Best fitness achieved
    pub best_fitness: Option<f64>,
    /// Number of evaluations performed
    pub evaluations: usize,
    /// History of evaluations
    pub history: Vec<(HyperparameterConfig, f64, Duration)>,
}

/// Successive halving optimizer
#[derive(Debug)]
pub struct SuccessiveHalving {
    /// Hyperparameter space
    pub space: HyperparameterSpace,
    /// Initial resource allocation
    pub initial_resource: f64,
    /// Resource multiplier between rounds
    pub resource_multiplier: f64,
    /// Initial number of configurations
    pub initial_configs: usize,
    /// Current round
    pub current_round: usize,
}

impl SuccessiveHalving {
    /// Create a new successive halving optimizer
    pub fn new(space: HyperparameterSpace) -> Self {
        Self {
            space,
            initial_resource: 1.0,
            resource_multiplier: 2.0,
            initial_configs: 16,
            current_round: 0,
        }
    }

    /// Run a round of successive halving
    pub fn run_round<F>(
        &mut self,
        configs: &mut Vec<HyperparameterConfig>,
        resource: f64,
        objective: F,
    ) -> Result<Vec<f64>>
    where
        F: Fn(&HyperparameterConfig, f64) -> Result<f64> + Send + Sync,
    {
        let mut fitnesses = Vec::new();

        for config in configs.iter() {
            let fitness = objective(config, resource)?;
            fitnesses.push(fitness);
        }

        // Sort configurations by fitness
        let mut config_fitness: Vec<_> = configs
            .iter()
            .cloned()
            .zip(fitnesses.iter().cloned())
            .collect();
        config_fitness.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Keep top half
        let keep_count = (config_fitness.len() + 1) / 2;
        *configs = config_fitness
            .into_iter()
            .take(keep_count)
            .map(|(c, _)| c)
            .collect();

        Ok(fitnesses)
    }
}

/// BOHB (Bayesian Optimization with Hyperband) optimizer
#[derive(Debug)]
pub struct BohbOptimizer {
    /// Hyperparameter space
    pub space: HyperparameterSpace,
    /// Bandwidth for KDE sampling
    pub bandwidth: f64,
}

impl BohbOptimizer {
    /// Create a new BOHB optimizer
    pub fn new(space: HyperparameterSpace) -> Self {
        Self {
            space,
            bandwidth: 0.1,
        }
    }

    /// Sample a configuration using KDE
    pub fn sample_with_kde(
        &self,
        history: &[(HyperparameterConfig, f64, f64)],
    ) -> Result<HyperparameterConfig> {
        // Simplified KDE sampling - just return a random configuration for now
        if history.is_empty() {
            return self.space.sample();
        }

        // In practice, this would fit a KDE to the good configurations
        // and sample from it. For now, return a random configuration.
        self.space.sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperband_creation() {
        let space = HyperparameterSpace::neural_network_space();
        let hyperband = HyperbandOptimizer::new(space);

        assert!(!hyperband.bracket_configs.is_empty());
        assert!(hyperband.max_resource > 0.0);
    }
}
