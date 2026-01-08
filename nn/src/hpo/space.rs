//! Hyperparameter space definitions.
//!
//! This module defines hyperparameter spaces, configurations, and sampling methods
//! for automated hyperparameter optimization.

use rand::Rng;
use std::collections::HashMap;

use crate::core::error::{NNError, Result};

/// Hyperparameter configuration value
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum HyperparameterValue {
    /// Continuous value
    Float(f64),
    /// Integer value
    Int(i64),
    /// Categorical choice
    Categorical(String),
    /// Boolean value
    Bool(bool),
}

/// Hyperparameter definition with range/constraints
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Hyperparameter {
    /// Continuous hyperparameter with bounds
    Float {
        name: String,
        min: f64,
        max: f64,
        log_scale: bool,
    },
    /// Integer hyperparameter with bounds
    Int {
        name: String,
        min: i64,
        max: i64,
        log_scale: bool,
    },
    /// Categorical hyperparameter with choices
    Categorical { name: String, choices: Vec<String> },
    /// Boolean hyperparameter
    Bool { name: String },
}

impl Hyperparameter {
    /// Sample a random value for this hyperparameter
    pub fn sample(&self) -> HyperparameterValue {
        let mut rng = rand::thread_rng();

        match self {
            Hyperparameter::Float {
                min,
                max,
                log_scale,
                ..
            } => {
                let value = if *log_scale {
                    // Log-uniform sampling
                    let log_min = min.ln();
                    let log_max = max.ln();
                    let log_sample = rng.gen_range(log_min..=log_max);
                    log_sample.exp()
                } else {
                    // Uniform sampling
                    rng.gen_range(*min..=*max)
                };
                HyperparameterValue::Float(value)
            }
            Hyperparameter::Int {
                min,
                max,
                log_scale,
                ..
            } => {
                let value = if *log_scale {
                    // Log-uniform sampling for integers
                    let log_min = (*min as f64).ln();
                    let log_max = (*max as f64).ln();
                    let log_sample = rng.gen_range(log_min..=log_max);
                    log_sample.exp() as i64
                } else {
                    // Uniform sampling
                    rng.gen_range(*min..=*max + 1)
                };
                HyperparameterValue::Int(value)
            }
            Hyperparameter::Categorical { choices, .. } => {
                let idx = rng.gen_range(0..choices.len());
                HyperparameterValue::Categorical(choices[idx].clone())
            }
            Hyperparameter::Bool { .. } => HyperparameterValue::Bool(rng.gen_bool(0.5)),
        }
    }

    /// Get the name of this hyperparameter
    pub fn name(&self) -> &str {
        match self {
            Hyperparameter::Float { name, .. } => name,
            Hyperparameter::Int { name, .. } => name,
            Hyperparameter::Categorical { name, .. } => name,
            Hyperparameter::Bool { name } => name,
        }
    }

    /// Get the dimensionality contribution of this hyperparameter
    pub fn dimensionality(&self) -> usize {
        match self {
            Hyperparameter::Float { .. } => 1,
            Hyperparameter::Int { .. } => 1,
            Hyperparameter::Categorical { choices, .. } => choices.len(),
            Hyperparameter::Bool { .. } => 1,
        }
    }
}

/// Complete hyperparameter configuration
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HyperparameterConfig {
    /// Configuration values
    pub values: HashMap<String, HyperparameterValue>,
}

impl HyperparameterConfig {
    /// Create a new empty configuration
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }
}

impl Default for HyperparameterConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperparameterConfig {
    /// Set a hyperparameter value
    pub fn set(&mut self, name: String, value: HyperparameterValue) -> &mut Self {
        self.values.insert(name, value);
        self
    }

    /// Get a hyperparameter value
    pub fn get(&self, name: &str) -> Option<&HyperparameterValue> {
        self.values.get(name)
    }

    /// Get float value with default
    pub fn get_float(&self, name: &str, default: f64) -> f64 {
        match self.values.get(name) {
            Some(HyperparameterValue::Float(v)) => *v,
            _ => default,
        }
    }

    /// Get int value with default
    pub fn get_int(&self, name: &str, default: i64) -> i64 {
        match self.values.get(name) {
            Some(HyperparameterValue::Int(v)) => *v,
            _ => default,
        }
    }

    /// Get categorical value with default
    pub fn get_categorical(&self, name: &str, default: &str) -> String {
        match self.values.get(name) {
            Some(HyperparameterValue::Categorical(v)) => v.clone(),
            _ => default.to_string(),
        }
    }

    /// Get bool value with default
    pub fn get_bool(&self, name: &str, default: bool) -> bool {
        match self.values.get(name) {
            Some(HyperparameterValue::Bool(v)) => *v,
            _ => default,
        }
    }

    /// Convert to vector representation for optimization algorithms
    pub fn to_vector(&self, space: &HyperparameterSpace) -> Vec<f64> {
        let mut vector = Vec::new();

        for param in &space.parameters {
            match self.values.get(param.name()) {
                Some(HyperparameterValue::Float(v)) => {
                    vector.push(*v);
                }
                Some(HyperparameterValue::Int(v)) => {
                    vector.push(*v as f64);
                }
                Some(HyperparameterValue::Bool(v)) => {
                    vector.push(if *v { 1.0 } else { 0.0 });
                }
                Some(HyperparameterValue::Categorical(choice)) => {
                    // One-hot encoding for categorical
                    if let Hyperparameter::Categorical { choices, .. } = param {
                        for c in choices {
                            vector.push(if c == choice { 1.0 } else { 0.0 });
                        }
                    }
                }
                None => {
                    // Default values
                    vector.push(0.0);
                }
            }
        }

        vector
    }

    /// Create from vector representation
    pub fn from_vector(vector: &[f64], space: &HyperparameterSpace) -> Result<Self> {
        let mut config = HyperparameterConfig::new();
        let mut idx = 0;

        for param in &space.parameters {
            match param {
                Hyperparameter::Float {
                    name,
                    min,
                    max,
                    log_scale,
                } => {
                    if idx >= vector.len() {
                        return Err(NNError::InvalidConfiguration {
                            message: "Vector too short for hyperparameter space".to_string(),
                        });
                    }
                    let mut value = vector[idx];

                    // Handle NaN/inf before clamping
                    if !value.is_finite() {
                        value = 0.0; // Reset non-finite values to 0
                    }

                    // Clamp to bounds
                    value = value.max(*min).min(*max);

                    if *log_scale && value <= 0.0 {
                        value = *min;
                    }

                    config.set(name.clone(), HyperparameterValue::Float(value));
                    idx += 1;
                }
                Hyperparameter::Int {
                    name,
                    min,
                    max,
                    log_scale: _,
                } => {
                    if idx >= vector.len() {
                        return Err(NNError::InvalidConfiguration {
                            message: "Vector too short for hyperparameter space".to_string(),
                        });
                    }
                    let value = vector[idx] as i64;

                    // Clamp to bounds
                    let clamped = value.max(*min).min(*max);

                    config.set(name.clone(), HyperparameterValue::Int(clamped));
                    idx += 1;
                }
                Hyperparameter::Bool { name } => {
                    if idx >= vector.len() {
                        return Err(NNError::InvalidConfiguration {
                            message: "Vector too short for hyperparameter space".to_string(),
                        });
                    }
                    let value = vector[idx] > 0.5;
                    config.set(name.clone(), HyperparameterValue::Bool(value));
                    idx += 1;
                }
                Hyperparameter::Categorical { name, choices } => {
                    // Find the one-hot encoded choice
                    let start_idx = idx;
                    let _end_idx = (idx + choices.len()).min(vector.len());

                    let mut max_prob = 0.0;
                    let mut best_choice = &choices[0];

                    for (i, choice) in choices.iter().enumerate() {
                        if start_idx + i < vector.len() {
                            let prob = vector[start_idx + i];
                            if prob > max_prob {
                                max_prob = prob;
                                best_choice = choice;
                            }
                        }
                    }

                    config.set(
                        name.clone(),
                        HyperparameterValue::Categorical(best_choice.clone()),
                    );
                    idx += choices.len();
                }
            }
        }

        Ok(config)
    }
}

/// Hyperparameter search space definition
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct HyperparameterSpace {
    /// Hyperparameters in the space
    pub parameters: Vec<Hyperparameter>,
    /// Constraints between hyperparameters
    pub constraints: Vec<HyperparameterConstraint>,
}

impl HyperparameterSpace {
    /// Create a new hyperparameter space
    pub fn new() -> Self {
        Self::default()
    }
}

impl HyperparameterSpace {
    /// Add a hyperparameter to the space
    pub fn add_parameter(&mut self, param: Hyperparameter) -> &mut Self {
        self.parameters.push(param);
        self
    }

    /// Add a constraint between hyperparameters
    pub fn add_constraint(&mut self, constraint: HyperparameterConstraint) -> &mut Self {
        self.constraints.push(constraint);
        self
    }

    /// Get the total dimensionality of the space
    pub fn dimensionality(&self) -> usize {
        self.parameters.iter().map(|p| p.dimensionality()).sum()
    }

    /// Sample a random configuration from the space
    pub fn sample(&self) -> Result<HyperparameterConfig> {
        let mut config = HyperparameterConfig::new();

        // Sample parameters
        for param in &self.parameters {
            config.set(param.name().to_string(), param.sample());
        }

        // Apply constraints (simplified - just resample if violated)
        let max_attempts = 10;
        for _ in 0..max_attempts {
            if self.check_constraints(&config) {
                return Ok(config);
            }

            // Resample
            for param in &self.parameters {
                config.set(param.name().to_string(), param.sample());
            }
        }

        // Return best effort even if constraints not satisfied
        Ok(config)
    }

    /// Check if a configuration satisfies all constraints
    pub fn check_constraints(&self, config: &HyperparameterConfig) -> bool {
        for constraint in &self.constraints {
            if !constraint.check(config) {
                return false;
            }
        }
        true
    }

    /// Create a standard neural network hyperparameter space
    pub fn neural_network_space() -> Self {
        let mut space = Self::new();

        // Learning rate (log scale)
        space.add_parameter(Hyperparameter::Float {
            name: "learning_rate".to_string(),
            min: 1e-5,
            max: 1e-1,
            log_scale: true,
        });

        // Batch size
        space.add_parameter(Hyperparameter::Int {
            name: "batch_size".to_string(),
            min: 16,
            max: 512,
            log_scale: true,
        });

        // Optimizer
        space.add_parameter(Hyperparameter::Categorical {
            name: "optimizer".to_string(),
            choices: vec!["adam".to_string(), "sgd".to_string(), "adamw".to_string()],
        });

        // Weight decay
        space.add_parameter(Hyperparameter::Float {
            name: "weight_decay".to_string(),
            min: 0.0,
            max: 1e-2,
            log_scale: false,
        });

        // Dropout rate
        space.add_parameter(Hyperparameter::Float {
            name: "dropout".to_string(),
            min: 0.0,
            max: 0.5,
            log_scale: false,
        });

        // Hidden dimension
        space.add_parameter(Hyperparameter::Int {
            name: "hidden_dim".to_string(),
            min: 64,
            max: 1024,
            log_scale: true,
        });

        space
    }

    /// Create a convolutional neural network hyperparameter space
    pub fn cnn_space() -> Self {
        let mut space = Self::neural_network_space();

        // Kernel size
        space.add_parameter(Hyperparameter::Int {
            name: "kernel_size".to_string(),
            min: 3,
            max: 11,
            log_scale: false,
        });

        // Number of filters
        space.add_parameter(Hyperparameter::Int {
            name: "num_filters".to_string(),
            min: 32,
            max: 512,
            log_scale: true,
        });

        // Activation function
        space.add_parameter(Hyperparameter::Categorical {
            name: "activation".to_string(),
            choices: vec!["relu".to_string(), "gelu".to_string(), "elu".to_string()],
        });

        space
    }
}

/// Hyperparameter constraint
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum HyperparameterConstraint {
    /// Value must be less than another
    LessThan { param1: String, param2: String },
    /// Value must be greater than another
    GreaterThan { param1: String, param2: String },
    /// Conditional constraint
    Conditional {
        condition_param: String,
        condition_value: HyperparameterValue,
        constrained_param: String,
        constraint: Box<HyperparameterConstraint>,
    },
}

impl HyperparameterConstraint {
    /// Check if constraint is satisfied
    pub fn check(&self, config: &HyperparameterConfig) -> bool {
        match self {
            HyperparameterConstraint::LessThan { param1, param2 } => {
                let val1 = config.get_float(param1, 0.0);
                let val2 = config.get_float(param2, f64::MAX);
                val1 < val2
            }
            HyperparameterConstraint::GreaterThan { param1, param2 } => {
                let val1 = config.get_float(param1, f64::MAX);
                let val2 = config.get_float(param2, 0.0);
                val1 > val2
            }
            HyperparameterConstraint::Conditional {
                condition_param,
                condition_value,
                constrained_param: _,
                constraint,
            } => {
                // Check if condition is met
                let current_value = config.get(condition_param);
                let condition_met = match (current_value, condition_value) {
                    (Some(HyperparameterValue::Float(v1)), HyperparameterValue::Float(v2)) => {
                        (v1 - v2).abs() < 1e-6
                    }
                    (Some(HyperparameterValue::Int(v1)), HyperparameterValue::Int(v2)) => v1 == v2,
                    (
                        Some(HyperparameterValue::Categorical(v1)),
                        HyperparameterValue::Categorical(v2),
                    ) => v1 == v2,
                    (Some(HyperparameterValue::Bool(v1)), HyperparameterValue::Bool(v2)) => {
                        v1 == v2
                    }
                    _ => false,
                };

                if condition_met {
                    constraint.check(config)
                } else {
                    true // Constraint doesn't apply
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperparameter_sampling() {
        let param = Hyperparameter::Float {
            name: "learning_rate".to_string(),
            min: 1e-4,
            max: 1e-1,
            log_scale: true,
        };

        let value = param.sample();
        match value {
            HyperparameterValue::Float(v) => {
                assert!((1e-4..=1e-1).contains(&v));
            }
            _ => panic!("Expected float value"),
        }
    }

    #[test]
    fn test_hyperparameter_config() {
        let mut config = HyperparameterConfig::new();
        config.set("lr".to_string(), HyperparameterValue::Float(0.01));
        config.set("batch_size".to_string(), HyperparameterValue::Int(32));

        assert_eq!(config.get_float("lr", 0.0), 0.01);
        assert_eq!(config.get_int("batch_size", 0), 32);
    }

    #[test]
    fn test_neural_network_space() {
        let space = HyperparameterSpace::neural_network_space();
        assert!(!space.parameters.is_empty());

        let config = space.sample().unwrap();
        assert!(!config.values.is_empty());
    }

    #[test]
    fn test_constraint_checking() {
        let constraint = HyperparameterConstraint::LessThan {
            param1: "a".to_string(),
            param2: "b".to_string(),
        };

        let mut config = HyperparameterConfig::new();
        config.set("a".to_string(), HyperparameterValue::Float(1.0));
        config.set("b".to_string(), HyperparameterValue::Float(2.0));

        assert!(constraint.check(&config));

        config.set("a".to_string(), HyperparameterValue::Float(3.0));
        assert!(!constraint.check(&config));
    }
}
