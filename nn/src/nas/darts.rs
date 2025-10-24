//! Differentiable Architecture Search (DARTS).
//!
//! This module implements DARTS, a method for differentiable neural architecture
//! search that uses continuous relaxation of the architecture search space.

use rand::Rng;

use super::search_space::{Architecture, ArchitectureSpace, ArchitectureType, LayerSpec};
use crate::error::{NNError, Result};

/// Architecture parameter for DARTS (continuous relaxation)
#[derive(Debug, Clone)]
pub struct ArchitectureParameter {
    /// Architecture type
    pub architecture_type: ArchitectureType,
    /// Layer operation choices (continuous parameters)
    pub layer_choices: Vec<Vec<f64>>, // [layer_idx][operation_idx] -> weight
    /// Skip connection choices
    pub skip_choices: Vec<Vec<f64>>, // [from_layer][to_layer] -> weight
    /// Temperature for softmax relaxation
    pub temperature: f64,
}

impl ArchitectureParameter {
    /// Create a new architecture parameter
    pub fn new(
        architecture_type: ArchitectureType,
        num_layers: usize,
        num_operations: usize,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let layer_choices = (0..num_layers)
            .map(|_| (0..num_operations).map(|_| rng.gen::<f64>()).collect())
            .collect();

        // Initialize skip connections (simplified)
        let skip_choices = vec![vec![0.0; num_layers]; num_layers];

        Self {
            architecture_type,
            layer_choices,
            skip_choices,
            temperature: 1.0,
        }
    }

    /// Get discrete architecture by sampling from continuous parameters
    pub fn sample_discrete(&self, search_space: &ArchitectureSpace) -> Result<Architecture> {
        let mut architecture = Architecture::new(self.architecture_type);

        // Sample input layer
        architecture.add_layer(LayerSpec::Conv2D {
            out_channels: 32,
            kernel_size: 3,
            stride: 1,
            padding: 1,
        });

        // Sample operations for each layer
        for layer_idx in 0..self.layer_choices.len() {
            let choices = &self.layer_choices[layer_idx];

            // Sample operation based on softmax probabilities
            let probs = self.softmax(choices, self.temperature);
            let sampled_op = self.sample_from_probs(&probs);

            // Convert operation index to layer spec
            let layer_spec = self.operation_to_layer_spec(sampled_op, search_space)?;
            architecture.add_layer(layer_spec);
        }

        // Add output layer
        architecture.add_layer(LayerSpec::Linear { out_features: 10 });

        // Add connections (simplified: sequential)
        for i in 0..architecture.layers.len().saturating_sub(1) {
            architecture.add_connection(i, i + 1);
        }

        architecture.validate()?;
        Ok(architecture)
    }

    /// Get the final discrete architecture (argmax)
    pub fn get_final_architecture(&self, search_space: &ArchitectureSpace) -> Result<Architecture> {
        let mut architecture = Architecture::new(self.architecture_type);

        // Input layer
        architecture.add_layer(LayerSpec::Conv2D {
            out_channels: 32,
            kernel_size: 3,
            stride: 1,
            padding: 1,
        });

        // Choose best operation for each layer
        for layer_idx in 0..self.layer_choices.len() {
            let choices = &self.layer_choices[layer_idx];
            let best_op = choices
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let layer_spec = self.operation_to_layer_spec(best_op, search_space)?;
            architecture.add_layer(layer_spec);
        }

        // Output layer
        architecture.add_layer(LayerSpec::Linear { out_features: 10 });

        // Sequential connections
        for i in 0..architecture.layers.len().saturating_sub(1) {
            architecture.add_connection(i, i + 1);
        }

        architecture.validate()?;
        Ok(architecture)
    }

    /// Convert operation index to layer specification
    fn operation_to_layer_spec(
        &self,
        op_idx: usize,
        search_space: &ArchitectureSpace,
    ) -> Result<LayerSpec> {
        if op_idx >= search_space.layer_types.len() {
            return Err(NNError::InvalidConfiguration {
                message: format!(
                    "Operation index {} out of bounds for {} operations",
                    op_idx,
                    search_space.layer_types.len()
                ),
            });
        }

        let layer_type = &search_space.layer_types[op_idx];
        let mut rng = rand::thread_rng();

        match layer_type {
            super::search_space::LayerType::Conv2D => {
                let range = search_space
                    .parameter_ranges
                    .get(layer_type)
                    .ok_or_else(|| NNError::InvalidConfiguration {
                        message: "Conv2D parameters not found".to_string(),
                    })?;
                Ok(LayerSpec::Conv2D {
                    out_channels: rng.gen_range(range.out_channels.0..=range.out_channels.1),
                    kernel_size: rng.gen_range(range.kernel_size.0..=range.kernel_size.1),
                    stride: rng.gen_range(range.stride.0..=range.stride.1),
                    padding: rng.gen_range(range.padding.0..=range.padding.1),
                })
            }
            super::search_space::LayerType::Linear => {
                let range = search_space
                    .parameter_ranges
                    .get(layer_type)
                    .ok_or_else(|| NNError::InvalidConfiguration {
                        message: "Linear parameters not found".to_string(),
                    })?;
                Ok(LayerSpec::Linear {
                    out_features: rng.gen_range(range.out_features.0..=range.out_features.1),
                })
            }
            super::search_space::LayerType::Attention => {
                let range = search_space
                    .parameter_ranges
                    .get(layer_type)
                    .ok_or_else(|| NNError::InvalidConfiguration {
                        message: "Attention parameters not found".to_string(),
                    })?;
                Ok(LayerSpec::Attention {
                    num_heads: rng.gen_range(range.num_heads.0..=range.num_heads.1),
                    sparse_pattern: None,
                })
            }
            _ => Err(NNError::InvalidConfiguration {
                message: format!("Unsupported layer type for DARTS: {:?}", layer_type),
            }),
        }
    }

    /// Softmax function with temperature
    fn softmax(&self, values: &[f64], temperature: f64) -> Vec<f64> {
        let scaled: Vec<f64> = values.iter().map(|&x| x / temperature).collect();
        let max_val = scaled.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_values: Vec<f64> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f64 = exp_values.iter().sum();

        exp_values.iter().map(|&x| x / sum).collect()
    }

    /// Sample from probability distribution
    fn sample_from_probs(&self, probs: &[f64]) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();

        let mut cumsum = 0.0;
        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if r <= cumsum {
                return i;
            }
        }

        probs.len() - 1 // fallback
    }
}

/// Differentiable Architecture Search (DARTS)
#[derive(Debug)]
pub struct DartsNAS {
    /// Architecture parameters (continuous relaxation)
    pub architecture_params: ArchitectureParameter,
    /// Search space definition
    pub search_space: ArchitectureSpace,
    /// Learning rate for architecture parameters
    pub arch_learning_rate: f64,
    /// Architecture weight decay
    pub arch_weight_decay: f64,
    /// Number of architecture updates per iteration
    pub arch_updates_per_iter: usize,
    /// Current iteration
    pub current_iteration: usize,
}

impl DartsNAS {
    /// Create a new DARTS NAS
    pub fn new(search_space: ArchitectureSpace, num_layers: usize) -> Self {
        let num_operations = search_space.layer_types.len();
        let arch_type = search_space.architecture_type;
        let architecture_params = ArchitectureParameter::new(arch_type, num_layers, num_operations);

        Self {
            architecture_params,
            search_space,
            arch_learning_rate: 0.001,
            arch_weight_decay: 0.0001,
            arch_updates_per_iter: 1,
            current_iteration: 0,
        }
    }

    /// Perform one step of differentiable architecture search
    pub fn search_step<F>(&mut self, fitness_fn: F) -> Result<f64>
    where
        F: Fn(&Architecture) -> Result<f64> + Send + Sync,
    {
        // Sample architecture from current parameters
        let architecture = self
            .architecture_params
            .sample_discrete(&self.search_space)?;
        let fitness = fitness_fn(&architecture)?;

        // Compute architecture gradients (simplified)
        // In a real DARTS implementation, this would compute gradients through the validation loss
        self.update_architecture_parameters(fitness);

        self.current_iteration += 1;

        Ok(fitness)
    }

    /// Run full DARTS search
    pub fn search<F>(&mut self, fitness_fn: F, num_iterations: usize) -> Result<Architecture>
    where
        F: Fn(&Architecture) -> Result<f64> + Send + Sync,
    {
        let mut best_fitness = f64::NEG_INFINITY;
        let mut best_architecture = None;

        for _ in 0..num_iterations {
            let fitness = self.search_step(&fitness_fn)?;

            if fitness > best_fitness {
                best_fitness = fitness;
                best_architecture = Some(
                    self.architecture_params
                        .sample_discrete(&self.search_space)?,
                );
            }
        }

        // Return the final discrete architecture
        self.architecture_params
            .get_final_architecture(&self.search_space)
    }

    /// Update architecture parameters based on fitness
    fn update_architecture_parameters(&mut self, fitness: f64) {
        // Simplified architecture parameter update
        // In real DARTS, this would use gradients from validation loss

        for layer_choices in &mut self.architecture_params.layer_choices {
            for choice in layer_choices.iter_mut() {
                // Simple gradient-like update (simplified)
                let gradient = (fitness - 0.5) * 0.01; // Simplified gradient computation
                *choice += self.arch_learning_rate * gradient;
                *choice -= self.arch_weight_decay * *choice; // L2 regularization
            }

            // Normalize to prevent explosion
            let sum: f64 = layer_choices.iter().map(|x| x.exp()).sum();
            for choice in layer_choices.iter_mut() {
                *choice = choice.exp() / sum;
                *choice = choice.ln(); // Convert back to log space
            }
        }
    }

    /// Get architecture entropy (for monitoring convergence)
    pub fn architecture_entropy(&self) -> f64 {
        let mut total_entropy = 0.0;
        let mut total_params = 0;

        for layer_choices in &self.architecture_params.layer_choices {
            if !layer_choices.is_empty() {
                let probs = self
                    .architecture_params
                    .softmax(layer_choices, self.architecture_params.temperature);
                let entropy = -probs
                    .iter()
                    .map(|&p| if p > 0.0 { p * p.ln() } else { 0.0 })
                    .sum::<f64>();
                total_entropy += entropy;
                total_params += 1;
            }
        }

        if total_params > 0 {
            total_entropy / total_params as f64
        } else {
            0.0
        }
    }

    /// Get the most likely operations for each layer
    pub fn get_most_likely_operations(&self) -> Vec<usize> {
        self.architecture_params
            .layer_choices
            .iter()
            .map(|choices| {
                choices
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }
}

/// Cell-based architecture search (common in DARTS)
#[derive(Debug)]
pub struct DartsCell {
    /// Operations in the cell
    pub operations: Vec<CellOperation>,
    /// Cell type (normal or reduction)
    pub cell_type: CellType,
}

#[derive(Debug, Clone)]
pub enum CellOperation {
    /// Convolution operations
    Conv3x3,
    Conv5x5,
    Conv7x7,
    /// Pooling operations
    MaxPool3x3,
    AvgPool3x3,
    /// Other operations
    SkipConnect,
    None,
}

#[derive(Debug, Clone, Copy)]
pub enum CellType {
    Normal,
    Reduction,
}

impl DartsCell {
    /// Create a new DARTS cell
    pub fn new(cell_type: CellType, num_nodes: usize) -> Self {
        let mut operations = Vec::new();

        // Initialize operations between nodes
        for _ in 0..num_nodes {
            for _ in 0..num_nodes {
                // Add all possible operations (in practice, this would be learned)
                operations.push(CellOperation::Conv3x3);
                operations.push(CellOperation::Conv5x5);
                operations.push(CellOperation::SkipConnect);
            }
        }

        Self {
            operations,
            cell_type,
        }
    }

    /// Sample a discrete cell architecture
    pub fn sample_discrete(&self) -> Vec<CellOperation> {
        // In a real implementation, this would sample based on learned parameters
        self.operations.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::super::search_space::{LayerType, ParameterRange};
    use super::*;

    #[test]
    fn test_architecture_parameter_creation() {
        let params = ArchitectureParameter::new(ArchitectureType::CNN, 5, 3);
        assert_eq!(params.layer_choices.len(), 5);
        assert_eq!(params.layer_choices[0].len(), 3);
    }

    #[test]
    fn test_darts_sampling() {
        let mut search_space = ArchitectureSpace::new(ArchitectureType::CNN);
        search_space.add_layer_type(LayerType::Conv2D, ParameterRange::default());
        search_space.add_layer_type(LayerType::Linear, ParameterRange::default());

        let params = ArchitectureParameter::new(ArchitectureType::CNN, 3, 2);
        let architecture = params.sample_discrete(&search_space).unwrap();

        assert!(architecture.layers.len() >= 2); // input + output + at least one hidden
        assert!(architecture.validate().is_ok());
    }

    #[test]
    fn test_darts_search_step() {
        let mut search_space = ArchitectureSpace::new(ArchitectureType::CNN);
        search_space.add_layer_type(LayerType::Conv2D, ParameterRange::default());

        let mut darts = DartsNAS::new(search_space, 3);

        // Simple fitness function
        let fitness_fn = |_: &Architecture| Ok(rand::random::<f64>());

        let fitness = darts.search_step(fitness_fn).unwrap();
        assert!(fitness.is_finite());
        assert_eq!(darts.current_iteration, 1);
    }

    #[test]
    fn test_architecture_entropy() {
        let mut search_space = ArchitectureSpace::new(ArchitectureType::CNN);
        search_space.add_layer_type(LayerType::Conv2D, ParameterRange::default());

        let darts = DartsNAS::new(search_space, 3);
        let entropy = darts.architecture_entropy();

        assert!(entropy >= 0.0);
        assert!(entropy.is_finite());
    }

    #[test]
    fn test_cell_operations() {
        let cell = DartsCell::new(CellType::Normal, 4);
        assert!(!cell.operations.is_empty());

        let sampled = cell.sample_discrete();
        assert_eq!(sampled.len(), cell.operations.len());
    }
}
