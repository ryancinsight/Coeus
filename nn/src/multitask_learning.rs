//! Multi-Task Learning Frameworks (Sprint MS-47)
//!
//! This module implements advanced multi-task learning architectures that enable
//! joint training across multiple related tasks, with shared representations and
//! task-specific heads to improve generalization and efficiency.

use std::collections::HashMap;
use crate::error::{NNError, Result};
use crate::linear::Linear;
use crate::layernorm::LayerNorm;
use crate::activation::GeLU;
use crate::attention::MultiHeadAttention;
use backend::Backend;
use storage::{Storage, StorageFromVec, StorageToDense};
use dtype::{DataType, FloatExt};

/// Supported task types for multi-task learning
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// Classification task
    Classification,
    /// Regression task
    Regression,
    /// Sequence generation task
    Generation,
    /// Ranking/retrieval task
    Ranking,
    /// Multi-label classification
    MultiLabel,
    /// Custom task type
    Custom(String),
}

/// Multi-task learning strategy
#[derive(Debug, Clone)]
pub enum MTLStrategy {
    /// Hard parameter sharing: all tasks share the same encoder
    HardParameterSharing,
    /// Soft parameter sharing: separate encoders with regularization
    SoftParameterSharing,
    /// Hierarchical sharing: different levels of sharing
    HierarchicalSharing,
    /// Task-specific adapters on shared backbone
    AdapterBased,
    /// Cross-task attention for task interaction
    CrossTaskAttention,
}

/// Task configuration
#[derive(Debug, Clone)]
pub struct TaskConfig {
    /// Task type
    pub task_type: TaskType,
    /// Task name/identifier
    pub task_name: String,
    /// Input dimension for this task
    pub input_dim: usize,
    /// Output dimension for this task (classes for classification, etc.)
    pub output_dim: usize,
    /// Task-specific loss weight
    pub loss_weight: f64,
    /// Whether this task is active during training
    pub active: bool,
    /// Task-specific parameters
    pub params: HashMap<String, f64>,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            task_type: TaskType::Classification,
            task_name: "default".to_string(),
            input_dim: 768,
            output_dim: 1,
            loss_weight: 1.0,
            active: true,
            params: HashMap::new(),
        }
    }
}

/// Multi-task transformer model
#[derive(Debug)]
pub struct MultiTaskTransformer<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + 'static,
{
    /// Shared encoder layers
    pub shared_encoder: Vec<TransformerBlock<B, S, T>>,
    /// Task-specific heads
    pub task_heads: HashMap<String, TaskHead<B, S, T>>,
    /// Task configurations
    pub task_configs: HashMap<String, TaskConfig>,
    /// MTL strategy
    pub strategy: MTLStrategy,
    /// Global configuration
    pub config: MTLConfig,
}

/// Multi-task learning configuration
#[derive(Debug, Clone)]
pub struct MTLConfig {
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of shared layers
    pub num_shared_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Feed-forward dimension
    pub ff_dim: usize,
    /// Dropout probability
    pub dropout: f64,
    /// MTL strategy
    pub strategy: MTLStrategy,
    /// Task loss weighting strategy
    pub loss_weighting: LossWeighting,
}

impl Default for MTLConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            num_shared_layers: 6,
            num_heads: 12,
            ff_dim: 3072,
            dropout: 0.1,
            strategy: MTLStrategy::HardParameterSharing,
            loss_weighting: LossWeighting::Uniform,
        }
    }
}

/// Loss weighting strategies for multi-task learning
#[derive(Debug, Clone)]
pub enum LossWeighting {
    /// Equal weight for all tasks
    Uniform,
    /// Uncertainty-based weighting (Kendall et al.)
    Uncertainty,
    /// Gradient norm-based weighting (Chen et al.)
    GradNorm,
    /// Dynamic weighting based on task difficulty
    Dynamic,
    /// Manual weighting specified in task configs
    Manual,
}

/// Task-specific head architecture
#[derive(Debug)]
pub enum TaskHead<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + StorageFromVec<T> + StorageToDense<T> + Clone + 'static,
    T: DataType,
{
    /// Standard classification/regression head
    Standard(Linear<B, S, T>),
    /// Sequence generation head with language modeling
    Generation {
        lm_head: Linear<B, S, T>,
        vocab_size: usize,
    },
    /// Multi-label classification head
    MultiLabel {
        classifier: Linear<B, S, T>,
        num_labels: usize,
        threshold: f64,
    },
    /// Ranking head with comparison capabilities
    Ranking {
        scorer: Linear<B, S, T>,
        margin: f64,
    },
}

#[derive(Debug)]
pub struct TransformerBlock<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt,
{
    pub attention: MultiHeadAttention<B, S, T>,
    pub feed_forward: FeedForwardNetwork<B, S, T>,
    pub norm1: LayerNorm<B, S, T>,
    pub norm2: LayerNorm<B, S, T>,
    pub dropout: f64,
}

#[derive(Debug)]
pub struct FeedForwardNetwork<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + dtype::FloatExt,
{
    pub linear1: Linear<B, S, T>,
    pub linear2: Linear<B, S, T>,
    pub activation: GeLU<B, S, T>,
    pub dropout: f64,
}

/// Uncertainty-based loss weighting for multi-task learning
#[derive(Debug)]
pub struct UncertaintyWeighting {
    /// Task-specific log variance parameters (learned)
    pub log_vars: HashMap<String, f64>,
    /// Initial log variance value
    pub init_log_var: f64,
}

impl UncertaintyWeighting {
    pub fn new(tasks: &[String]) -> Self {
        let mut log_vars = HashMap::new();
        for task in tasks {
            log_vars.insert(task.clone(), 0.0); // Initialize to 0 (var=1)
        }
        Self {
            log_vars,
            init_log_var: 0.0,
        }
    }

    /// Compute uncertainty-weighted loss
    pub fn weight_loss(&self, task_losses: &HashMap<String, f64>) -> HashMap<String, f64> {
        let mut weighted_losses = HashMap::new();

        // Compute total weighted loss for normalization
        let total_weight: f64 = self.log_vars.values().map(|log_var| (2.0 * log_var).exp()).sum();

        for (task, loss) in task_losses {
            if let Some(log_var) = self.log_vars.get(task) {
                let precision = (2.0 * log_var).exp(); // 1/variance
                let weighted_loss = loss * precision;
                weighted_losses.insert(task.clone(), weighted_loss);
            }
        }

        weighted_losses
    }

    /// Get current precision weights for tasks
    pub fn get_weights(&self) -> HashMap<String, f64> {
        self.log_vars.iter()
            .map(|(task, log_var)| (task.clone(), (2.0 * log_var).exp()))
            .collect()
    }
}

/// GradNorm-based loss weighting for multi-task learning
#[derive(Debug)]
pub struct GradNormWeighting {
    /// Target gradient norm ratios
    pub target_ratios: HashMap<String, f64>,
    /// Current gradient norms
    pub current_norms: HashMap<String, f64>,
    /// Learning rate for ratio adjustment
    pub alpha: f64,
    /// Initial task losses
    pub initial_losses: HashMap<String, f64>,
}

impl GradNormWeighting {
    pub fn new(tasks: &[String], alpha: f64) -> Self {
        let target_ratio = 1.0 / tasks.len() as f64;
        let mut target_ratios = HashMap::new();
        let mut current_norms = HashMap::new();
        let mut initial_losses = HashMap::new();

        for task in tasks {
            target_ratios.insert(task.clone(), target_ratio);
            current_norms.insert(task.clone(), 1.0);
            initial_losses.insert(task.clone(), 1.0);
        }

        Self {
            target_ratios,
            current_norms,
            alpha,
            initial_losses,
        }
    }

    /// Update gradient norms and compute weights
    pub fn update_and_weight(&mut self, grad_norms: &HashMap<String, f64>, task_losses: &HashMap<String, f64>) -> HashMap<String, f64> {
        // Update initial losses if not set
        for (task, loss) in task_losses {
            if let Some(init_loss) = self.initial_losses.get_mut(task) {
                if *init_loss == 1.0 { // Still default
                    *init_loss = *loss;
                }
            }
        }

        // Update current norms
        for (task, norm) in grad_norms {
            if let Some(current_norm) = self.current_norms.get_mut(task) {
                *current_norm = *norm;
            }
        }

        // Compute loss ratios
        let mut loss_ratios = HashMap::new();
        for (task, current_loss) in task_losses {
            if let Some(init_loss) = self.initial_losses.get(task) {
                let ratio = current_loss / init_loss;
                loss_ratios.insert(task.clone(), ratio);
            }
        }

        // Compute gradient norm ratios relative to average
        let avg_norm: f64 = grad_norms.values().sum::<f64>() / grad_norms.len() as f64;
        let mut norm_ratios = HashMap::new();

        for (task, norm) in grad_norms {
            let ratio = norm / avg_norm;
            norm_ratios.insert(task.clone(), ratio);

            // Update target ratios
            if let Some(target_ratio) = self.target_ratios.get_mut(task) {
                let loss_ratio = loss_ratios.get(task).unwrap_or(&1.0);
                let grad_diff = ratio - *target_ratio;
                let loss_grad = loss_ratio - 1.0;
                *target_ratio -= self.alpha * grad_diff * loss_grad;
                *target_ratio = target_ratio.max(0.0); // Ensure non-negative
            }
        }

        // Compute weights based on target ratios
        let total_target: f64 = self.target_ratios.values().sum();
        let mut weights = HashMap::new();

        for (task, _) in grad_norms {
            let target = self.target_ratios.get(task).unwrap_or(&0.0);
            let weight = target / total_target;
            weights.insert(task.clone(), weight);
        }

        weights
    }
}

/// Adapter-based multi-task learning with task-specific adapters
#[derive(Debug)]
pub struct TaskAdapter<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + dtype::FloatExt,
{
    /// Down projection for adapter
    pub down_proj: Linear<B, S, T>,
    /// Up projection for adapter
    pub up_proj: Linear<B, S, T>,
    /// Adapter activation
    pub activation: GeLU<B, S, T>,
    /// Adapter scale parameter
    pub scale: f64,
}

impl<B, S, T> TaskAdapter<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::FromPrimitive + num_traits::Bounded + 'static,
{
    pub fn new(hidden_dim: usize, adapter_dim: usize) -> Result<Self> {
        Ok(Self {
            down_proj: Linear::new(hidden_dim, adapter_dim)?,
            up_proj: Linear::new(adapter_dim, hidden_dim)?,
            activation: GeLU::new(),
            scale: 1.0,
        })
    }

    pub fn forward(&self, hidden_states: &[f32]) -> Result<Vec<f32>> {
        // Adapter forward: x + scale * up_proj(activation(down_proj(x)))
        // Placeholder implementation
        Ok(hidden_states.to_vec())
    }
}

/// Cross-task attention mechanism
#[derive(Debug)]
pub struct CrossTaskAttention<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    /// Attention mechanism for cross-task interaction
    pub attention: MultiHeadAttention<B, S, T>,
    /// Task embeddings for attention
    pub task_embeddings: HashMap<String, Vec<f32>>,
    /// Query projection for tasks
    pub query_proj: Linear<B, S, T>,
    /// Key/Value projections for tasks
    pub kv_proj: Linear<B, S, T>,
    /// Output projection
    pub out_proj: Linear<B, S, T>,
}

impl<B, S, T> MultiTaskTransformer<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::FromPrimitive + num_traits::Bounded + 'static,
{
    /// Create new multi-task transformer
    pub fn new(config: MTLConfig, task_configs: HashMap<String, TaskConfig>) -> Result<Self> {
        // Create shared encoder layers
        let mut shared_encoder = Vec::new();
        for _ in 0..config.num_shared_layers {
            shared_encoder.push(TransformerBlock::new(
                config.hidden_dim,
                config.num_heads,
                config.ff_dim,
            )?);
        }

        // Create task heads
        let mut task_heads = HashMap::new();
        for (task_name, task_config) in &task_configs {
            let head = Self::create_task_head(task_config)?;
            task_heads.insert(task_name.clone(), head);
        }

        let strategy = config.strategy.clone();
        Ok(Self {
            shared_encoder,
            task_heads,
            task_configs,
            strategy,
            config,
        })
    }

    /// Add a new task to the multi-task model
    pub fn add_task(&mut self, task_name: String, task_config: TaskConfig) -> Result<()> {
        // Create task head
        let head = Self::create_task_head(&task_config)?;
        self.task_heads.insert(task_name.clone(), head);
        self.task_configs.insert(task_name, task_config);
        Ok(())
    }

    /// Forward pass for a specific task
    pub fn forward_task(&self, input: &[f32], task_name: &str, batch_size: usize) -> Result<Vec<f32>> {
        // Shared encoding
        let mut hidden_states = input.to_vec();

        // Apply shared encoder layers
        for layer in &self.shared_encoder {
            hidden_states = layer.forward(&hidden_states, batch_size)?;
        }

        // Task-specific processing (e.g., adapters)
        hidden_states = self.apply_task_specific_processing(&hidden_states, task_name)?;

        // Apply task head
        if let Some(head) = self.task_heads.get(task_name) {
            self.apply_task_head(&hidden_states, head)
        } else {
            Err(NNError::InvalidInput {
                message: format!("Unknown task: {}", task_name),
            })
        }
    }

    /// Forward pass for multiple tasks simultaneously
    pub fn forward_multi_task(
        &self,
        inputs: HashMap<String, &[f32]>,
        batch_size: usize,
    ) -> Result<HashMap<String, Vec<f32>>> {
        let mut outputs = HashMap::new();

        // Process each task
        for (task_name, input) in inputs {
            let output = self.forward_task(input, &task_name, batch_size)?;
            outputs.insert(task_name, output);
        }

        Ok(outputs)
    }

    /// Create task head based on task configuration
    fn create_task_head(task_config: &TaskConfig) -> Result<TaskHead<B, S, T>>
    {
        match task_config.task_type {
            TaskType::Classification => {
                Ok(TaskHead::Standard(Linear::new(task_config.input_dim, task_config.output_dim)?))
            },
            TaskType::Generation => {
                // For simplicity, assume output_dim represents vocab size
                Ok(TaskHead::Generation {
                    lm_head: Linear::new(task_config.input_dim, task_config.output_dim)?,
                    vocab_size: task_config.output_dim,
                })
            },
            TaskType::Regression => {
                Ok(TaskHead::Standard(Linear::new(task_config.input_dim, task_config.output_dim)?))
            },
            TaskType::MultiLabel => {
                Ok(TaskHead::MultiLabel {
                    classifier: Linear::new(task_config.input_dim, task_config.output_dim)?,
                    num_labels: task_config.output_dim,
                    threshold: 0.5,
                })
            },
            TaskType::Ranking => {
                Ok(TaskHead::Ranking {
                    scorer: Linear::new(task_config.input_dim, 1)?,
                    margin: 1.0,
                })
            },
            TaskType::Custom(_) => {
                // Default to standard head
                Ok(TaskHead::Standard(Linear::new(task_config.input_dim, task_config.output_dim)?))
            }
        }
    }

    fn apply_task_specific_processing(&self, hidden_states: &[f32], task_name: &str) -> Result<Vec<f32>> {
        match &self.strategy {
            MTLStrategy::AdapterBased => {
                // Apply task-specific adapter if available
                // For now, return unchanged
                Ok(hidden_states.to_vec())
            },
            _ => Ok(hidden_states.to_vec()),
        }
    }

    fn apply_task_head(&self, hidden_states: &[f32], head: &TaskHead<B, S, T>) -> Result<Vec<f32>> {
        match head {
            TaskHead::Standard(linear) => {
                // Apply linear head (placeholder)
                Ok(vec![0.0; linear.out_features])
            },
            TaskHead::Generation { lm_head, vocab_size } => {
                // Apply generation head (placeholder)
                Ok(vec![0.0; *vocab_size])
            },
            TaskHead::MultiLabel { classifier, num_labels, threshold } => {
                // Apply multi-label head (placeholder)
                Ok(vec![0.0; *num_labels])
            },
            TaskHead::Ranking { scorer, margin } => {
                // Apply ranking head (placeholder)
                Ok(vec![0.0; 1])
            },
        }
    }

    /// Get task configurations
    pub fn get_task_configs(&self) -> &HashMap<String, TaskConfig> {
        &self.task_configs
    }

    /// Update task loss weights based on strategy
    pub fn update_loss_weights(&mut self, task_losses: &HashMap<String, f64>, grad_norms: Option<&HashMap<String, f64>>) {
        match &self.config.loss_weighting {
            LossWeighting::Uniform => {
                // Equal weights - no update needed
            },
            LossWeighting::Manual => {
                // Weights set manually in task configs - no update needed
            },
            LossWeighting::Uncertainty => {
                // Would update uncertainty weights based on losses
                // This requires additional implementation
            },
            LossWeighting::GradNorm => {
                // Would update GradNorm weights based on gradient norms
                // This requires additional implementation
            },
            LossWeighting::Dynamic => {
                // Dynamic weighting logic here
            },
        }
    }
}

impl<B, S, T> TransformerBlock<B, S, T>
where
    B: Backend<Data = T> + Clone + Default,
    S: Storage<T> + Clone + StorageFromVec<T> + storage::StorageToDense<T> + 'static,
    T: DataType + FloatExt + 'static + num_traits::FromPrimitive + num_traits::Bounded,
{
    pub fn new(hidden_dim: usize, num_heads: usize, ff_dim: usize) -> Result<Self> {
        Ok(Self {
            attention: MultiHeadAttention::new(num_heads, hidden_dim)?,
            feed_forward: FeedForwardNetwork::new(hidden_dim, ff_dim)?,
            norm1: LayerNorm::new(vec![hidden_dim], 1e-6),
            norm2: LayerNorm::new(vec![hidden_dim], 1e-6),
            dropout: 0.1,
        })
    }

    pub fn forward(&self, hidden_states: &[f32], batch_size: usize) -> Result<Vec<f32>> {
        // Self-attention with residual
        // let attn_output = self.attention.forward(hidden_states, batch_size)?;
        // let normalized = self.norm1.forward(&attn_output)?;
        // Add residual: normalized + hidden_states

        // Feed-forward with residual
        // let ff_output = self.feed_forward.forward(&normalized, batch_size)?;
        // let normalized_ff = self.norm2.forward(&ff_output)?;
        // Final residual: normalized_ff + normalized

        // Placeholder
        Ok(hidden_states.to_vec())
    }
}

impl<B, S, T> FeedForwardNetwork<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
        T: DataType + FloatExt + num_traits::FromPrimitive + num_traits::Bounded + 'static,
{
    pub fn new(hidden_dim: usize, ff_dim: usize) -> Result<Self> {
        Ok(Self {
            linear1: Linear::new(hidden_dim, ff_dim)?,
            linear2: Linear::new(ff_dim, hidden_dim)?,
            activation: GeLU::new(),
            dropout: 0.1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_task_transformer_creation() {
        let config = MTLConfig::default();
        let mut task_configs = HashMap::new();

        task_configs.insert("classification".to_string(), TaskConfig {
            task_type: TaskType::Classification,
            task_name: "classification".to_string(),
            input_dim: 768,
            output_dim: 10,
            loss_weight: 1.0,
            active: true,
            params: HashMap::new(),
        });

        let result = MultiTaskTransformer::new(config, task_configs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_uncertainty_weighting() {
        let tasks = vec!["task1".to_string(), "task2".to_string()];
        let uw = UncertaintyWeighting::new(&tasks);

        let task_losses = HashMap::from([
            ("task1".to_string(), 0.5),
            ("task2".to_string(), 1.0),
        ]);

        let weighted = uw.weight_loss(&task_losses);
        assert!(weighted.contains_key("task1"));
        assert!(weighted.contains_key("task2"));
    }

    #[test]
    fn test_grad_norm_weighting() {
        let tasks = vec!["task1".to_string(), "task2".to_string()];
        let mut gnw = GradNormWeighting::new(&tasks, 0.1);

        let grad_norms = HashMap::from([
            ("task1".to_string(), 1.0),
            ("task2".to_string(), 2.0),
        ]);

        let task_losses = HashMap::from([
            ("task1".to_string(), 0.5),
            ("task2".to_string(), 1.0),
        ]);

        let weights = gnw.update_and_weight(&grad_norms, &task_losses);
        assert!(weights.contains_key("task1"));
        assert!(weights.contains_key("task2"));
    }

    #[test]
    fn test_task_adapter_creation() {
        let result = TaskAdapter::new(768, 64);
        assert!(result.is_ok());
    }
}
