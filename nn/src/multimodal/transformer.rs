//! # Multimodal Transformer
//!
//! Main transformer implementation that orchestrates multimodal processing
//! across different input modalities with configurable fusion strategies.

use std::collections::HashMap;
use crate::error::{NNError, Result};
use crate::attention::MultiHeadAttention;
use crate::linear::Linear;
use crate::layernorm::LayerNorm;
use crate::activation::GeLU;
use crate::functional::linear;
use crate::module::ModuleExt;
use tensor::Tensor;
use backend::Backend;
use storage::{Storage, StorageFromVec, StorageToDense};
use autograd::ops::mean;
use tensor::ops::concatenate_tensors;
use dtype::DataType;
use dtype::traits::FloatExt;
use super::modality::{Modality, ModalityConfig};
use super::attention::CrossModalAttention;
use super::fusion::{FusionStrategy, Fusion, FusionBlock, FeedForward};
use super::encoder::Encoder;
use super::task::Task;

/// Multimodal transformer for joint processing of multiple modalities
///
/// This is the main component of the multimodal system, capable of processing
/// multiple input modalities (vision, language, audio, etc.) simultaneously
/// and producing task-specific outputs through cross-modal understanding.
///
/// ## Architecture
///
/// The transformer follows a three-stage pipeline:
/// 1. **Encoding**: Each input modality is processed by a specialized encoder
/// 2. **Fusion**: Encoded representations interact through attention
/// 3. **Task Output**: Fused representations are processed by task outputs
///
/// ## Generic Parameters
///
/// - `B`: Backend type (e.g., `CpuBackend<Float32>`)
/// - `S`: Storage type (e.g., `DenseStorage<Float32>`)
/// - `T`: Data type (e.g., `Float32`)
///
/// ## Example
///
/// ```rust,ignore
/// // Create transformer for vision-language tasks
/// let config = MultimodalConfig {
///     modalities: vec![Modality::Vision, Modality::Language],
///     hidden_dim: 768,
///     num_fusion_layers: 6,
///     fusion_strategy: FusionStrategy::HierarchicalFusion,
///     dropout: 0.1,
/// };
///
/// let transformer = MultimodalTransformer::<CpuBackend<Float32>, _, Float32>::new(config)?;
/// ```
#[derive(Debug)]
pub struct MultimodalTransformer<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + storage::StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + dtype::FloatExt + num_traits::FromPrimitive + num_traits::Bounded,
{
    /// Specialized encoders for each modality
    /// Maps modality types to their corresponding encoders
    pub encoders: HashMap<Modality, Encoder<B, S, T>>,
    /// Stack of fusion layers for inter-modality interaction
    pub fusion_layers: Vec<FusionBlock<B, S, T>>,
    /// Fusion layer for combining different fusion strategies
    pub fusion: Option<Fusion<B, S, T>>,
    /// Task-specific outputs for different downstream tasks
    pub tasks: HashMap<String, Task<B, S, T>>,
    /// Global configuration defining the multimodal architecture
    pub config: MultimodalConfig,
}

/// Global configuration for the multimodal transformer architecture
///
/// This struct defines the overall architecture of the multimodal system,
/// including which modalities to support, fusion strategy, and shared parameters.
#[derive(Debug)]
pub struct MultimodalConfig {
    /// List of modalities that this transformer will support
    /// Each modality will get its own encoder and participate in cross-modal fusion
    pub modalities: Vec<Modality>,
    /// Shared hidden dimension used across all modalities and fusion layers
    /// Must be consistent for cross-modal attention to work properly
    pub hidden_dim: usize,
    /// Number of cross-modal fusion layers to apply
    /// More layers allow deeper cross-modal interaction but increase computation
    pub num_fusion_layers: usize,
    /// Strategy for fusing information across modalities
    /// See [`FusionStrategy`] for available options and their trade-offs
    pub fusion_strategy: FusionStrategy,
    /// Global dropout probability applied throughout the network
    pub dropout: f64,
}

impl Default for MultimodalConfig {
    fn default() -> Self {
        Self {
            modalities: vec![Modality::Vision, Modality::Language],
            hidden_dim: 768,
            num_fusion_layers: 6,
            fusion_strategy: FusionStrategy::HierarchicalFusion,
            dropout: 0.1,
        }
    }
}

impl<B, S, T> MultimodalTransformer<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::FromPrimitive + num_traits::Bounded,
{
    /// Create new multimodal transformer
    pub fn new(config: MultimodalConfig) -> Result<Self> {
        let mut encoders = HashMap::new();
        let mut fusion_layers = Vec::new();

        // Create modality-specific encoders
        for modality in &config.modalities {
            let modality_config = Self::create_modality_config(modality, &config);
            let encoder = Encoder::new(modality_config)?;
            encoders.insert(modality.clone(), encoder);
        }

        // Create fusion layers
        for _ in 0..config.num_fusion_layers {
            let fusion_block = Self::create_fusion_block(&config)?;
            fusion_layers.push(fusion_block);
        }

        // Create fusion layer
        let fusion = Some(Fusion::new(config.hidden_dim, config.fusion_strategy.clone())?);

        Ok(Self {
            encoders,
            fusion_layers,
            fusion,
            tasks: HashMap::new(),
            config,
        })
    }

    /// Create modality configuration for a specific modality
    fn create_modality_config(modality: &Modality, config: &MultimodalConfig) -> ModalityConfig {
        ModalityConfig {
            modality: modality.clone(),
            input_dim: match modality {
                Modality::Vision => 2048,    // CLIP vision features
                Modality::Language => 768,   // BERT embeddings
                Modality::Audio => 1024,     // Audio features
                Modality::Custom(_) => config.hidden_dim,
            },
            hidden_dim: config.hidden_dim,
            num_layers: 6,
            num_heads: 12,
            max_seq_len: 512,
            dropout: config.dropout,
            params: HashMap::new(),
        }
    }

    /// Create cross-modal attention mechanisms based on fusion strategy
    fn create_cross_attention_for_strategy(
        config: &MultimodalConfig,
        cross_attention: &mut Vec<CrossModalAttention<B, S, T>>,
    ) -> Result<()> {
        match config.fusion_strategy {
            FusionStrategy::HierarchicalFusion => {
                Self::create_hierarchical_attention(config, cross_attention)
            },
            FusionStrategy::AttentionFusion | FusionStrategy::CrossModalFusion => {
                Self::create_all_to_all_attention(config, cross_attention)
            },
            _ => Ok(()), // Early/Late fusion handled elsewhere
        }
    }

    /// Create bidirectional cross-attention for hierarchical fusion
    fn create_hierarchical_attention(
        config: &MultimodalConfig,
        cross_attention: &mut Vec<CrossModalAttention<B, S, T>>,
    ) -> Result<()> {
        // Bidirectional cross-attention between all modality pairs
        for i in 0..config.modalities.len() {
            for j in 0..config.modalities.len() {
                if i != j {
                    let cross_attn = CrossModalAttention::new(
                        config.hidden_dim,
                        12,
                        config.modalities[i].clone(),
                        vec![config.modalities[j].clone()],
                    )?;
                    cross_attention.push(cross_attn);
                }
            }
        }
        Ok(())
    }

    /// Create all-to-all cross-modal attention
    fn create_all_to_all_attention(
        config: &MultimodalConfig,
        cross_attention: &mut Vec<CrossModalAttention<B, S, T>>,
    ) -> Result<()> {
        // All-to-all cross-modal attention
        for (i, query_mod) in config.modalities.iter().enumerate() {
            let mut kv_mods = config.modalities.clone();
            kv_mods.remove(i); // Remove self
            let cross_attn = CrossModalAttention::new(
                config.hidden_dim,
                12,
                query_mod.clone(),
                kv_mods,
            )?;
            cross_attention.push(cross_attn);
        }
        Ok(())
    }

    /// Create a single fusion layer based on strategy
    fn create_fusion_block(config: &MultimodalConfig) -> Result<FusionBlock<B, S, T>> {
        let mut intra_attention = HashMap::new();
        let mut cross_attention = Vec::new();
        let mut feed_forward = HashMap::new();
        let mut norms = HashMap::new();

        // Create intra-modal attention for each modality
        for modality in &config.modalities {
            intra_attention.insert(
                modality.clone(),
                MultiHeadAttention::new(config.hidden_dim, 12)?
            );

            feed_forward.insert(
                modality.clone(),
                FeedForward {
                    linear1: Linear::new(config.hidden_dim, config.hidden_dim * 4)?,
                    linear2: Linear::new(config.hidden_dim * 4, config.hidden_dim)?,
                    gelu: GeLU::new(),
                    dropout: config.dropout,
                }
            );
        }

        // Create cross-modal attention based on fusion strategy
        Self::create_cross_attention_for_strategy(config, &mut cross_attention)?;

        // Create layer norms
        for modality in &config.modalities {
            norms.insert(
                format!("{}_intra", modality.as_str()),
                LayerNorm::new(vec![config.hidden_dim], 1e-6)
            );
            norms.insert(
                format!("{}_cross", modality.as_str()),
                LayerNorm::new(vec![config.hidden_dim], 1e-6)
            );
            norms.insert(
                format!("{}_ff", modality.as_str()),
                LayerNorm::new(vec![config.hidden_dim], 1e-6)
            );
        }

        Ok(FusionBlock {
            intra_attention,
            cross_attention,
            feed_forward,
            norms,
        })
    }

    /// Add a task-specific output
    pub fn add_task(&mut self, task_name: String, task: Task<B, S, T>) -> Result<()> {
        self.tasks.insert(task_name, task);
        Ok(())
    }

    /// Forward pass for multimodal inputs
    pub fn forward(
        &self,
        inputs: &HashMap<Modality, Tensor<B, S, T>>,
        task: &str,
        mask: Option<&Tensor<B, S, T>>,
    ) -> Result<Tensor<B, S, T>> {
        // Step 1: Encode individual modalities
        let mut modality_embeddings = HashMap::new();
        for (modality, input_tensor) in inputs {
            if let Some(encoder) = self.encoders.get(modality) {
                let embedding = encoder.forward(input_tensor, mask)?;
                modality_embeddings.insert(modality.clone(), embedding);
            }
        }

        if modality_embeddings.is_empty() {
            return Err(NNError::InvalidInput {
                message: "No valid modality inputs provided".into()
            });
        }

        // Step 2: Apply cross-modal fusion
        let mut fused_embeddings = modality_embeddings;
        for fusion_layer in &self.fusion_layers {
            fused_embeddings = fusion_layer.forward(&fused_embeddings, mask)?;
        }

        // Step 3: Apply task-specific output
        if let Some(task_output) = self.tasks.get(task) {
            Ok(match task_output {
                Task::Classification(head) => {
                    // For classification, combine all modality representations
                    let combined_embedding = self.combine_modality_embeddings(&fused_embeddings)?;
                    // Global average pooling across sequence dimension
                    let pooled = mean(&combined_embedding, Some(&[1]), false)?;
                    linear(&pooled, &head.classifier.weight.data, Some(&head.classifier.bias.data))?
                },
                Task::Regression(head) => {
                    // For regression, combine all modality representations
                    let combined_embedding = self.combine_modality_embeddings(&fused_embeddings)?;
                    let pooled = mean(&combined_embedding, Some(&[1]), false)?;
                    linear(&pooled, &head.weight.data, Some(&head.bias.data))?
                },
                Task::Generation(head) => {
                    // For generation, use the primary modality (language if available, otherwise first)
                    let primary_embedding = self.select_primary_modality(&fused_embeddings)?;
                    linear(&primary_embedding, &head.lm_head.weight.data, Some(&head.lm_head.bias.data))?
                },
                Task::Retrieval(head) => {
                    // For retrieval, combine all modality representations
                    let combined_embedding = self.combine_modality_embeddings(&fused_embeddings)?;
                    let pooled = mean(&combined_embedding, Some(&[1]), false)?;
                    linear(&pooled, &head.projection.weight.data, Some(&head.projection.bias.data))?
                },
            })
        } else {
            Err(NNError::InvalidConfiguration { message: format!("Task head '{}' not found", task) })
        }
    }

    /// Get the number of parameters in the model
    pub fn num_parameters(&self) -> usize {
        let mut total = 0usize;

        // Count parameters in encoders
        for encoder in self.encoders.values() {
            total += encoder.num_parameters();
        }

        // Count parameters in fusion layers
        for layer in &self.fusion_layers {
            for attention in layer.intra_attention.values() {
                total += attention.num_parameters();
            }
            for cross_attn in &layer.cross_attention {
                total += cross_attn.num_parameters();
            }
            for ff in layer.feed_forward.values() {
                total += ff.num_parameters();
            }
            for norm in layer.norms.values() {
                total += norm.num_parameters();
            }
        }

        // Count parameters in task outputs
        for task in self.tasks.values() {
            total += task.num_parameters();
        }

        total
    }

    /// Combine embeddings from multiple modalities into a single representation
    fn combine_modality_embeddings(&self, embeddings: &HashMap<Modality, Tensor<B, S, T>>) -> Result<Tensor<B, S, T>> {
        if embeddings.is_empty() {
            return Err(NNError::InvalidInput {
                message: "No modality embeddings to combine".into()
            });
        }

        if embeddings.len() == 1 {
            // Single modality - return as is
            return Ok(embeddings.values().next().unwrap().clone());
        }

        // Multiple modalities - use configured fusion strategy
        match self.config.fusion_strategy {
            FusionStrategy::EarlyFusion | FusionStrategy::HierarchicalFusion => {
                // Concatenate along the hidden dimension
                let mut embedding_list = Vec::new();
                for embedding in embeddings.values() {
                    embedding_list.push(embedding.clone());
                }
                Ok(concatenate_tensors(&embedding_list, 2)?) // Concat along hidden dim
            },
            FusionStrategy::LateFusion | FusionStrategy::AttentionFusion => {
                // Average pooling across modalities
                let mut combined = embeddings.values().next().unwrap().clone();
                for embedding in embeddings.values().skip(1) {
                    combined = tensor::ops::arithmetic::add(&combined, embedding)?;
                }
                // Average by dividing by number of modalities
                let scale = num_traits::cast(1.0 / embeddings.len() as f64).unwrap();
                Ok(tensor::ops::arithmetic::scalar_mul(&combined, scale)?)
            },
            FusionStrategy::CrossModalFusion => {
                // Use the first embedding as primary representation
                Ok(embeddings.values().next().unwrap().clone())
            },
        }
    }

    /// Select the primary modality for generation tasks
    fn select_primary_modality(&self, embeddings: &HashMap<Modality, Tensor<B, S, T>>) -> Result<Tensor<B, S, T>> {
        // Prefer language modality for generation, otherwise use first available
        if let Some(language_embedding) = embeddings.get(&Modality::Language) {
            Ok(language_embedding.clone())
        } else {
            embeddings.values().next()
                .ok_or_else(|| NNError::InvalidInput {
                    message: "No modality embeddings available".into()
                })
                .map(|tensor| tensor.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use storage::DenseStorage;
    use dtype::float::Float32;

    #[test]
    fn test_multimodal_config_default() {
        let config = MultimodalConfig::default();
        assert_eq!(config.modalities, vec![Modality::Vision, Modality::Language]);
        assert_eq!(config.hidden_dim, 768);
        assert_eq!(config.num_fusion_layers, 6);
        assert_eq!(config.dropout, 0.1);
    }

    #[test]
    fn test_multimodal_transformer_creation() {
        let config = MultimodalConfig {
            modalities: vec![Modality::Vision, Modality::Language],
            hidden_dim: 768,
            num_fusion_layers: 6,
            fusion_strategy: FusionStrategy::HierarchicalFusion,
            dropout: 0.1,
        };

        let transformer = MultimodalTransformer::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config).unwrap();
        assert_eq!(transformer.config.modalities.len(), 2);
        assert_eq!(transformer.config.hidden_dim, 768);
        assert_eq!(transformer.fusion_layers.len(), 6);
        assert!(transformer.encoders.contains_key(&Modality::Vision));
        assert!(transformer.encoders.contains_key(&Modality::Language));
    }

    #[test]
    fn test_combine_modality_embeddings_single() {
        let config = MultimodalConfig {
            modalities: vec![Modality::Vision],
            hidden_dim: 768,
            num_fusion_layers: 6,
            fusion_strategy: FusionStrategy::HierarchicalFusion,
            dropout: 0.1,
        };

        let transformer = MultimodalTransformer::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config).unwrap();

        // Create test embeddings
        let mut embeddings = HashMap::new();
        let vision_tensor = Tensor::from_vec(vec![Float32::new(1.0); 768], &[1, 10, 768]).unwrap();
        embeddings.insert(Modality::Vision, vision_tensor);

        let combined = transformer.combine_modality_embeddings(&embeddings).unwrap();
        assert_eq!(combined.shape().dims(), &[1, 10, 768]);
    }

    #[test]
    fn test_select_primary_modality_language() {
        let config = MultimodalConfig {
            modalities: vec![Modality::Vision, Modality::Language],
            hidden_dim: 768,
            num_fusion_layers: 6,
            fusion_strategy: FusionStrategy::HierarchicalFusion,
            dropout: 0.1,
        };

        let transformer = MultimodalTransformer::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config).unwrap();

        // Create test embeddings
        let mut embeddings = HashMap::new();
        let vision_tensor = Tensor::from_vec(vec![Float32::new(1.0); 768], &[1, 10, 768]).unwrap();
        let language_tensor = Tensor::from_vec(vec![Float32::new(2.0); 768], &[1, 10, 768]).unwrap();
        embeddings.insert(Modality::Vision, vision_tensor);
        embeddings.insert(Modality::Language, language_tensor);

        let primary = transformer.select_primary_modality(&embeddings).unwrap();
        // Should prefer language modality
        assert_eq!(primary.shape().dims(), &[1, 10, 768]);
        // Check that it's the language tensor (sum should be 2.0 * 768 * 10)
        let sum: f32 = primary.as_slice().iter().map(|x| x.get()).sum();
        assert_eq!(sum, 2.0 * 768.0 * 10.0);
    }

    #[test]
    fn test_select_primary_modality_fallback() {
        let config = MultimodalConfig {
            modalities: vec![Modality::Vision],
            hidden_dim: 768,
            num_fusion_layers: 6,
            fusion_strategy: FusionStrategy::HierarchicalFusion,
            dropout: 0.1,
        };

        let transformer = MultimodalTransformer::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config).unwrap();

        // Create test embeddings with only vision
        let mut embeddings = HashMap::new();
        let vision_tensor = Tensor::from_vec(vec![Float32::new(1.0); 768], &[1, 10, 768]).unwrap();
        embeddings.insert(Modality::Vision, vision_tensor);

        let primary = transformer.select_primary_modality(&embeddings).unwrap();
        // Should return the only available modality
        assert_eq!(primary.shape().dims(), &[1, 10, 768]);
    }
}
