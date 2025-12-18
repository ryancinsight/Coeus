//! # Multimodal Fusion Strategies
//!
//! Implementation of various strategies for fusing information across multiple modalities.
//! Provides both simple concatenation and complex cross-modal transformer approaches.

use std::collections::HashMap;
use crate::error::Result;
use crate::attention::MultiHeadAttention;
use crate::linear::Linear;
use crate::layernorm::LayerNorm;
use crate::activation::GeLU;
use crate::dropout::Dropout;
use crate::functional::linear;
use crate::module::{Module, ModuleExt};
use tensor::Tensor;
use backend::Backend;
use storage::{Storage, StorageFromVec, StorageToDense};
use dtype::DataType;
use super::modality::Modality;
use super::attention::CrossModalAttention;

/// Strategy for fusing information across multiple modalities
///
/// Different fusion strategies offer trade-offs between computational complexity,
/// representational capacity, and cross-modal interaction depth.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionStrategy {
    /// Early fusion: concatenate all modality inputs before any processing
    /// - Pros: Simple, preserves all cross-modal interactions
    /// - Cons: High dimensionality, may not scale to many modalities
    EarlyFusion,
    /// Late fusion: process each modality separately, then combine outputs
    /// - Pros: Modular, allows modality-specific optimization
    /// - Cons: Limited cross-modal interaction
    LateFusion,
    /// Hierarchical fusion: progressive fusion with increasing interaction depth
    /// - Pros: Balances interaction and complexity, allows hierarchical understanding
    /// - Cons: More complex to implement and tune
    HierarchicalFusion,
    /// Attention-based fusion: learnable attention weights for modality combination
    /// - Pros: Adaptive weighting, handles variable modality importance
    /// - Cons: Requires learning attention parameters
    AttentionFusion,
    /// Cross-modal fusion: dedicated transformer layers for cross-modal interaction
    /// - Pros: Rich cross-modal understanding, scalable
    /// - Cons: Highest computational cost
    CrossModalFusion,
}

/// Fusion layer
#[derive(Debug)]
pub struct Fusion<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + storage::StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::FromPrimitive + num_traits::Bounded,
{
    /// Fusion strategy
    pub strategy: FusionStrategy,
    /// Fusion layers
    pub fusion_layers: Vec<FusionLayer<B, S, T>>,
    /// Output dimension
    pub output_dim: usize,
}

impl<B, S, T> Fusion<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + storage::StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::FromPrimitive + num_traits::Bounded,
{
    /// Create new multimodal fusion layer
    pub fn new(output_dim: usize, strategy: FusionStrategy) -> Result<Self> {
        let fusion_layers = Vec::new(); // Initialize empty fusion layers
        Ok(Self {
            strategy,
            fusion_layers,
            output_dim,
        })
    }

    /// Add a fusion layer to the fusion pipeline
    pub fn add_fusion_layer(&mut self, layer: FusionLayer<B, S, T>) -> Result<()> {
        self.fusion_layers.push(layer);
        Ok(())
    }
}

#[derive(Debug)]
pub enum FusionLayer<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::FromPrimitive + num_traits::Bounded,
{
    /// Simple concatenation
    Concat(Linear<B, S, T>),
    /// Attention-based fusion
    Attention(MultiHeadAttention<B, S, T>, Linear<B, S, T>),
    /// Cross-modal transformer block
    CrossTransformer(FusionBlock<B, S, T>),
    /// Adaptive fusion with learned weights
    AdaptiveFusion(Vec<Linear<B, S, T>>),
}

/// Fusion block for hierarchical fusion
#[derive(Debug)]
pub struct FusionBlock<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::FromPrimitive + num_traits::Bounded + std::cmp::PartialOrd,
{
    /// Self-attention within each modality
    pub intra_attention: HashMap<Modality, MultiHeadAttention<B, S, T>>,
    /// Cross-modal attention between modalities
    pub cross_attention: Vec<CrossModalAttention<B, S, T>>,
    /// Feed-forward networks
    pub feed_forward: HashMap<Modality, FeedForward<B, S, T>>,
    /// Layer norms
    pub norms: HashMap<String, LayerNorm<B, S, T>>,
}

impl<B, S, T> FusionBlock<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::FromPrimitive + num_traits::Bounded + std::cmp::PartialOrd,
{
    /// Forward pass through cross-modal transformer block
    pub fn forward(
        &self,
        modality_embeddings: &HashMap<Modality, Tensor<B, S, T>>,
        mask: Option<&Tensor<B, S, T>>,
    ) -> Result<HashMap<Modality, Tensor<B, S, T>>> {
        let mut updated_embeddings = HashMap::new();

        // Step 1: Intra-modal self-attention for each modality
        for (modality, embedding) in modality_embeddings {
            if let Some(attn) = self.intra_attention.get(modality) {
                let norm_key = format!("{}_intra", modality.as_str());
                if let Some(norm) = self.norms.get(&norm_key) {
                    let norm_out = norm.forward(embedding)?;
                    let attn_out = attn.forward_cross_attention(&norm_out, &norm_out, &norm_out)?;
                    let residual = embedding + &attn_out;
                    updated_embeddings.insert(modality.clone(), residual);
                } else {
                    // Fallback without layer norm
                    updated_embeddings.insert(modality.clone(), embedding.clone());
                }
            } else {
                // No intra attention for this modality - pass through
                updated_embeddings.insert(modality.clone(), embedding.clone());
            }
        }

        // Step 2: Cross-modal attention
        let mut cross_modal_outputs = HashMap::new();

        // Create key/value maps for cross-attention
        let mut all_keys = HashMap::new();
        let mut all_values = HashMap::new();

        for (modality, embedding) in &updated_embeddings {
            all_keys.insert(modality.clone(), embedding.clone());
            all_values.insert(modality.clone(), embedding.clone());
        }

        // Apply each cross-attention mechanism
        for cross_attn in &self.cross_attention {
            if let Some(query_embedding) = updated_embeddings.get(&cross_attn.query_modality) {
                let attended = cross_attn.forward(query_embedding, &all_keys, &all_values, mask)?;

                // Store updated embedding for this query modality
                cross_modal_outputs.insert(cross_attn.query_modality.clone(), attended);
            }
        }

        // For modalities that weren't updated by cross-attention, keep original
        for (modality, embedding) in &updated_embeddings {
            if !cross_modal_outputs.contains_key(modality) {
                cross_modal_outputs.insert(modality.clone(), embedding.clone());
            }
        }

        // Step 3: Feed-forward networks for each modality
        let mut final_outputs = HashMap::new();

        for (modality, embedding) in &cross_modal_outputs {
            if let Some(ff) = self.feed_forward.get(modality) {
                let norm_key = format!("{}_ff", modality.as_str());
                if let Some(norm) = self.norms.get(&norm_key) {
                    let norm_out = norm.forward(embedding)?;
                    let ff_out = ff.forward(&norm_out)?;
                    let residual = embedding + &ff_out;
                    final_outputs.insert(modality.clone(), residual);
                } else {
                    // Fallback without layer norm
                    final_outputs.insert(modality.clone(), embedding.clone());
                }
            } else {
                final_outputs.insert(modality.clone(), embedding.clone());
            }
        }

        Ok(final_outputs)
    }
}

#[derive(Debug)]
pub struct FeedForward<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + storage::StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::FromPrimitive + num_traits::Bounded,
{
    pub linear1: Linear<B, S, T>,
    pub linear2: Linear<B, S, T>,
    pub gelu: GeLU<B, S, T>,
    pub dropout: f64,
}

impl<B, S, T> FeedForward<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::FromPrimitive + num_traits::Bounded,
{
    /// Create new feed-forward network
    pub fn new(hidden_dim: usize, ff_dim: usize, dropout: f64) -> Result<Self> {
        let linear1 = Linear::new(hidden_dim, ff_dim)?;
        let linear2 = Linear::new(ff_dim, hidden_dim)?;
        let gelu = GeLU::new();
        let dropout = dropout;

        Ok(Self {
            linear1,
            linear2,
            gelu,
            dropout,
        })
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor<B, S, T>) -> Result<Tensor<B, S, T>> {
        // Linear transformation to higher dimension
        let hidden = linear(input, &self.linear1.weight.data, Some(&self.linear1.bias.data))?;

        // Apply GELU activation
        let activated = self.gelu.forward(&hidden)?;

        // TODO: Apply dropout when tensor dropout method is available
        // For now, skip dropout
        let dropped = activated;

        // Linear transformation back to hidden dimension
        linear(&dropped, &self.linear2.weight.data, Some(&self.linear2.bias.data))
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.linear1.num_parameters() + self.linear2.num_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use backend::CpuBackend;
    use storage::DenseStorage;
    use dtype::float::Float32;

    #[test]
    fn test_feed_forward_creation() {
        let ff = FeedForward::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(768, 3072, 0.1).unwrap();
        assert_eq!(ff.dropout, 0.1);
    }

    #[test]
    fn test_fusion_strategy_enum() {
        let strategies = vec![
            FusionStrategy::EarlyFusion,
            FusionStrategy::LateFusion,
            FusionStrategy::HierarchicalFusion,
            FusionStrategy::AttentionFusion,
            FusionStrategy::CrossModalFusion,
        ];

        assert_eq!(strategies.len(), 5);
    }

    #[test]
    fn test_fusion_creation() {
        let fusion = Fusion::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(768, FusionStrategy::HierarchicalFusion).unwrap();
        assert_eq!(fusion.strategy, FusionStrategy::HierarchicalFusion);
        assert_eq!(fusion.output_dim, 768);
        assert_eq!(fusion.fusion_layers.len(), 0); // Starts empty
    }
}
