//! Cross-Modal Attention Mechanisms (Sprint MS-47)
//!
//! This module implements advanced attention mechanisms that enable interaction
//! between different modalities (vision, language, audio) for joint understanding.

use crate::activation::GeLU;
use crate::attention::MultiHeadAttention;
use crate::error::Result;
use crate::layernorm::LayerNorm;
use crate::linear::Linear;
use crate::multimodal::Modality;
use backend::Backend;
use dtype::DataType;
use dtype::FloatExt;
use std::collections::HashMap;
use storage::{Storage, StorageFromVec, StorageToDense};

/// Types of cross-modal attention patterns
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionPattern {
    /// Vision → Language (image-conditioned text generation)
    VisionToLanguage,
    /// Language → Vision (text-conditioned image generation)
    LanguageToVision,
    /// Audio → Language (speech-conditioned text generation)
    AudioToLanguage,
    /// Language → Audio (text-conditioned speech generation)
    LanguageToAudio,
    /// Vision → Audio (video-conditioned audio generation)
    VisionToAudio,
    /// Audio → Vision (audio-conditioned video generation)
    AudioToVision,
    /// Bidirectional (vision ↔ language)
    BidirectionalVL,
    /// Bidirectional (audio ↔ language)
    BidirectionalAL,
    /// Tri-modal (vision ↔ language ↔ audio)
    Trimodal,
}

/// Cross-modal transformer layer combining intra and inter-modal attention
#[derive(Debug)]
pub struct CrossModalTransformerLayer<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::Bounded + num_traits::FromPrimitive,
{
    /// Intra-modal self-attention for each modality
    pub intra_attention: HashMap<Modality, MultiHeadAttention<B, S, T>>,
    /// Cross-modal attention mechanisms
    pub cross_attention: Vec<CrossModalAttention<B, S, T>>,
    /// Feed-forward networks per modality
    pub feed_forward: HashMap<Modality, FeedForwardNetwork<B, S, T>>,
    /// Layer norms
    pub layer_norms: HashMap<String, LayerNorm<B, S, T>>,
    /// Attention pattern for this layer
    pub attention_pattern: AttentionPattern,
    /// Dropout probability
    pub dropout: f64,
}

#[derive(Debug)]
pub struct CrossModalAttention<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::Bounded + num_traits::FromPrimitive,
{
    /// Query modality
    pub query_modality: Modality,
    /// Key/Value modalities (can be multiple for fusion)
    pub kv_modalities: Vec<Modality>,
    /// Attention mechanism
    pub attention: MultiHeadAttention<B, S, T>,
    /// Query projection (optional for cross-domain projection)
    pub query_proj: Option<Linear<B, S, T>>,
    /// Key projection
    pub key_proj: Option<Linear<B, S, T>>,
    /// Value projection
    pub value_proj: Option<Linear<B, S, T>>,
    /// Output projection
    pub out_proj: Linear<B, S, T>,
    /// Layer normalization
    pub norm: LayerNorm<B, S, T>,
    /// Cross-attention type
    pub attention_type: CrossAttentionType,
}

#[derive(Debug, Clone)]
pub enum CrossAttentionType {
    /// Direct cross-attention between modalities
    Direct,
    /// Gated cross-attention with learnable gates
    Gated,
    /// Multi-head cross-attention with modality-specific heads
    MultiHeadGated,
    /// Hierarchical cross-attention with multiple levels
    Hierarchical,
}

#[derive(Debug)]
pub struct FeedForwardNetwork<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType + FloatExt + 'static,
{
    pub linear1: Linear<B, S, T>,
    pub linear2: Linear<B, S, T>,
    pub activation: GeLU<B, S, T>,
    pub dropout: f64,
}

/// Co-Attention mechanism for vision-language tasks
#[derive(Debug)]
pub struct CoAttention<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::Bounded + num_traits::FromPrimitive,
{
    /// Vision encoder dimension
    pub vision_dim: usize,
    /// Language encoder dimension
    pub text_dim: usize,
    /// Hidden dimension for attention
    pub hidden_dim: usize,
    /// Whether to use symmetric co-attention
    pub symmetric: bool,
    /// Vision-to-text attention
    pub v2t_attention: MultiHeadAttention<B, S, T>,
    /// Text-to-vision attention
    pub t2v_attention: MultiHeadAttention<B, S, T>,
    /// Fusion layer
    pub fusion: Linear<B, S, T>,
}

/// Multi-Modal Fusion Attention (MFA)
/// Learns attention weights across multiple modalities dynamically
#[derive(Debug)]
pub struct MultimodalFusionAttention<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::Bounded + num_traits::FromPrimitive,
{
    /// Number of modalities
    pub num_modalities: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads per modality
    pub num_heads: usize,
    /// Modality-specific projections
    pub modality_projections: HashMap<Modality, Linear<B, S, T>>,
    /// Multi-head attention for fusion
    pub fusion_attention: MultiHeadAttention<B, S, T>,
    /// Learnable modality importance weights
    pub modality_weights: Vec<f64>,
    /// Temperature for softmax weighting
    pub temperature: f64,
}

/// Hierarchical Cross-Attention for complex multimodal tasks
#[derive(Debug)]
pub struct HierarchicalCrossAttention<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::Bounded + num_traits::FromPrimitive,
{
    /// Global attention across all modalities
    pub global_attention: MultiHeadAttention<B, S, T>,
    /// Pairwise attention between modality pairs
    pub pairwise_attention: HashMap<(Modality, Modality), MultiHeadAttention<B, S, T>>,
    /// Local attention within each modality
    pub local_attention: HashMap<Modality, MultiHeadAttention<B, S, T>>,
    /// Hierarchical levels (coarse to fine)
    pub num_levels: usize,
    /// Downsampling projections for hierarchical processing
    pub downsample_proj: HashMap<Modality, Vec<Linear<B, S, T>>>,
}

/// Attention-based Modality Selector
/// Dynamically selects which modalities to attend to based on task and input
#[derive(Debug)]
pub struct ModalitySelector<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::Bounded + num_traits::FromPrimitive,
{
    /// Input dimension for each modality
    pub input_dims: HashMap<Modality, usize>,
    /// Selection network (produces attention weights over modalities)
    pub selector_network: MultiHeadAttention<B, S, T>,
    /// Modality embeddings for selection
    pub modality_embeddings: HashMap<Modality, Vec<f32>>,
    /// Selection threshold
    pub selection_threshold: f32,
    /// Whether to allow hard selection (0/1) or soft weights
    pub hard_selection: bool,
}

/// Progressive Cross-Modal Integration (PCMI)
/// Gradually integrates modalities through multiple stages
#[derive(Debug)]
pub struct ProgressiveCrossModalIntegration<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::Bounded + num_traits::FromPrimitive,
{
    /// Number of integration stages
    pub num_stages: usize,
    /// Integration blocks for each stage
    pub integration_blocks: Vec<IntegrationBlock<B, S, T>>,
    /// Modality ordering (which modality gets integrated when)
    pub integration_order: Vec<Modality>,
}

#[derive(Debug)]
pub struct IntegrationBlock<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::Bounded + num_traits::FromPrimitive,
{
    /// Modalities to integrate at this stage
    pub modalities: Vec<Modality>,
    /// Cross-attention layers for this stage
    pub cross_attention: Vec<CrossModalAttention<B, S, T>>,
    /// Fusion mechanism for this stage
    pub fusion: FusionMechanism<B, S, T>,
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum FusionMechanism<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + StorageToDense<T> + 'static,
    T: DataType,
{
    /// Simple concatenation
    Concatenation,
    /// Learnable weighted sum
    WeightedSum,
    /// Attention-based fusion
    AttentionFusion(MultiHeadAttention<B, S, T>),
    /// Gated fusion with learned gates
    GatedFusion,
}

impl<B, S, T> CrossModalTransformerLayer<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::FromPrimitive + num_traits::Bounded,
{
    /// Create new cross-modal transformer layer
    pub fn new(
        modalities: &[Modality],
        hidden_dim: usize,
        num_heads: usize,
        attention_pattern: AttentionPattern,
    ) -> Result<Self> {
        let mut intra_attention = HashMap::new();
        let mut feed_forward = HashMap::new();

        // Create intra-modal attention and FFN for each modality
        for modality in modalities {
            intra_attention.insert(
                modality.clone(),
                MultiHeadAttention::new(hidden_dim, num_heads)?,
            );
            feed_forward.insert(
                modality.clone(),
                FeedForwardNetwork::new(hidden_dim, hidden_dim * 4)?,
            );
        }

        // Create cross-modal attention based on pattern
        let cross_attention = Self::create_cross_attention_layers(
            modalities,
            hidden_dim,
            num_heads,
            &attention_pattern,
        )?;

        // Create layer norms
        let mut layer_norms = HashMap::new();
        layer_norms.insert(
            "intra_norm".to_string(),
            LayerNorm::new(vec![hidden_dim], 1e-6),
        );
        layer_norms.insert(
            "cross_norm".to_string(),
            LayerNorm::new(vec![hidden_dim], 1e-6),
        );
        layer_norms.insert(
            "ff_norm".to_string(),
            LayerNorm::new(vec![hidden_dim], 1e-6),
        );

        Ok(Self {
            intra_attention,
            cross_attention,
            feed_forward,
            layer_norms,
            attention_pattern,
            dropout: 0.1,
        })
    }

    /// Create cross-attention layers based on attention pattern
    fn create_cross_attention_layers(
        modalities: &[Modality],
        hidden_dim: usize,
        num_heads: usize,
        pattern: &AttentionPattern,
    ) -> Result<Vec<CrossModalAttention<B, S, T>>> {
        let mut cross_attention = Vec::new();

        match pattern {
            AttentionPattern::VisionToLanguage => {
                if modalities.contains(&Modality::Vision)
                    && modalities.contains(&Modality::Language)
                {
                    cross_attention.push(CrossModalAttention::new(
                        hidden_dim,
                        num_heads,
                        Modality::Language,
                        vec![Modality::Vision],
                    )?);
                }
            }
            AttentionPattern::LanguageToVision => {
                if modalities.contains(&Modality::Vision)
                    && modalities.contains(&Modality::Language)
                {
                    cross_attention.push(CrossModalAttention::new(
                        hidden_dim,
                        num_heads,
                        Modality::Vision,
                        vec![Modality::Language],
                    )?);
                }
            }
            AttentionPattern::BidirectionalVL => {
                if modalities.contains(&Modality::Vision)
                    && modalities.contains(&Modality::Language)
                {
                    // V->L attention
                    cross_attention.push(CrossModalAttention::new(
                        hidden_dim,
                        num_heads,
                        Modality::Language,
                        vec![Modality::Vision],
                    )?);
                    // L->V attention
                    cross_attention.push(CrossModalAttention::new(
                        hidden_dim,
                        num_heads,
                        Modality::Vision,
                        vec![Modality::Language],
                    )?);
                }
            }
            AttentionPattern::Trimodal => {
                // Create pairwise cross-attention for all modality combinations
                let modal_pairs = vec![
                    (Modality::Vision, Modality::Language),
                    (Modality::Vision, Modality::Audio),
                    (Modality::Language, Modality::Audio),
                ];

                for (query_mod, kv_mod) in modal_pairs {
                    if modalities.contains(&query_mod) && modalities.contains(&kv_mod) {
                        cross_attention.push(CrossModalAttention::new(
                            hidden_dim,
                            num_heads,
                            query_mod,
                            vec![kv_mod],
                        )?);
                    }
                }
            }
            _ => {
                // Default: create pairwise cross-attention for adjacent modalities
                // (This would need to be expanded based on the specific pattern)
            }
        }

        Ok(cross_attention)
    }

    /// Forward pass through the cross-modal transformer layer
    pub fn forward(
        &self,
        modality_embeddings: &HashMap<Modality, Vec<f32>>,
        batch_size: usize,
    ) -> Result<HashMap<Modality, Vec<f32>>> {
        let mut updated_embeddings = HashMap::new();

        // First, apply intra-modal self-attention for each modality
        for (modality, embedding) in modality_embeddings {
            if let Some(attn) = self.intra_attention.get(modality) {
                // Apply intra-modal attention (placeholder)
                let updated = self.apply_intra_attention(attn, embedding, batch_size)?;
                updated_embeddings.insert(modality.clone(), updated);
            }
        }

        // Then, apply cross-modal attention
        for cross_attn in &self.cross_attention {
            if let Some(query_embedding) =
                updated_embeddings.get(&cross_attn.query_modality).cloned()
            {
                let attended = self.apply_cross_attention(
                    cross_attn,
                    &query_embedding,
                    modality_embeddings,
                    batch_size,
                )?;
                // Update embedding with cross-attention output
                if let Some(existing) = updated_embeddings.get_mut(&cross_attn.query_modality) {
                    // Residual connection
                    for i in 0..existing.len() {
                        existing[i] += attended[i];
                    }
                }
            }
        }

        // Finally, apply feed-forward networks
        for (modality, embedding) in updated_embeddings.iter_mut() {
            if let Some(ffn) = self.feed_forward.get(modality) {
                let ff_output = self.apply_feed_forward(ffn, embedding)?;
                // Residual connection
                for i in 0..embedding.len() {
                    embedding[i] += ff_output[i];
                }
            }
        }

        Ok(updated_embeddings)
    }

    fn apply_intra_attention(
        &self,
        _attn: &MultiHeadAttention<B, S, T>,
        embedding: &[f32],
        _batch_size: usize,
    ) -> Result<Vec<f32>> {
        // Placeholder: in real implementation, this would call attn.forward()
        Ok(embedding.to_vec())
    }

    fn apply_cross_attention(
        &self,
        _cross_attn: &CrossModalAttention<B, S, T>,
        query: &[f32],
        _all_embeddings: &HashMap<Modality, Vec<f32>>,
        _batch_size: usize,
    ) -> Result<Vec<f32>> {
        // Placeholder: combine query with key/value from other modalities
        Ok(query.to_vec())
    }

    fn apply_feed_forward(
        &self,
        _ffn: &FeedForwardNetwork<B, S, T>,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        // Placeholder: apply feed-forward network
        Ok(input.to_vec())
    }
}

impl<B, S, T> CrossModalAttention<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::FromPrimitive + num_traits::Bounded,
{
    /// Create new cross-modal attention
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        query_modality: Modality,
        kv_modalities: Vec<Modality>,
    ) -> Result<Self> {
        let attention = MultiHeadAttention::new(hidden_dim, num_heads)?;
        let out_proj = Linear::new(hidden_dim, hidden_dim)?;
        let norm = LayerNorm::new(vec![hidden_dim], 1e-6);

        Ok(Self {
            query_modality,
            kv_modalities,
            attention,
            query_proj: None,
            key_proj: None,
            value_proj: None,
            out_proj,
            norm,
            attention_type: CrossAttentionType::Direct,
        })
    }

    /// Configure attention type
    pub fn with_attention_type(mut self, attention_type: CrossAttentionType) -> Self {
        self.attention_type = attention_type;
        self
    }

    /// Forward pass for co-attention
    pub fn forward(&self, _vision: &Vec<f32>, _text: &Vec<f32>) -> Result<(Vec<f32>, Vec<f32>)> {
        // Stub implementation - return dummy outputs
        Ok((vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]))
    }
}

impl<B, S, T> FeedForwardNetwork<B, S, T>
where
    B: Backend<Data = T> + Clone,
    S: Storage<T> + Clone + StorageFromVec<T> + storage::StorageToDense<T> + 'static,
    T: DataType + FloatExt + num_traits::FromPrimitive + 'static,
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

impl<B, S, T> CoAttention<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::Bounded + num_traits::FromPrimitive,
{
    /// Create new co-attention mechanism
    pub fn new(vision_dim: usize, text_dim: usize, hidden_dim: usize) -> Result<Self> {
        Ok(Self {
            vision_dim,
            text_dim,
            hidden_dim,
            symmetric: true,
            v2t_attention: MultiHeadAttention::new(hidden_dim, 8)?,
            t2v_attention: MultiHeadAttention::new(hidden_dim, 8)?,
            fusion: Linear::new(hidden_dim * 2, hidden_dim)?,
        })
    }

    /// Apply co-attention to vision and text features
    pub fn forward(
        &self,
        vision_features: &[f32],
        text_features: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        // Vision-to-text attention
        // let v2t_out = self.v2t_attention.forward(vision_features, text_features, text_features)?;

        // Text-to-vision attention
        // let t2v_out = self.t2v_attention.forward(text_features, vision_features, vision_features)?;

        // Placeholder return
        Ok((vision_features.to_vec(), text_features.to_vec()))
    }
}

impl<B, S, T> MultimodalFusionAttention<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::Bounded + num_traits::FromPrimitive,
{
    /// Create new multimodal fusion attention
    pub fn new(modalities: &[Modality], hidden_dim: usize, num_heads: usize) -> Result<Self> {
        let num_modalities = modalities.len();
        let mut modality_projections = HashMap::new();

        // Create projections for each modality
        for modality in modalities {
            modality_projections.insert(
                modality.clone(),
                Linear::new(hidden_dim, hidden_dim)?, // Assume all modalities project to same dim
            );
        }

        let fusion_attention = MultiHeadAttention::new(hidden_dim, num_heads * num_modalities)?;
        let modality_weights = vec![1.0 / num_modalities as f64; num_modalities];

        Ok(Self {
            num_modalities,
            hidden_dim,
            num_heads,
            modality_projections,
            fusion_attention,
            modality_weights,
            temperature: 1.0,
        })
    }

    /// Forward pass with dynamic modality weighting
    pub fn forward(&self, modality_features: &HashMap<Modality, Vec<f32>>) -> Result<Vec<f32>> {
        // Project all modalities to common space
        let mut projected_features = Vec::new();
        let mut modality_order = Vec::new();

        for (modality, features) in modality_features {
            modality_order.push(modality.clone());
            // Apply projection (placeholder)
            projected_features.push(features.clone());
        }

        // Apply multimodal attention fusion
        // Placeholder: return average of all modalities
        let mut fused = vec![0.0; self.hidden_dim];
        for features in &projected_features {
            for i in 0..fused.len().min(features.len()) {
                fused[i] += features[i] / projected_features.len() as f32;
            }
        }

        Ok(fused)
    }
}

impl<B, S, T> ProgressiveCrossModalIntegration<B, S, T>
where
    B: Backend<Data = T> + Clone + Default + 'static,
    S: Storage<T> + Clone + Default + StorageFromVec<T> + storage::StorageToDense<T>,
    T: DataType + 'static + dtype::FloatExt + num_traits::Bounded + num_traits::FromPrimitive,
{
    /// Create new progressive integration
    pub fn new(modalities: Vec<Modality>, num_stages: usize) -> Result<Self> {
        let integration_order = modalities.clone(); // Simple order
        let mut integration_blocks = Vec::new();

        // Create integration blocks for each stage
        for stage in 0..num_stages {
            let modalities_for_stage = if stage == 0 {
                vec![integration_order[0].clone()]
            } else {
                integration_order[..=(stage.min(integration_order.len() - 1))].to_vec()
            };

            integration_blocks.push(IntegrationBlock {
                modalities: modalities_for_stage,
                cross_attention: Vec::new(), // Would be populated based on stage logic
                fusion: FusionMechanism::AttentionFusion(MultiHeadAttention::new(768, 8)?),
            });
        }

        Ok(Self {
            num_stages,
            integration_blocks,
            integration_order,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multimodal::Modality;
    use backend::CpuBackend;
    use dtype::float::Float32;
    use storage::DenseStorage;

    type B = CpuBackend<Float32>;
    type S = DenseStorage<Float32>;
    type T = Float32;

    #[test]
    fn test_cross_modal_transformer_layer_creation() {
        let modalities = vec![Modality::Vision, Modality::Language];
        let layer = CrossModalTransformerLayer::<B, S, T>::new(
            &modalities,
            768,
            12,
            AttentionPattern::BidirectionalVL,
        );
        assert!(layer.is_ok());
    }

    #[test]
    fn test_cross_modal_attention_creation() {
        let cross_attn = CrossModalAttention::<B, S, T>::new(
            768,
            12,
            Modality::Language,
            vec![Modality::Vision],
        );
        assert!(cross_attn.is_ok());
    }

    #[test]
    fn test_co_attention_creation() {
        let co_attn = CoAttention::<B, S, T>::new(768, 768, 768);
        assert!(co_attn.is_ok());
    }

    #[test]
    fn test_multimodal_fusion_attention_creation() {
        let modalities = vec![Modality::Vision, Modality::Language, Modality::Audio];
        let mfa = MultimodalFusionAttention::<B, S, T>::new(&modalities, 768, 8);
        assert!(mfa.is_ok());
    }

    #[test]
    fn test_progressive_integration_creation() {
        let modalities = vec![Modality::Vision, Modality::Language];
        let pcmi = ProgressiveCrossModalIntegration::<B, S, T>::new(modalities, 3);
        assert!(pcmi.is_ok());
    }
}
