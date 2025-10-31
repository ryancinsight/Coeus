//! Complete working example demonstrating unified multimodal transformer
//! from Sprint MS-54 Advanced Multimodal AI with Temporal Reasoning

use std::collections::HashMap;
use nn::multimodal::{
    MultimodalTransformer, MultimodalConfig, Task, Classifier,
    Generator, Retriever, Modality, FusionStrategy, SimilarityType
};
use nn::linear::Linear;
use nn::error::Result;
use tensor::{Tensor, backend::CpuBackend, storage::DenseStorage, DataType::Float32};

/// Demonstrates complete end-to-end multimodal processing pipeline
fn main() -> Result<()> {
    println!("🚀 Sprint MS-54: Advanced Multimodal AI with Temporal Reasoning");
    println!("===========================================================");
    demo_unified_multimodal_transformer()?;
    demo_multimodal_classification()?;
    demo_multimodal_retrieval()?;
    demo_multimodal_generation()?;
    demo_model_analysis()?;

    println!("\n✅ All multimodal AI components successfully demonstrated!");
    println!("🏆 Sprint MS-54 Goals: Complete temporal multimodal AI system");
    Ok(())
}

/// Demonstrate unified multimodal transformer with Vision + Language + Audio
fn demo_unified_multimodal_transformer() -> Result<()> {
    println!("\n🎯 Unified Multimodal Transformer Demo");

    // Configure multimodal transformer for vision, language, and audio
    let config = MultimodalConfig {
        modalities: vec![Modality::Vision, Modality::Language, Modality::Audio],
        hidden_dim: 768,
        num_fusion_layers: 6,
        fusion_strategy: FusionStrategy::HierarchicalFusion,
        dropout: 0.1,
    };

    let mut transformer = MultimodalTransformer::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config)?;

    // Add task-specific heads
    transformer.add_task(
        "image_captioning".to_string(),
        Task::Generation(Generator {
            lm_head: Linear::new(768, 50000)?,
            vocab_size: 50000,
        })
    )?;

    // Create realistic multimodal inputs with proper tensor shapes
    let device = CpuBackend::<Float32>::default();

    // Vision: [batch=1, seq=197, hidden=2048] (simulated CLIP features, 197 patches)
    let vision_features = Tensor::randn(&[1, 197, 2048], &device)?;

    // Language: [batch=1, seq=128, hidden=768] (simulated BERT embeddings)
    let text_features = Tensor::randn(&[1, 128, 768], &device)?;

    // Audio: [batch=1, seq=100, hidden=1024] (simulated audio features)
    let audio_features = Tensor::randn(&[1, 100, 1024], &device)?;

    let inputs = HashMap::from([
        (Modality::Vision, vision_features),
        (Modality::Language, text_features),
        (Modality::Audio, audio_features),
    ]);

    // Forward pass for image captioning task (no mask for simplicity)
    let output = transformer.forward(&inputs, "image_captioning", None)?;

    println!("  📝 Generated captions: shape {:?}", output.shape());
    println!("  🔍 Cross-modal fusion completed with {} modalities", transformer.config.modalities.len());
    println!("  🧠 Model parameters: {}", transformer.num_parameters());

    Ok(())
}

/// Demonstrate multimodal classification task
fn demo_multimodal_classification() -> Result<()> {
    println!("\n🎯 Multimodal Classification Demo");

    let config = MultimodalConfig {
        modalities: vec![Modality::Vision, Modality::Language],
        hidden_dim: 512,
        num_fusion_layers: 4,
        fusion_strategy: FusionStrategy::HierarchicalFusion,
        dropout: 0.1,
    };

    let mut transformer = MultimodalTransformer::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config)?;

    // Add classification head for multimodal sentiment analysis
    transformer.add_task(
        "sentiment_classification".to_string(),
        Task::Classification(Classifier {
            classifier: Linear::new(512, 3)?, // 3 classes: positive, negative, neutral
            num_classes: 3,
        })
    )?;

    let device = CpuBackend::<Float32>::default();
    let image_features = Tensor::randn(&[1, 49, 2048], &device)?; // 7x7 patches
    let text_features = Tensor::randn(&[1, 64, 768], &device)?;   // Text sequence

    let inputs = HashMap::from([
        (Modality::Vision, image_features),
        (Modality::Language, text_features),
    ]);

    let output = transformer.forward(&inputs, "sentiment_classification", None)?;
    println!("  📊 Multimodal classification completed: shape {:?}", output.shape());
    println!("  🎯 3-class sentiment classification (positive/negative/neutral)");

    Ok(())
}


/// Demonstrate multimodal retrieval task
fn demo_multimodal_retrieval() -> Result<()> {
    println!("\n🎯 Multimodal Retrieval Demo");

    let config = MultimodalConfig {
        modalities: vec![Modality::Vision, Modality::Language],
        hidden_dim: 512,
        num_fusion_layers: 3,
        fusion_strategy: FusionStrategy::AttentionFusion,
        dropout: 0.1,
    };

    let mut transformer = MultimodalTransformer::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config)?;

    // Add retrieval head for image-text matching
    transformer.add_task(
        "image_text_retrieval".to_string(),
        Task::Retrieval(Retriever {
            projection: Linear::new(512, 512)?,
            similarity_type: SimilarityType::Cosine,
        })
    )?;

    let device = CpuBackend::<Float32>::default();
    let image_features = Tensor::randn(&[1, 49, 2048], &device)?;
    let text_features = Tensor::randn(&[1, 32, 768], &device)?;

    let inputs = HashMap::from([
        (Modality::Vision, image_features),
        (Modality::Language, text_features),
    ]);

    let output = transformer.forward(&inputs, "image_text_retrieval", None)?;
    println!("  🔍 Multimodal retrieval completed: shape {:?}", output.shape());
    println!("  📏 Cosine similarity-based retrieval embeddings");

    Ok(())
}

/// Demonstrate multimodal generation task
fn demo_multimodal_generation() -> Result<()> {
    println!("\n🎯 Multimodal Generation Demo");

    let config = MultimodalConfig {
        modalities: vec![Modality::Vision, Modality::Audio, Modality::Language],
        hidden_dim: 768,
        num_fusion_layers: 5,
        fusion_strategy: FusionStrategy::HierarchicalFusion,
        dropout: 0.1,
    };

    let mut transformer = MultimodalTransformer::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config)?;

    // Add generation head for text generation from image+audio
    transformer.add_task(
        "image_audio_to_text".to_string(),
        Task::Generation(Generator {
            lm_head: Linear::new(768, 30000)?, // 30K vocabulary
            vocab_size: 30000,
        })
    )?;

    let device = CpuBackend::<Float32>::default();
    let image_features = Tensor::randn(&[1, 197, 2048], &device)?;
    let audio_features = Tensor::randn(&[1, 200, 1024], &device)?;
    let text_features = Tensor::randn(&[1, 50, 768], &device)?; // Context text

    let inputs = HashMap::from([
        (Modality::Vision, image_features),
        (Modality::Audio, audio_features),
        (Modality::Language, text_features),
    ]);

    let output = transformer.forward(&inputs, "image_audio_to_text", None)?;
    println!("  🎵 Multimodal generation completed: shape {:?}", output.shape());
    println!("  📝 Generated text: sequence of {} tokens", output.shape()[1]);

    Ok(())
}

/// Demonstrate model analysis capabilities
fn demo_model_analysis() -> Result<()> {
    println!("\n🎯 Model Analysis Demo");

    let config = MultimodalConfig {
        modalities: vec![Modality::Vision, Modality::Language, Modality::Audio],
        hidden_dim: 768,
        num_fusion_layers: 6,
        fusion_strategy: FusionStrategy::HierarchicalFusion,
        dropout: 0.1,
    };

    let transformer = MultimodalTransformer::<CpuBackend<Float32>, DenseStorage<Float32>, Float32>::new(config)?;

    println!("  📊 Model Architecture Analysis:");
    println!("    - Modal Encoder → Transformer Layers → Cross-Modal Fusion → Task Heads");
    println!("    - 3 modalities: Vision, Language, Audio");
    println!("    - {} hidden dimensions", transformer.config.hidden_dim);
    println!("    - {} fusion layers", transformer.config.num_fusion_layers);
    println!("    - Total parameters: {}", transformer.num_parameters());
    println!("    - Hierarchical fusion strategy");

    // Analyze per-modality computing
    println!("\n  🔬 Per-Modality Analysis:");
    for modality in &transformer.config.modalities {
        if let Some(encoder) = transformer.encoders.get(modality) {
            println!("    - {} encoder: {} layers, {} parameters",
                     modality.as_str(), encoder.config.num_layers, encoder.num_parameters());
        }
    }

    Ok(())
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_multimodal_pipeline() -> Result<()> {
        demo_unified_multimodal_transformer()?;
        demo_multimodal_classification()?;
        demo_multimodal_retrieval()?;
        demo_multimodal_generation()?;
        demo_model_analysis()?;
        Ok(())
    }
}
